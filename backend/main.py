from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import chess
import os
import uuid
import threading
from typing import Optional

from engine import find_best_move, evaluate, set_network_eval

app = FastAPI(title="Alpha-Beta Chess Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_DIFFICULTIES = {1, 2, 3, 4}
DIFFICULTY_DEPTH = {1: 2, 2: 3, 3: 4, 4: 5}
VALID_COLORS = {"white", "black"}

# Optional trained neural network. Set CHESS_MODEL=/path/to/nn.pt to have
# the engine evaluate positions with the network instead of the classic
# material + piece-square evaluation. Loaded lazily so the server still
# starts without torch installed.
MODEL_PATH = os.environ.get("CHESS_MODEL", "")
EVALUATOR_NAME = "classic (material + piece-square tables)"


def load_model() -> None:
    """Load the trained value network and route evaluations through it."""
    global EVALUATOR_NAME
    if not MODEL_PATH:
        return
    try:
        import torch
        from nn import ValueNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = ValueNet().to(device)
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        net.eval()
        set_network_eval(net.value)
        EVALUATOR_NAME = f"neural net ({MODEL_PATH}, {device})"
    except Exception as exc:  # noqa: BLE001 - log and fall back to classic eval
        print(f"[startup] failed to load model {MODEL_PATH}: {exc}", flush=True)


load_model()


class MoveRequest(BaseModel):
    move: str

    @field_validator("move")
    @classmethod
    def validate_move_format(cls, v: str) -> str:
        if not v or len(v) < 4 or len(v) > 5:
            raise ValueError(
                "Move must be 4-5 characters in UCI format (e.g. e2e4, e7e8q)"
            )
        return v


class NewGameRequest(BaseModel):
    difficulty: int = 3
    player_color: str = "white"

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: int) -> int:
        if v not in VALID_DIFFICULTIES:
            raise ValueError(f"Difficulty must be one of {sorted(VALID_DIFFICULTIES)}")
        return v

    @field_validator("player_color")
    @classmethod
    def validate_color(cls, v: str) -> str:
        if v not in VALID_COLORS:
            raise ValueError(f"player_color must be one of {sorted(VALID_COLORS)}")
        return v


class Game:
    def __init__(self, difficulty: int = 3, player_color: str = "white"):
        self.id = str(uuid.uuid4())[:8]
        self.board = chess.Board()
        self.difficulty = difficulty
        self.player_color = player_color
        self.lock = threading.Lock()


games: dict[str, Game] = {}
games_lock = threading.Lock()

default_game = Game()
games[default_game.id] = default_game


def board_state(game: Game) -> dict:
    b = game.board
    is_draw = False
    draw_reason = None

    if b.is_stalemate():
        is_draw = True
        draw_reason = "stalemate"
    elif b.halfmove_clock >= 100:
        is_draw = True
        draw_reason = "50-move rule"
    elif b.is_repetition(3):
        is_draw = True
        draw_reason = "threefold repetition"
    elif b.is_insufficient_material():
        is_draw = True
        draw_reason = "insufficient material"

    return {
        "game_id": game.id,
        "fen": b.fen(),
        "turn": "white" if b.turn == chess.WHITE else "black",
        "is_check": b.is_check(),
        "is_checkmate": b.is_checkmate(),
        "is_stalemate": b.is_stalemate(),
        "is_draw": is_draw,
        "draw_reason": draw_reason,
        "is_game_over": b.is_game_over(),
        "legal_moves": [m.uci() for m in b.legal_moves],
        "move_stack": [m.uci() for m in b.move_stack],
        "result": b.result() if b.is_game_over() else None,
        "player_color": game.player_color,
        "difficulty": game.difficulty,
    }


def get_game(game_id: Optional[str] = None) -> Game:
    if game_id:
        with games_lock:
            game = games.get(game_id)
        if not game:
            raise HTTPException(404, f"Game {game_id} not found")
        return game
    with games_lock:
        return list(games.values())[-1]


@app.post("/api/new-game")
def new_game(req: NewGameRequest):
    game = Game(difficulty=req.difficulty, player_color=req.player_color)
    with games_lock:
        games[game.id] = game
        if len(games) > 50:
            oldest = next(iter(games))
            del games[oldest]

    with game.lock:
        state = board_state(game)

        if req.player_color == "black":
            depth = DIFFICULTY_DEPTH[req.difficulty]
            best_move, info = find_best_move(
                game.board, max_depth=depth, time_limit=5.0
            )
            if best_move:
                game.board.push(best_move)
            state = board_state(game)
            state["engine_move"] = best_move.uci() if best_move else None
            state["engine_info"] = info

    return state


@app.get("/api/state")
def get_state(game_id: Optional[str] = None):
    game = get_game(game_id)
    return board_state(game)


@app.post("/api/move")
def player_move(req: MoveRequest, game_id: Optional[str] = None):
    game = get_game(game_id)

    with game.lock:
        if game.board.is_game_over():
            raise HTTPException(409, "Game is already over")

        player_is_white = game.player_color == "white"
        if (player_is_white and game.board.turn != chess.WHITE) or (
            not player_is_white and game.board.turn != chess.BLACK
        ):
            raise HTTPException(409, "Not your turn")

        try:
            move = chess.Move.from_uci(req.move)
        except ValueError:
            raise HTTPException(400, f"Malformed move string: {req.move}")

        if move not in game.board.legal_moves:
            raise HTTPException(422, f"Illegal move: {req.move}")

        game.board.push(move)
        return board_state(game)


@app.post("/api/engine-move")
def engine_move(game_id: Optional[str] = None):
    game = get_game(game_id)

    with game.lock:
        if game.board.is_game_over():
            return board_state(game)

        engine_is_white = game.player_color == "black"
        if (engine_is_white and game.board.turn != chess.WHITE) or (
            not engine_is_white and game.board.turn != chess.BLACK
        ):
            raise HTTPException(409, "Not engine's turn")

        depth = DIFFICULTY_DEPTH[game.difficulty]
        best_move, info = find_best_move(game.board, max_depth=depth, time_limit=5.0)

        if best_move is None:
            raise HTTPException(400, "No legal moves for engine")

        game.board.push(best_move)
        state = board_state(game)
        state["engine_move"] = best_move.uci()
        state["engine_info"] = info
        return state


@app.post("/api/undo")
def undo_move(game_id: Optional[str] = None):
    game = get_game(game_id)

    with game.lock:
        if len(game.board.move_stack) >= 2:
            game.board.pop()
            game.board.pop()
        elif len(game.board.move_stack) == 1:
            game.board.pop()
        return board_state(game)


@app.get("/api/eval")
def get_eval(game_id: Optional[str] = None):
    game = get_game(game_id)
    return {"score": evaluate(game.board)}


@app.get("/api/engine")
def engine_info():
    """Which evaluator is active and how difficulty maps to search depth."""
    return {"evaluator": EVALUATOR_NAME, "difficulty_depths": DIFFICULTY_DEPTH}


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
