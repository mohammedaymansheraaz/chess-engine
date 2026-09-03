from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from typing import Optional

import chess
import os
import threading
import uuid

from engine import find_best_move, evaluate, set_network_eval
import style

app = FastAPI(title="Alpha-Beta Chess Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

VALID_DIFFICULTIES = {1, 2, 3, 4}
DEPTH_BY_DIFFICULTY = {1: 2, 2: 3, 3: 4, 4: 5}
VALID_COLORS = {"white", "black"}

MODEL_PATH = os.environ.get("CHESS_MODEL", "")
EVALUATOR_NAME = "classic"


def load_model():
    global EVALUATOR_NAME

    if not MODEL_PATH:
        return

    try:
        import torch
        from nn import ValueNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ValueNet().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        set_network_eval(model.value)
        EVALUATOR_NAME = f"neural net ({MODEL_PATH}, {device})"
    except Exception as exc:
        print(f"Could not load model {MODEL_PATH}: {exc}", flush=True)


load_model()


class MoveRequest(BaseModel):
    move: str

    @field_validator("move")
    @classmethod
    def check_move(cls, value: str) -> str:
        if len(value) not in (4, 5):
            raise ValueError("Move must be in UCI format, for example e2e4")
        return value


class NewGameRequest(BaseModel):
    difficulty: int = 3
    player_color: str = "white"

    @field_validator("difficulty")
    @classmethod
    def check_difficulty(cls, value: int) -> int:
        if value not in VALID_DIFFICULTIES:
            raise ValueError(f"Difficulty must be one of {sorted(VALID_DIFFICULTIES)}")
        return value

    @field_validator("player_color")
    @classmethod
    def check_color(cls, value: str) -> str:
        if value not in VALID_COLORS:
            raise ValueError(f"player_color must be one of {sorted(VALID_COLORS)}")
        return value


class Game:
    def __init__(self, difficulty=3, player_color="white"):
        self.id = str(uuid.uuid4())[:8]
        self.board = chess.Board()
        self.difficulty = difficulty
        self.player_color = player_color
        self.lock = threading.Lock()


games = {}
games_lock = threading.Lock()

default_game = Game()
games[default_game.id] = default_game


def board_state(game: Game):
    board = game.board
    draw_reason = None

    if board.is_stalemate():
        draw_reason = "stalemate"
    elif board.halfmove_clock >= 100:
        draw_reason = "50-move rule"
    elif board.is_repetition(3):
        draw_reason = "threefold repetition"
    elif board.is_insufficient_material():
        draw_reason = "insufficient material"

    return {
        "game_id": game.id,
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_draw": draw_reason is not None,
        "draw_reason": draw_reason,
        "is_game_over": board.is_game_over(),
        "legal_moves": [move.uci() for move in board.legal_moves],
        "move_stack": [move.uci() for move in board.move_stack],
        "result": board.result() if board.is_game_over() else None,
        "player_color": game.player_color,
        "difficulty": game.difficulty,
    }


def get_game(game_id: Optional[str] = None):
    with games_lock:
        if game_id:
            game = games.get(game_id)
        else:
            game = list(games.values())[-1]

    if game is None:
        raise HTTPException(404, f"Game {game_id} not found")

    return game


@app.post("/api/new-game")
def new_game(req: NewGameRequest):
    game = Game(req.difficulty, req.player_color)

    with games_lock:
        games[game.id] = game
        if len(games) > 50:
            oldest_id = next(iter(games))
            del games[oldest_id]

    with game.lock:
        state = board_state(game)

        if req.player_color == "black":
            depth = DEPTH_BY_DIFFICULTY[req.difficulty]
            best_move, info = find_best_move(
                game.board,
                max_depth=depth,
                time_limit=5.0,
            )
            if best_move:
                game.board.push(best_move)

            state = board_state(game)
            state["engine_move"] = best_move.uci() if best_move else None
            state["engine_info"] = info

    return state


@app.get("/api/state")
def get_state(game_id: Optional[str] = None):
    return board_state(get_game(game_id))


@app.post("/api/move")
def player_move(req: MoveRequest, game_id: Optional[str] = None):
    game = get_game(game_id)

    with game.lock:
        board = game.board

        if board.is_game_over():
            raise HTTPException(409, "Game is already over")

        player_is_white = game.player_color == "white"
        player_turn = chess.WHITE if player_is_white else chess.BLACK
        if board.turn != player_turn:
            raise HTTPException(409, "Not your turn")

        try:
            move = chess.Move.from_uci(req.move)
        except ValueError:
            raise HTTPException(400, f"Malformed move string: {req.move}")

        if move not in board.legal_moves:
            raise HTTPException(422, f"Illegal move: {req.move}")

        board.push(move)
        return board_state(game)


@app.post("/api/engine-move")
def engine_move(game_id: Optional[str] = None):
    game = get_game(game_id)

    with game.lock:
        board = game.board

        if board.is_game_over():
            return board_state(game)

        engine_is_white = game.player_color == "black"
        engine_turn = chess.WHITE if engine_is_white else chess.BLACK
        if board.turn != engine_turn:
            raise HTTPException(409, "Not engine's turn")

        depth = DEPTH_BY_DIFFICULTY[game.difficulty]
        best_move, info = find_best_move(board, max_depth=depth, time_limit=5.0)
        if best_move is None:
            raise HTTPException(400, "No legal moves for engine")

        board.push(best_move)

        state = board_state(game)
        state["engine_move"] = best_move.uci()
        state["engine_info"] = info
        return state


@app.post("/api/undo")
def undo_move(game_id: Optional[str] = None):
    game = get_game(game_id)

    with game.lock:
        moves = game.board.move_stack
        if len(moves) >= 2:
            game.board.pop()
            game.board.pop()
        elif moves:
            game.board.pop()

        return board_state(game)


@app.get("/api/eval")
def get_eval(game_id: Optional[str] = None):
    return {"score": evaluate(get_game(game_id).board)}


@app.get("/api/engine")
def engine_info():
    return {
        "evaluator": EVALUATOR_NAME,
        "difficulty_depths": DEPTH_BY_DIFFICULTY,
    }


@app.get("/api/style")
def get_style():
    return style.current_style_weights()


@app.post("/api/style")
def set_style(
    preset: Optional[str] = None,
    aggression: float = 0.0,
    positional: float = 0.0,
    risk: float = 0.0,
):
    if preset:
        if preset not in style.DEFAULT_WEIGHTS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown preset {preset!r}; choose from {list(style.DEFAULT_WEIGHTS)}",
            )
        return style.set_style_preset(preset)

    return style.set_style_weights(
        aggression=aggression,
        positional=positional,
        risk_avoid=risk,
    )


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
