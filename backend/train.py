"""
Self-play temporal-difference (TD) training for the AlphaBeta engine.

Two bots with different starting "personalities" play each other many
times. After each game, both update their evaluation weights using TD(0)
learning: each position's predicted score is nudged toward the score of
the position that followed it (bootstrapping), and the terminal position
is nudged toward the actual game result.

The value function is linear in a small feature vector:

    V(s) = w0*pawn_diff + w1*knight_diff + ... + w5*queen_diff + w6*bishop_pair

where "diff" means white count minus black count (from white's
perspective). TD updates each weight by the feature value times the
temporal-difference error:

    w_i  +=  alpha * (target - V(s)) * feature_i(s)

This is genuine machine learning: no labels, the signal is the outcome of
self-play games. Run it for thousands of games and the engine's material
values converge on the values that best predict winning positions.

Usage:
    python train.py --games 2000 --depth 2 --checkpoint 200
"""

import argparse
import json
import math
import os
import random
import time

import chess

import engine

# Feature index -> chess piece type (king omitted from features; its value
# is always 0 and it is never captured).
FEATURE_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
NUM_MATERIAL_FEATURES = len(FEATURE_PIECES)
BISHOP_PAIR_IDX = NUM_MATERIAL_FEATURES  # index of the bishop-pair feature


def material_weights_dict(weights: list) -> dict:
    """Map the feature list back onto piece types for engine.set_eval_weights."""
    result = {pt: weights[i] for i, pt in enumerate(FEATURE_PIECES)}
    result[chess.KING] = 0
    return result


def squash(value: float) -> float:
    """Map a centipawn evaluation into [-1, 1] using tanh.

    The TD target (game result) lives in [-1, 1], but engine.evaluate()
    returns centipawns with magnitudes of a couple thousand. Feeding raw
    centipawns into the error term produces updates of order alpha*2000,
    which slam the weights into their clipping bounds in a few games and
    destroy learning. Squashing the value first keeps the error in [-2, 2].
    """
    return math.tanh(value / 1000.0)


def feature_vector(board: chess.Board) -> list:
    """Material-count differences (white - black) plus bishop-pair feature."""
    counts = {pt: 0 for pt in FEATURE_PIECES}
    for square, piece in board.piece_map().items():
        if piece.piece_type in counts:
            counts[piece.piece_type] += 1 if piece.color == chess.WHITE else -1
    features = [float(counts[pt]) for pt in FEATURE_PIECES]

    white_bishops = counts[chess.BISHOP]
    black_bishops = -counts[chess.BISHOP]
    bishop_pair = (1 if white_bishops >= 2 else 0) - (1 if black_bishops >= 2 else 0)
    features.append(float(bishop_pair))
    return features


def apply_weights(weights: list) -> None:
    engine.set_eval_weights(
        material_weights_dict(weights), bishop_pair=int(weights[BISHOP_PAIR_IDX])
    )


def default_weights() -> list:
    return [
        float(engine.PIECE_VALUES[chess.PAWN]),
        float(engine.PIECE_VALUES[chess.KNIGHT]),
        float(engine.PIECE_VALUES[chess.BISHOP]),
        float(engine.PIECE_VALUES[chess.ROOK]),
        float(engine.PIECE_VALUES[chess.QUEEN]),
        float(engine.BISHOP_PAIR_BONUS),
    ]


def save_weights(path: str, weights: list) -> None:
    data = {
        "weights": weights,
        "pawn": weights[0],
        "knight": weights[1],
        "bishop": weights[2],
        "rook": weights[3],
        "queen": weights[4],
        "bishop_pair": weights[5],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_weights(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return [float(x) for x in data["weights"]]


def play_game(weights_a: list, weights_b: list, depth: int, explore: float) -> tuple:
    """Play a self-play game. Bot A is white, bot B is black.

    Returns (states, result) where states is a list of dicts with the
    feature vector and full engine evaluation (white's perspective) at
    each position before a move, and result is +1 (white wins),
    -1 (black wins) or 0 (draw).
    """
    board = chess.Board()
    states = []
    ply = 0

    while not board.is_game_over():
        # Apply the mover's weights first so the recorded value reflects
        # the bot that will actually search this position. evaluate() is
        # always from white's perspective, which is consistent for both.
        if board.turn == chess.WHITE:
            weights = weights_a
        else:
            weights = weights_b
        apply_weights(weights)

        states.append(
            {
                "features": feature_vector(board),
                "value": squash(engine.evaluate(board)),
            }
        )

        # Exploration: occasionally play a random move so games vary.
        legal = list(board.legal_moves)
        if random.random() < explore:
            move = random.choice(legal)
        else:
            move, _ = engine.find_best_move(board, max_depth=depth, time_limit=30)
            if move is None:
                break
        board.push(move)
        ply += 1

    if board.is_checkmate():
        result = 1.0 if board.turn == chess.BLACK else -1.0
    elif (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_draw()
    ):
        result = 0.0
    else:
        result = 0.0

    return states, result


# Sane bounds per material feature, so weights can't explode.
WEIGHT_MIN = [50, 180, 180, 300, 600, -50]
WEIGHT_MAX = [200, 450, 450, 700, 1200, 80]


def td_update(
    weights: list, states: list, result: float, alpha: float, gamma: float
) -> list:
    """One TD(0) pass over a game: nudge weights toward observed outcomes.

    Semi-gradient TD(0) on the engine's full evaluation: each position's
    value is bootstrapped from the value of the position that followed,
    and the final position is anchored to the actual game result. Only the
    material (and bishop-pair) weights are learnable — the piece-square
    tables act as a fixed baseline. Weights are clipped to sane ranges.
    """
    n = len(states)
    values = [s["value"] for s in states]

    for t in range(n):
        if t + 1 < n:
            target = gamma * values[t + 1]
        else:
            target = result
        error = target - values[t]
        f = states[t]["features"]
        for i in range(len(weights)):
            weights[i] += alpha * error * f[i]
            weights[i] = max(WEIGHT_MIN[i], min(WEIGHT_MAX[i], weights[i]))
    return weights


def play_match(weights_a: list, weights_b: list, games: int, depth: int) -> dict:
    """A vs B, alternating colors. Returns {a_wins, b_wins, draws}."""
    stats = {"a_wins": 0, "b_wins": 0, "draws": 0}
    for g in range(games):
        a_is_white = g % 2 == 0
        board = chess.Board()
        while not board.is_game_over():
            white_turn = board.turn == chess.WHITE
            if white_turn:
                w = weights_a if a_is_white else weights_b
            else:
                w = weights_b if a_is_white else weights_a
            apply_weights(w)
            move, _ = engine.find_best_move(board, max_depth=depth, time_limit=30)
            if move is None:
                break
            board.push(move)
        stats = count_result(stats, board, a_is_white)
    return stats


def count_result(stats: dict, board: chess.Board, a_is_white: bool) -> dict:
    """Record a finished game for player A (white iff a_is_white)."""
    if board.is_checkmate():
        white_mated = board.turn == chess.WHITE
        # A wins when the opponent's king is mated.
        if white_mated and not a_is_white:
            stats["a_wins"] += 1
        elif not white_mated and a_is_white:
            stats["a_wins"] += 1
        else:
            stats["b_wins"] += 1
    else:
        stats["draws"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description="TD self-play training")
    parser.add_argument("--games", type=int, default=1000, help="total self-play games")
    parser.add_argument("--depth", type=int, default=2, help="search depth per move")
    parser.add_argument("--alpha", type=float, default=0.02, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--explore", type=float, default=0.15, help="random-move probability"
    )
    parser.add_argument(
        "--checkpoint", type=int, default=100, help="games between checkpoints"
    )
    parser.add_argument(
        "--out", default="trained_weights.json", help="output weight file"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Two personalities: bot A starts from classic values, bot B is
    # materialistic (values heavy pieces more, minor pieces less).
    weights_a = default_weights()
    weights_b = default_weights()
    weights_b[3] *= 1.2  # rook
    weights_b[4] *= 1.1  # queen
    weights_b[1] *= 0.9  # knight
    weights_b[2] *= 0.9  # bishop

    print(f"Training {args.games} self-play games at depth {args.depth}")
    print(f"Bot A (classic):  {[round(w, 1) for w in weights_a]}")
    print(f"Bot B (greedy):   {[round(w, 1) for w in weights_b]}")
    print(f"alpha={args.alpha} gamma={args.gamma} explore={args.explore}\n")

    best_score = -1.0
    t0 = time.time()

    for game_idx in range(1, args.games + 1):
        # Alternate who is white so both learn both sides.
        if game_idx % 2 == 0:
            w_white, w_black = weights_a, weights_b
        else:
            w_white, w_black = weights_b, weights_a

        states, result = play_game(w_white, w_black, args.depth, args.explore)

        # TD update for BOTH bots from the same game. evaluate() is always
        # from white's perspective, so both learn the same target.
        w_white = td_update(w_white, states, result, args.alpha, args.gamma)
        w_black = td_update(w_black, states, result, args.alpha, args.gamma)

        if game_idx % args.checkpoint == 0:
            # Evaluate trained bot A against the fixed baseline.
            baseline = default_weights()
            score = evaluate_bot(weights_a, baseline, games=20, depth=args.depth)
            elapsed = time.time() - t0
            print(
                f"[game {game_idx:>6}] A win% vs baseline: {score:.1f}  "
                f"A={[round(w, 1) for w in weights_a]}  ({elapsed:.0f}s)"
            )
            if score > best_score:
                best_score = score
                save_weights(args.out, weights_a)

    # Final checkpoint & save.
    baseline = default_weights()
    final_score = evaluate_bot(weights_a, baseline, games=40, depth=args.depth)
    save_weights(args.out, weights_a)
    print(f"\nDone. Bot A final win% vs baseline: {final_score:.1f}")
    print(f"Saved weights to {args.out}:")
    print(
        f"  pawn={weights_a[0]:.1f} knight={weights_a[1]:.1f} bishop={weights_a[2]:.1f} "
        f"rook={weights_a[3]:.1f} queen={weights_a[4]:.1f} bishop_pair={weights_a[5]:.1f}"
    )


def evaluate_bot(weights: list, baseline: list, games: int, depth: int) -> float:
    """Win percentage of `weights` vs `baseline` over `games` games."""
    wins = 0.0
    for g in range(games):
        a_is_white = g % 2 == 0
        board = chess.Board()
        while not board.is_game_over():
            white_turn = board.turn == chess.WHITE
            if white_turn:
                w = weights if a_is_white else baseline
            else:
                w = baseline if a_is_white else weights
            apply_weights(w)
            move, _ = engine.find_best_move(board, max_depth=depth, time_limit=30)
            if move is None:
                break
            board.push(move)
        stats = count_result({"a_wins": 0, "b_wins": 0, "draws": 0}, board, a_is_white)
        wins += stats["a_wins"] + 0.5 * stats["draws"]
    return wins / games * 100.0


if __name__ == "__main__":
    main()
