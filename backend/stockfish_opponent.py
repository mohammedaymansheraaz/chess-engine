import argparse
import math
import os

import chess
import chess.engine
import torch

import nn_train
from nn import ValueNet


SF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "stockfish", "stockfish16"
)


def open_sf(skill: int | None = None, threads: int = 2, hash_mb: int = 64):
    eng = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    eng.configure({"Threads": threads, "Hash": hash_mb})
    if skill is not None:
        eng.configure({"Skill Level": skill})
    return eng


def play_one_game(
    net,
    sf,
    net_white: bool,
    depth: int | None,
    time_limit: float | None,
    sf_skill: int | None,
):
    board = chess.Board()
    net_eval = lambda b: net.value(b)

    while not board.is_game_over():
        if (board.turn == chess.WHITE) == net_white:
            move = nn_train.batched_minimax_move(board, depth, net_eval)
        else:
            if time_limit is not None:
                result = sf.play(board, chess.engine.Limit(time=time_limit))
            else:
                result = sf.play(board, chess.engine.Limit(depth=depth))
            move = result.move

        if move is None:
            break
        board.push(move)

    if board.is_checkmate():
        winner_white = board.turn == chess.BLACK
        return "net" if winner_white == net_white else "sf"
    return "draw"


def play_match_vs_stockfish(
    net,
    games: int = 20,
    depth: int = 2,
    sf_skill: int | None = None,
    sf_time: float | None = None,
):
    net.eval()
    sf = open_sf(skill=sf_skill)
    stats = {"net_wins": 0, "sf_wins": 0, "draws": 0}

    try:
        for g in range(games):
            net_white = g % 2 == 0
            result = play_one_game(net, sf, net_white, depth, sf_time, sf_skill)
            if result == "net":
                stats["net_wins"] += 1
            elif result == "sf":
                stats["sf_wins"] += 1
            else:
                stats["draws"] += 1
    finally:
        sf.quit()

    return stats


def estimate_elo_from_match(
    net_wins: int, sf_wins: int, draws: int, sf_elo_at_handicap: float
):
    n = net_wins + sf_wins + draws
    if n == 0:
        return None

    score = (net_wins + 0.5 * draws) / n
    score = max(1e-3, min(1 - 1e-3, score))
    delta = 400.0 * math.log10(score / (1 - score))
    return sf_elo_at_handicap + delta


def estimate_elo(
    net,
    depth: int = 2,
    sf_skill: int | None = None,
    sf_time: float | None = None,
    games: int = 20,
    sf_elo_at_handicap: float = 2000.0,
):
    stats = play_match_vs_stockfish(net, games, depth, sf_skill, sf_time)
    elo = estimate_elo_from_match(
        stats["net_wins"], stats["sf_wins"], stats["draws"], sf_elo_at_handicap
    )
    return stats, elo


def main():
    parser = argparse.ArgumentParser(description="Match neural net vs Stockfish")
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--depth", type=int, default=2, help="net search depth")
    parser.add_argument("--sf-skill", type=int, default=None, help="handicap skill level")
    parser.add_argument("--sf-time", type=float, default=None, help="per-move SF time (s)")
    parser.add_argument(
        "--sf-elo-anchor",
        type=float,
        default=2000.0,
        help="assumed SF Elo at this handicap",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hash", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ValueNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    print(f"loaded {args.model} on {device}")

    stats, elo = estimate_elo(
        net, args.depth, args.sf_skill, args.sf_time, args.games, args.sf_elo_anchor
    )
    print(
        f"vs Stockfish (skill={args.sf_skill}, time={args.sf_time}, "
        f"depth={args.depth}):"
    )
    print(f"  net wins : {stats['net_wins']}")
    print(f"  sf wins  : {stats['sf_wins']}")
    print(f"  draws   : {stats['draws']}")
    print(f"  estimated Elo: {elo:.0f} (anchor SF={args.sf_elo_anchor})")


if __name__ == "__main__":
    main()
