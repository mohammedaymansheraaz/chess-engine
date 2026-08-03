"""
Head-to-head match: trained neural net vs the classic hand-written engine.

Both sides use the SAME batched full-width minimax at the same depth; only
the evaluator differs (neural net vs material + PST). Running the neural
evaluator unbatched through alpha-beta quiescence is ~1000x slower than the
classic one, so both sides share the batched search to keep matches fast.
Run it on any trained model, or compare snapshots from different training
checkpoints to watch the improvement rate.

Usage:
    python match.py --model nn.pt --games 40 --depth 3
    python match.py --model nn.pt --model2 snapshots/model_4000.pt --games 20
"""

import argparse
import chess
import torch

import nn_train
from nn import ValueNet


def play_game(board, white_eval, black_eval, depth):
    """Play one game. Evaluators are callables returning a float in [-1, 1]
    from the side to move's perspective (None = classic hand eval)."""
    while not board.is_game_over():
        evaluator = white_eval if board.turn == chess.WHITE else black_eval
        move = nn_train.batched_minimax_move(board, depth, evaluator)
        if move is None:
            break
        board.push(move)

    if board.is_checkmate():
        return "white" if board.turn == chess.BLACK else "black"
    return "draw"


def load_model(path, device):
    import torch

    net = ValueNet().to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser(description="Neural net vs classic engine match")
    parser.add_argument("--model", required=True, help="trained .pt checkpoint")
    parser.add_argument("--model2", default=None, help="second checkpoint (net vs net)")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args.model, device)
    net2 = load_model(args.model2, device) if args.model2 else None

    stats = {"net_wins": 0, "net2_wins": 0, "draws": 0}

    for g in range(args.games):
        net_is_white = g % 2 == 0
        board = chess.Board()
        if net2 is None:
            winner = play_game(
                board,
                net.value if net_is_white else None,
                None if net_is_white else net.value,
                args.depth,
            )
            if winner == "draw":
                stats["draws"] += 1
            elif (winner == "white") == net_is_white:
                stats["net_wins"] += 1
            else:
                stats["net2_wins"] += 1
        else:
            winner = play_game(
                board,
                net.value if net_is_white else net2.value,
                net2.value if net_is_white else net.value,
                args.depth,
            )
            if winner == "draw":
                stats["draws"] += 1
            elif (winner == "white") == net_is_white:
                stats["net_wins"] += 1
            else:
                stats["net2_wins"] += 1

    total = sum(stats.values())
    print(f"Match over {total} games at depth {args.depth}")
    if net2 is None:
        print(f"  neural net      : {stats['net_wins']} wins")
        print(f"  classic engine  : {stats['net2_wins']} wins")
    else:
        print(f"  {args.model} : {stats['net_wins']} wins")
        print(f"  {args.model2}: {stats['net2_wins']} wins")
    print(f"  draws           : {stats['draws']}")
    pct = stats["net_wins"] / total * 100
    print(f"  net score       : {pct:.1f}%")


if __name__ == "__main__":
    main()
