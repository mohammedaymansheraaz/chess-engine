"""
Self-play reinforcement learning for the neural value network.

Games are generated either by the hand-written classic engine (the default
"teacher", which plays decisive, chess-like games) or by the network itself
(true AlphaZero-style RL). Every position is labeled with the game outcome
(+1 win, -1 loss, 0 draw from the side to move's perspective) and kept in a
replay buffer; the network is trained on random minibatches to predict those
outcomes. Early in training the target is blended with the classic engine's
own evaluation (a mix that ramps to pure outcome over time), which gives the
network a strong prior on material and piece placement so it never has to
bootstrap from a sea of all-draw games.

Because the network's evaluation feeds its own moves, it genuinely improves
with each game. At checkpoints it is pitted against the classic evaluation
engine at a fixed search depth to measure progress.

Usage:
    python nn_train.py --games 5000 --checkpoint 100 --depth 2 --out nn.pt
"""

import argparse
import csv
import glob
import math
import os
import random
import time

import chess
import torch
import torch.nn.functional as F

import engine
from nn import ValueNet, board_to_tensor, fens_to_tensor


# --- Self-play ------------------------------------------------------------


def pick_move_1ply(net: ValueNet, board: chess.Board, temp: float, explore: float):
    """Fast one-ply move selection: score every child position with one
    batched network forward, then softmax with temperature."""
    moves = list(board.legal_moves)
    if random.random() < explore:
        return random.choice(moves)

    tensors = []
    for move in moves:
        board.push(move)
        tensors.append(board_to_tensor(board))
        board.pop()
    x = torch.stack(tensors).to(next(net.parameters()).device)
    with torch.no_grad():
        opp_value = net(x).squeeze(-1)  # opponent's value after our move
    our_value = -opp_value  # our value of making this move

    scores = our_value - our_value.max()
    probs = F.softmax(scores / temp, dim=0)
    pick = int(torch.multinomial(probs, 1).item())
    return moves[pick]


def _minimax_tree(board: chess.Board, depth: int, leaves: list, collect):
    """Grow the minimax tree. Terminal positions return a fixed value
    (-1 mated, 0 drawn) from the side to move's perspective; at depth 0 the
    live board is handed to `collect(board)` (which appends a tensor or a
    score to `leaves`) and the node returns None. Internal nodes return a
    list of child nodes."""
    if board.is_checkmate():
        return -1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.halfmove_clock >= 100
        or board.is_repetition(3)
    ):
        return 0.0
    if depth == 0:
        collect(board)
        return None
    children = []
    for move in board.legal_moves:
        board.push(move)
        children.append(_minimax_tree(board, depth - 1, leaves, collect))
        board.pop()
    return children


def batched_minimax_move(
    board: chess.Board,
    depth: int,
    evaluator,
    temp: float = 0.0,
    explore: float = 0.0,
):
    """Full-width minimax to `depth` plies. `evaluator` scores a board from
    the side to move's perspective in [-1, 1]; it is either a ValueNet
    (leaves evaluated in ONE batched GPU forward) or a plain callable
    (evaluated per board). This avoids running the expensive neural
    evaluator unbatched through alpha-beta quiescence, which made a single
    match game take ~40 minutes. temp=0 picks greedily, temp>0
    softmax-samples; explore>0 occasionally returns a random move."""
    moves = list(board.legal_moves)
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]
    if explore > 0 and random.random() < explore:
        return random.choice(moves)

    leaves = []
    is_net = isinstance(evaluator, ValueNet)
    if is_net:
        collect = lambda b: leaves.append(board_to_tensor(b))
    else:
        collect = lambda b: leaves.append(classic_evaluator(b))
    root = _minimax_tree(board, depth, leaves, collect)
    if isinstance(root, float):
        return random.choice(moves)

    if is_net and leaves:
        x = torch.stack(leaves).to(next(evaluator.parameters()).device)
        was_training = evaluator.training
        evaluator.eval()
        with torch.no_grad():
            values = evaluator(x).squeeze(-1).tolist()
        if was_training:
            evaluator.train()
    else:
        values = leaves

    it = iter(values)

    def value_of(node):
        if isinstance(node, float):
            return node
        if node is None:
            return next(it)
        return -max(value_of(child) for child in node)

    scores = [-value_of(child) for child in root]
    scores_t = torch.tensor(scores, dtype=torch.float32)
    if temp > 0:
        scores_t = scores_t - scores_t.max()
        probs = F.softmax(scores_t / temp, dim=0)
        pick = int(torch.multinomial(probs, 1).item())
    else:
        pick = int(scores_t.argmax().item())
    return moves[pick]


def classic_evaluator(board: chess.Board) -> float:
    """The classic engine's evaluation as a [-1, 1] score from the side to
    move's perspective, so it can be swapped into the same batched minimax
    as the network (fair comparison: same search, different evaluator)."""
    value = engine.evaluate(board)
    if board.turn == chess.BLACK:
        value = -value
    return squash(value)


def pick_move_deep(
    net: ValueNet, board: chess.Board, depth: int, temp: float, explore: float
):
    """Deep move selection via batched full-width minimax, then softmax over
    root scores."""
    return batched_minimax_move(board, depth, net, temp=temp, explore=explore)


def squash(value: float) -> float:
    """Map a centipawn evaluation into [-1, 1] using tanh."""
    return math.tanh(value / 1000.0)


def static_target(board: chess.Board) -> float:
    """The classic engine's static evaluation, from the side to move's
    perspective. Used to bootstrap the network: early in training the value
    target blends this with the raw game outcome, so the net never has to
    learn chess from a sea of all-draw games."""
    value = engine.evaluate(board)
    if board.turn == chess.BLACK:
        value = -value
    return squash(value)


def material_advantage(board: chess.Board) -> float:
    """White-positive material balance in centipawns (king ignored)."""
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        score += sign * values.get(piece.piece_type, 0)
    return score * 100


def self_play(
    net: ValueNet,
    depth: int,
    temp: float,
    explore: float,
    max_plies: int = 300,
    policy: str = "classic",
) -> list:
    """One self-play game. Returns a list of (fen, outcome, static) where:
    outcome is the game result (+1 win, -1 loss, 0 draw, or a material-based
    score when a game hits the ply cap) and static is the classic engine's
    evaluation, both from the side to move's perspective.

    policy="classic" generates games with the hand-written engine (a
    reliable teacher that produces decisive, chess-like games); policy="net"
    generates games with the network itself (true AlphaZero-style RL)."""
    board = chess.Board()
    positions = []  # (fen, side to move)
    ply = 0

    while not board.is_game_over() and ply < max_plies:
        positions.append((board.fen(), board.turn))
        if policy == "classic":
            if random.random() < explore:
                move = random.choice(list(board.legal_moves))
            else:
                move, _ = engine.find_best_move(board, max_depth=depth, time_limit=30)
                if move is None:
                    break
        elif depth <= 1:
            move = pick_move_1ply(net, board, temp, explore)
        else:
            move = pick_move_deep(net, board, depth, temp, explore)
        board.push(move)
        ply += 1

    if board.is_checkmate():
        outcome = 1.0 if board.turn == chess.BLACK else -1.0  # white's result
    elif board.is_game_over():
        outcome = 0.0
    else:
        # Hit the ply cap without a decision: score by material so the game
        # still teaches the network something useful instead of a forced draw.
        outcome = squash(material_advantage(board))

    data = []
    for fen, stm in positions:
        sign = 1.0 if stm == chess.WHITE else -1.0
        data.append((fen, outcome * sign, static_target(chess.Board(fen)) * sign))
    return data


# --- Training -------------------------------------------------------------


def train_step(
    net: ValueNet,
    opt,
    batch_fens: list,
    batch_outcomes: list,
    batch_static: list,
    mix: float,
) -> float:
    """One optimizer step. Each position's target blends the game outcome
    with the classic static evaluation: mix=0 copies the classic eval,
    mix=1 learns purely from game results, and values in between transfer
    the classic eval's prior smoothly into the RL signal."""
    x = fens_to_tensor(batch_fens).to(next(net.parameters()).device)
    outcomes = torch.tensor(batch_outcomes, dtype=torch.float32, device=x.device)
    static = torch.tensor(batch_static, dtype=torch.float32, device=x.device)
    y = mix * outcomes + (1.0 - mix) * static
    net.train()
    pred = net(x).squeeze(-1)
    loss = F.mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())


# --- Evaluation vs the classic engine -------------------------------------


def play_match(net: ValueNet, games: int, depth: int) -> dict:
    """Network vs the hand-written evaluation engine, alternating colors.

    Both sides use the SAME batched full-width minimax at the same depth;
    only the evaluator differs (the network vs classic_evaluator). Running
    the network through alpha-beta quiescence one position at a time is
    ~1000x slower than the classic evaluator, so both sides share this
    batched search to keep the match fast."""
    stats = {"net_wins": 0, "classic_wins": 0, "draws": 0}
    net.eval()

    for g in range(games):
        net_is_white = g % 2 == 0
        board = chess.Board()
        while not board.is_game_over():
            white_turn = board.turn == chess.WHITE
            use_net = net_is_white == white_turn
            move = batched_minimax_move(
                board, depth, net if use_net else classic_evaluator
            )
            if move is None:
                break
            board.push(move)

        if board.is_checkmate():
            white_mated = board.turn == chess.WHITE
            net_won = (net_is_white and not white_mated) or (
                not net_is_white and white_mated
            )
            if net_won:
                stats["net_wins"] += 1
            else:
                stats["classic_wins"] += 1
        else:
            stats["draws"] += 1
    return stats


# --- Main -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Neural self-play training")
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument(
        "--sp-depth", type=int, default=1, help="self-play search depth"
    )
    parser.add_argument("--depth", type=int, default=2, help="eval-match search depth")
    parser.add_argument("--temp", type=float, default=0.5, help="self-play temperature")
    parser.add_argument(
        "--explore", type=float, default=0.15, help="random-move probability"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--train-per-game",
        type=int,
        default=2,
        help="optimizer steps after each self-play game",
    )
    parser.add_argument(
        "--buffer", type=int, default=80000, help="replay buffer size (positions)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=512,
        help="train only after this many positions",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=100, help="games between evals"
    )
    parser.add_argument("--match-games", type=int, default=50, help="games per eval")
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="save a model snapshot every N games (0 = off)",
    )
    parser.add_argument(
        "--mix-ramp",
        type=int,
        default=1000,
        help="games over which the target ramps from classic-eval imitation toward game outcomes",
    )
    parser.add_argument(
        "--mix-max",
        type=float,
        default=0.65,
        help="ceiling on the imitation->outcome mix (never fully drops the teacher anchor; prevents catastrophic forgetting after ramp completes)",
    )
    parser.add_argument(
        "--gen-policy",
        choices=["classic", "net"],
        default="classic",
        help="self-play generator: the classic engine (teacher) or the network itself",
    )
    parser.add_argument(
        "--gen-depth", type=int, default=2, help="classic-teacher search depth"
    )
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--res-blocks", type=int, default=6)
    parser.add_argument("--out", default="nn.pt")
    parser.add_argument("--load", default=None, help="resume from a checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ValueNet(channels=args.channels, res_blocks=args.res_blocks).to(device)
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buffer = []  # rolling list of (fen, outcome, static); capped
    best_winrate = -1.0
    t0 = time.time()

    log_path = os.path.splitext(args.out)[0] + "_progress.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        ["game", "winrate", "wins", "losses", "draws", "loss", "seconds"]
    )

    print(f"device: {device}  params: {sum(p.numel() for p in net.parameters()):,}")
    print(
        f"{args.games} self-play games, generator={args.gen_policy}, "
        f"self-play depth {args.sp_depth}, eval-match depth {args.depth}, "
        f"buffer {args.buffer} positions, mix ramp {args.mix_ramp}\n"
    )

    for game_idx in range(1, args.games + 1):
        data = self_play(
            net,
            args.gen_depth if args.gen_policy == "classic" else args.sp_depth,
            args.temp,
            args.explore,
            policy=args.gen_policy,
        )
        buffer.extend(data)
        if len(buffer) > args.buffer:
            del buffer[: len(buffer) - args.buffer]

        mix = min(args.mix_max, game_idx / max(1, args.mix_ramp))
        if len(buffer) >= args.min_samples:
            for _ in range(args.train_per_game):
                batch = random.sample(buffer, min(args.batch, len(buffer)))
                fens = [p[0] for p in batch]
                outcomes = [p[1] for p in batch]
                statics = [p[2] for p in batch]
                loss = train_step(net, opt, fens, outcomes, statics, mix)

        if game_idx % args.checkpoint == 0 and len(buffer) >= args.min_samples:
            net.eval()
            stats = play_match(net, args.match_games, args.depth)
            total = sum(stats.values())
            winrate = (stats["net_wins"] + 0.5 * stats["draws"]) / total * 100.0
            elapsed = time.time() - t0
            print(
                f"[game {game_idx:>5}] winrate {winrate:5.1f}% "
                f"(W{stats['net_wins']} L{stats['classic_wins']} D{stats['draws']}) "
                f"loss {loss:.4f}  {elapsed:.0f}s"
            )
            log_writer.writerow(
                [
                    game_idx,
                    round(winrate, 2),
                    stats["net_wins"],
                    stats["classic_wins"],
                    stats["draws"],
                    round(loss, 4),
                    round(elapsed),
                ]
            )
            log_file.flush()
            if winrate > best_winrate:
                best_winrate = winrate
                net.save(args.out)
            if args.snapshot_every and game_idx % args.snapshot_every == 0:
                snap_dir = os.path.join(os.path.dirname(args.out) or ".", "snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                net.save(os.path.join(snap_dir, f"model_{game_idx}.pt"))
                for old in sorted(glob.glob(os.path.join(snap_dir, "model_*.pt")))[
                    :-10
                ]:
                    os.remove(old)
            net.train()

    # Final evaluation and save.
    net.eval()
    stats = play_match(net, args.match_games * 2, args.depth)
    total = sum(stats.values())
    final_winrate = (stats["net_wins"] + 0.5 * stats["draws"]) / total * 100.0
    net.save(args.out)
    log_writer.writerow(
        [
            args.games,
            round(final_winrate, 2),
            stats["net_wins"],
            stats["classic_wins"],
            stats["draws"],
            "",
            round(time.time() - t0),
        ]
    )
    log_file.close()
    print(
        f"\nDone. Final winrate vs classic eval: {final_winrate:.1f}% "
        f"(W{stats['net_wins']} L{stats['classic_wins']} D{stats['draws']}) "
        f"over {total} games"
    )
    print(f"Best checkpoint saved to {args.out} ({best_winrate:.1f}%)")
    print(f"Progress log: {log_path}")


if __name__ == "__main__":
    main()
