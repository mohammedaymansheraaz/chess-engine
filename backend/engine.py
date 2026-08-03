"""
Alpha-beta chess engine.

Board representation and legal move generation use python-chess (rules,
legality, check/mate detection are notoriously fiddly to get right from
scratch, so we lean on a battle-tested library for that layer). The actual
"engine" -- search and evaluation -- is implemented here:

  - Minimax with alpha-beta pruning (negamax formulation)
  - Iterative deepening (search depth 1, 2, 3... using remaining time)
  - Move ordering: captures first (MVV-LVA), then killer moves
  - Transposition table (Zobrist hashing via chess.polyglot.zobrist_hash)
  - Simple material + piece-square-table evaluation, bishop-pair bonus
  - Quiescence search on captures to avoid the horizon effect
"""

import time
import chess
import chess.polyglot

# --- Evaluation ------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables (white's perspective; mirrored for black).
# Encourage central control, pawn advancement, knight/bishop development.
PAWN_TABLE = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    50,
    10,
    10,
    20,
    30,
    30,
    20,
    10,
    10,
    5,
    5,
    10,
    25,
    25,
    10,
    5,
    5,
    0,
    0,
    0,
    20,
    20,
    0,
    0,
    0,
    5,
    -5,
    -10,
    0,
    0,
    -10,
    -5,
    5,
    5,
    10,
    10,
    -20,
    -20,
    10,
    10,
    5,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

KNIGHT_TABLE = [
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
    -40,
    -20,
    0,
    0,
    0,
    0,
    -20,
    -40,
    -30,
    0,
    10,
    15,
    15,
    10,
    0,
    -30,
    -30,
    5,
    15,
    20,
    20,
    15,
    5,
    -30,
    -30,
    0,
    15,
    20,
    20,
    15,
    0,
    -30,
    -30,
    5,
    10,
    15,
    15,
    10,
    5,
    -30,
    -40,
    -20,
    0,
    5,
    5,
    0,
    -20,
    -40,
    -50,
    -40,
    -30,
    -30,
    -30,
    -30,
    -40,
    -50,
]

BISHOP_TABLE = [
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    10,
    10,
    5,
    0,
    -10,
    -10,
    5,
    5,
    10,
    10,
    5,
    5,
    -10,
    -10,
    0,
    10,
    10,
    10,
    10,
    0,
    -10,
    -10,
    10,
    10,
    10,
    10,
    10,
    10,
    -10,
    -10,
    5,
    0,
    0,
    0,
    0,
    5,
    -10,
    -20,
    -10,
    -10,
    -10,
    -10,
    -10,
    -10,
    -20,
]

ROOK_TABLE = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    5,
    10,
    10,
    10,
    10,
    10,
    10,
    5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    -5,
    0,
    0,
    0,
    0,
    0,
    0,
    -5,
    0,
    0,
    0,
    5,
    5,
    0,
    0,
    0,
]

QUEEN_TABLE = [
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
    -10,
    0,
    0,
    0,
    0,
    0,
    0,
    -10,
    -10,
    0,
    5,
    5,
    5,
    5,
    0,
    -10,
    -5,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    0,
    0,
    5,
    5,
    5,
    5,
    0,
    -5,
    -10,
    5,
    5,
    5,
    5,
    5,
    0,
    -10,
    -10,
    0,
    5,
    0,
    0,
    0,
    0,
    -10,
    -20,
    -10,
    -10,
    -5,
    -5,
    -10,
    -10,
    -20,
]

KING_TABLE_MID = [
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -30,
    -40,
    -40,
    -50,
    -50,
    -40,
    -40,
    -30,
    -20,
    -30,
    -30,
    -40,
    -40,
    -30,
    -30,
    -20,
    -10,
    -20,
    -20,
    -20,
    -20,
    -20,
    -20,
    -10,
    20,
    20,
    0,
    0,
    0,
    0,
    20,
    20,
    20,
    30,
    10,
    0,
    0,
    10,
    30,
    20,
]

TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE_MID,
}

MATE_SCORE = 99999

# Material weights used by evaluate(). These are the values the self-play
# trainer (train.py) tunes with temporal-difference learning. They default
# to the classic hand-picked values in PIECE_VALUES.
EVAL_WEIGHTS = dict(PIECE_VALUES)

# Bishop-pair bonus, also learnable.
BISHOP_PAIR_BONUS = 30


def set_eval_weights(weights: dict, bishop_pair: int = BISHOP_PAIR_BONUS) -> None:
    """Replace the material weights used by evaluate().

    `weights` maps chess piece types to centipawn values, e.g.
    {chess.PAWN: 100, chess.KNIGHT: 320, ...}. The king is always kept at
    0 (it is never captured).
    """
    global EVAL_WEIGHTS, BISHOP_PAIR_BONUS
    EVAL_WEIGHTS = dict(weights)
    EVAL_WEIGHTS[chess.KING] = 0
    BISHOP_PAIR_BONUS = bishop_pair


def get_eval_weights() -> dict:
    return dict(EVAL_WEIGHTS)


# Optional neural-network evaluator. When set, evaluate() delegates to it
# instead of the hand-written material + PST scoring. The callable must
# accept a chess.Board and return a float in [-1, 1] from the side to
# move's perspective (positive = good for the side to move). See nn.py.
NETWORK_EVAL = None


def set_network_eval(fn) -> None:
    """Enable a neural evaluator, or pass None to fall back to hand eval.

    Swaps the evaluation used everywhere in the search (leaves, quiescence)
    at the next call, so a trained network can replace the classic eval
    without touching the search code.
    """
    global NETWORK_EVAL
    NETWORK_EVAL = fn


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns, positive = good for white.

    Material + piece-square tables, plus a bishop-pair bonus. The mobility
    term is intentionally omitted: board.legal_moves.count() generates the
    full move list on every call and was a measurable bottleneck in the
    search hot loop.
    """
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    if NETWORK_EVAL is not None:
        # The network scores from the side to move's perspective; convert
        # to the white-perspective convention the rest of evaluate() uses.
        value = NETWORK_EVAL(board)
        if board.turn == chess.BLACK:
            value = -value
        return int(round(value * 1000.0))

    score = 0
    white_bishops = 0
    black_bishops = 0

    for square, piece in board.piece_map().items():
        value = EVAL_WEIGHTS[piece.piece_type]
        table = TABLES[piece.piece_type]
        idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
        pst = table[idx]
        if piece.color == chess.WHITE:
            score += value + pst
            if piece.piece_type == chess.BISHOP:
                white_bishops += 1
        else:
            score -= value + pst
            if piece.piece_type == chess.BISHOP:
                black_bishops += 1

    if white_bishops >= 2:
        score += BISHOP_PAIR_BONUS
    if black_bishops >= 2:
        score -= BISHOP_PAIR_BONUS

    return score


# --- Move ordering -----------------------------------------------------


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """Most Valuable Victim - Least Valuable Attacker heuristic."""
    if not board.is_capture(move):
        return 0
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    victim_value = PIECE_VALUES[victim.piece_type] if victim else 100  # en passant
    attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
    return victim_value * 10 - attacker_value


def order_moves(board: chess.Board, moves, killers, tt_move=None):
    def key(move):
        score = 0
        if tt_move and move == tt_move:
            score += 2000  # transposition table move first
        if board.is_capture(move):
            score += 1000 + mvv_lva_score(board, move)
        elif move in killers:
            score += 500
        if move.promotion:
            score += 800
        return -score  # sort descending

    return sorted(moves, key=key)


# --- Transposition table ------------------------------------------------

# Bound flags: how the stored score relates to the true minimax value.
EXACT = 0  # score is exact
LOWERBOUND = 1  # score >= true value (fail-high)
UPPERBOUND = 2  # score <= true value (fail-low)

TT_SIZE = 1 << 20  # ~1M entries, soft cap before eviction


class TTEntry:
    __slots__ = ("depth", "flag", "score", "best_move")

    def __init__(self, depth, flag, score, best_move):
        self.depth = depth
        self.flag = flag
        self.score = score
        self.best_move = best_move


class TranspositionTable:
    def __init__(self):
        self.table = {}

    def probe(self, key):
        return self.table.get(key)

    def store(self, key, depth, flag, score, best_move):
        # Only replace if the new entry searched at least as deep.
        existing = self.table.get(key)
        if existing and existing.depth > depth:
            return
        self.table[key] = TTEntry(depth, flag, score, best_move)
        if len(self.table) > TT_SIZE:
            self._evict()

    def _evict(self):
        keys = list(self.table.keys())
        for k in keys[: len(keys) // 4]:
            del self.table[k]

    def clear(self):
        self.table.clear()


# --- Search --------------------------------------------------------------


class SearchStats:
    def __init__(self):
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 5.0
        self.aborted = False

    def should_stop(self):
        """Time check every 4096 nodes, so time.time() calls stay cheap."""
        if self.nodes & 4095 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.aborted = True
                return True
        return self.aborted


def quiescence(board: chess.Board, alpha: int, beta: int, stats: SearchStats) -> int:
    """Extend search on captures only, to avoid the horizon effect."""
    stats.nodes += 1

    if stats.should_stop():
        return 0

    stand_pat = evaluate(board)
    if board.turn == chess.BLACK:
        stand_pat = -stand_pat

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    captures = order_moves(board, captures, set())

    for move in captures:
        board.push(move)
        score = -quiescence(board, -beta, -alpha, stats)
        board.pop()

        if stats.aborted:
            return 0
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def alphabeta(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    stats: SearchStats,
    killers: dict,
    tt: TranspositionTable,
    ply: int,
) -> int:
    stats.nodes += 1

    if stats.should_stop():
        return 0

    if board.is_checkmate():
        # Mate score adjusted by remaining depth: the engine prefers the
        # fastest mate (fewest plies) and delays being mated the longest.
        return -(MATE_SCORE + depth)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    # Cheaper draw checks than board.can_claim_draw(), which walks the
    # entire move stack on every node.
    if board.halfmove_clock >= 100:  # 50-move rule
        return 0
    if board.is_repetition(3):  # threefold repetition
        return 0

    if depth == 0:
        return quiescence(board, alpha, beta, stats)

    tt_key = chess.polyglot.zobrist_hash(board)
    tt_entry = tt.probe(tt_key)
    tt_move = None

    # Use the stored score if it proves the window is already closed.
    if tt_entry and tt_entry.depth >= depth:
        if tt_entry.flag == EXACT:
            return tt_entry.score
        if tt_entry.flag == LOWERBOUND and tt_entry.score >= beta:
            return tt_entry.score
        if tt_entry.flag == UPPERBOUND and tt_entry.score <= alpha:
            return tt_entry.score

    if tt_entry:
        tt_move = tt_entry.best_move

    original_alpha = alpha
    ply_killers = killers.setdefault(ply, set())
    moves = order_moves(board, list(board.legal_moves), ply_killers, tt_move)

    best = -1000000
    best_move = None

    for move in moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -beta, -alpha, stats, killers, tt, ply + 1)
        board.pop()

        if stats.aborted:
            return 0

        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            if not board.is_capture(move):
                ply_killers.add(move)
            break  # beta cutoff

    if tt_entry is None or tt_entry.depth <= depth:
        if best <= original_alpha:
            flag = UPPERBOUND
        elif best >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        tt.store(tt_key, depth, flag, best, best_move)

    return best


def find_best_move(board: chess.Board, max_depth: int = 4, time_limit: float = 5.0):
    """Iterative deepening driver. Returns (best_move, info dict)."""
    stats = SearchStats()
    stats.start_time = time.time()
    stats.time_limit = time_limit
    killers = {}
    tt = TranspositionTable()

    best_move = None
    best_score = 0
    reached_depth = 0

    legal = list(board.legal_moves)
    if not legal:
        return None, {"depth": 0, "nodes": 0, "score": 0, "time": 0.0}
    if len(legal) == 1:
        return legal[0], {"depth": 1, "nodes": 1, "score": 0, "time": 0.0}

    for depth in range(1, max_depth + 1):
        if time.time() - stats.start_time > time_limit:
            break

        alpha, beta = -1000000, 1000000
        current_best = None
        current_best_score = -1000000

        root_tt = tt.probe(chess.polyglot.zobrist_hash(board))
        moves = order_moves(
            board,
            legal,
            killers.get(0, set()),
            root_tt.best_move if root_tt else None,
        )
        for move in moves:
            board.push(move)
            score = -alphabeta(board, depth - 1, -beta, -alpha, stats, killers, tt, 1)
            board.pop()

            if stats.aborted:
                break

            if score > current_best_score:
                current_best_score = score
                current_best = move
            if score > alpha:
                alpha = score

            if time.time() - stats.start_time > time_limit:
                break

        if stats.aborted and current_best is None:
            break

        if current_best is not None:
            best_move = current_best
            best_score = current_best_score
            reached_depth = depth
            # Move the best move to the front for the next iteration's ordering.
            legal.remove(current_best)
            legal.insert(0, current_best)

        if time.time() - stats.start_time > time_limit:
            break

    elapsed = time.time() - stats.start_time
    return best_move, {
        "depth": reached_depth,
        "nodes": stats.nodes,
        "score": best_score,
        "time": round(elapsed, 3),
    }
