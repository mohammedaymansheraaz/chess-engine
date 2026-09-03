import time

import chess
import chess.polyglot

import style


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]

QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

KING_TABLE_MID = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
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
EVAL_WEIGHTS = dict(PIECE_VALUES)
BISHOP_PAIR_BONUS = 30
NETWORK_EVAL = None


def set_eval_weights(weights: dict, bishop_pair: int = BISHOP_PAIR_BONUS):
    global EVAL_WEIGHTS, BISHOP_PAIR_BONUS
    EVAL_WEIGHTS = dict(weights)
    EVAL_WEIGHTS[chess.KING] = 0
    BISHOP_PAIR_BONUS = bishop_pair


def get_eval_weights():
    return dict(EVAL_WEIGHTS)


def set_network_eval(fn):
    global NETWORK_EVAL
    NETWORK_EVAL = fn


def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    if NETWORK_EVAL is not None:
        value = NETWORK_EVAL(board)
        if board.turn == chess.BLACK:
            value = -value
        return int(round(value * 1000)) + style.style_adjustment(board)

    score = 0
    white_bishops = 0
    black_bishops = 0

    for square, piece in board.piece_map().items():
        value = EVAL_WEIGHTS[piece.piece_type]
        table = TABLES[piece.piece_type]
        table_square = square if piece.color == chess.WHITE else chess.square_mirror(square)
        score_change = value + table[table_square]

        if piece.color == chess.WHITE:
            score += score_change
            if piece.piece_type == chess.BISHOP:
                white_bishops += 1
        else:
            score -= score_change
            if piece.piece_type == chess.BISHOP:
                black_bishops += 1

    if white_bishops >= 2:
        score += BISHOP_PAIR_BONUS
    if black_bishops >= 2:
        score -= BISHOP_PAIR_BONUS

    return score + style.style_adjustment(board)


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move):
        return 0

    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)

    victim_value = PIECE_VALUES[victim.piece_type] if victim else 100
    attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
    return victim_value * 10 - attacker_value


def order_moves(board: chess.Board, moves, killers, tt_move=None):
    scored = []

    for move in moves:
        score = 0

        if tt_move is not None and move == tt_move:
            score += 2000
        if board.is_capture(move):
            score += 1000 + mvv_lva_score(board, move)
        elif move in killers:
            score += 500
        if move.promotion:
            score += 800

        scored.append((score, move))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [move for _, move in scored]


EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2
TT_SIZE = 1 << 20


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
        old = self.table.get(key)
        if old is not None and old.depth > depth:
            return

        self.table[key] = TTEntry(depth, flag, score, best_move)

        if len(self.table) > TT_SIZE:
            self._evict()

    def _evict(self):
        remove_count = len(self.table) // 4
        for key in list(self.table)[:remove_count]:
            del self.table[key]

    def clear(self):
        self.table.clear()


class SearchStats:
    def __init__(self):
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 5.0
        self.aborted = False

    def should_stop(self):
        if self.nodes & 4095 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.aborted = True
        return self.aborted


def quiescence(board: chess.Board, alpha: int, beta: int, stats: SearchStats) -> int:
    stats.nodes += 1
    if stats.should_stop():
        return 0

    stand_pat = evaluate(board)
    if board.turn == chess.BLACK:
        stand_pat = -stand_pat

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures = [move for move in board.legal_moves if board.is_capture(move)]
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


def alphabeta(board, depth, alpha, beta, stats, killers, tt, ply):
    stats.nodes += 1
    if stats.should_stop():
        return 0

    if board.is_checkmate():
        return -(MATE_SCORE + depth)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if board.halfmove_clock >= 100 or board.is_repetition(3):
        return 0

    if depth == 0:
        return quiescence(board, alpha, beta, stats)

    key = chess.polyglot.zobrist_hash(board)
    entry = tt.probe(key)
    tt_move = entry.best_move if entry else None

    if entry and entry.depth >= depth:
        if entry.flag == EXACT:
            return entry.score
        if entry.flag == LOWERBOUND and entry.score >= beta:
            return entry.score
        if entry.flag == UPPERBOUND and entry.score <= alpha:
            return entry.score

    old_alpha = alpha
    ply_killers = killers.setdefault(ply, set())
    moves = order_moves(board, list(board.legal_moves), ply_killers, tt_move)

    best_score = -1000000
    best_move = None

    for move in moves:
        board.push(move)
        score = -alphabeta(
            board,
            depth - 1,
            -beta,
            -alpha,
            stats,
            killers,
            tt,
            ply + 1,
        )
        board.pop()

        if stats.aborted:
            return 0

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

        if alpha >= beta:
            if not board.is_capture(move):
                ply_killers.add(move)
            break

    if entry is None or entry.depth <= depth:
        if best_score <= old_alpha:
            flag = UPPERBOUND
        elif best_score >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        tt.store(key, depth, flag, best_score, best_move)

    return best_score


def find_best_move(board: chess.Board, max_depth=4, time_limit=5.0):
    stats = SearchStats()
    stats.start_time = time.time()
    stats.time_limit = time_limit

    killers = {}
    tt = TranspositionTable()
    legal = list(board.legal_moves)

    if not legal:
        return None, {"depth": 0, "nodes": 0, "score": 0, "time": 0.0}

    if len(legal) == 1:
        return legal[0], {"depth": 1, "nodes": 1, "score": 0, "time": 0.0}

    best_move = None
    best_score = 0
    reached_depth = 0

    for depth in range(1, max_depth + 1):
        if time.time() - stats.start_time > time_limit:
            break

        alpha = -1000000
        beta = 1000000
        current_move = None
        current_score = -1000000

        entry = tt.probe(chess.polyglot.zobrist_hash(board))
        tt_move = entry.best_move if entry else None
        moves = order_moves(board, legal, killers.get(0, set()), tt_move)

        for move in moves:
            board.push(move)
            score = -alphabeta(
                board,
                depth - 1,
                -beta,
                -alpha,
                stats,
                killers,
                tt,
                1,
            )
            board.pop()

            if stats.aborted:
                break

            if score > current_score:
                current_score = score
                current_move = move
            if score > alpha:
                alpha = score

            if time.time() - stats.start_time > time_limit:
                break

        if stats.aborted and current_move is None:
            break

        if current_move is not None:
            best_move = current_move
            best_score = current_score
            reached_depth = depth
            legal.remove(current_move)
            legal.insert(0, current_move)

        if time.time() - stats.start_time > time_limit:
            break

    elapsed = time.time() - stats.start_time
    return best_move, {
        "depth": reached_depth,
        "nodes": stats.nodes,
        "score": best_score,
        "time": round(elapsed, 3),
    }
