import chess


DEFAULT_WEIGHTS = {
    "balanced": {"aggression": 0, "positional": 0, "risk_avoid": 0},
    "aggressive": {"aggression": 120, "positional": -20, "risk_avoid": -60},
    "positional": {"aggression": -20, "positional": 100, "risk_avoid": 20},
    "defensive": {"aggression": -40, "positional": 20, "risk_avoid": 120},
}

_style = {
    "aggression": 0.0,
    "positional": 0.0,
    "risk_avoid": 0.0,
}


def set_style_weights(aggression=0.0, positional=0.0, risk_avoid=0.0):
    global _style
    _style = {
        "aggression": float(aggression),
        "positional": float(positional),
        "risk_avoid": float(risk_avoid),
    }
    return current_style_weights()


def set_style_preset(name):
    if name not in DEFAULT_WEIGHTS:
        raise ValueError(
            f"unknown style preset {name!r}; pick one of {list(DEFAULT_WEIGHTS)}"
        )
    return set_style_weights(**DEFAULT_WEIGHTS[name])


def current_style_weights():
    return dict(_style)


def _king_ring_squares(board, color):
    king_square = board.king(color)
    if king_square is None:
        return set()

    row, col = divmod(king_square, 8)
    ring = set()

    for row_change in (-1, 0, 1):
        for col_change in (-1, 0, 1):
            if row_change == 0 and col_change == 0:
                continue
            new_row = row + row_change
            new_col = col + col_change
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                ring.add(new_row * 8 + new_col)

    return ring


def style_adjustment(board):
    aggression = _style["aggression"]
    positional = _style["positional"]
    risk = _style["risk_avoid"]

    if aggression == 0 and positional == 0 and risk == 0:
        return 0

    white_king_ring = _king_ring_squares(board, chess.WHITE)
    black_king_ring = _king_ring_squares(board, chess.BLACK)

    white_attackers = sum(
        board.is_attacked_by(chess.WHITE, square) for square in black_king_ring
    )
    black_attackers = sum(
        board.is_attacked_by(chess.BLACK, square) for square in white_king_ring
    )
    aggression_score = white_attackers - black_attackers

    center = [chess.E4, chess.D4, chess.E5, chess.D5]
    white_center = sum(
        1
        for square in center
        if (piece := board.piece_at(square))
        and piece.color == chess.WHITE
        and piece.piece_type == chess.PAWN
    )
    black_center = sum(
        1
        for square in center
        if (piece := board.piece_at(square))
        and piece.color == chess.BLACK
        and piece.piece_type == chess.PAWN
    )

    def pawn_files(color):
        files = [0] * 8
        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.PAWN and piece.color == color:
                files[square % 8] += 1
        return files

    white_files = pawn_files(chess.WHITE)
    black_files = pawn_files(chess.BLACK)
    white_doubled = sum(max(0, count - 1) for count in white_files)
    black_doubled = sum(max(0, count - 1) for count in black_files)

    positional_score = (
        white_center
        - black_center
        - 2 * (white_doubled - black_doubled)
    )

    white_exposure = sum(
        1
        for square in white_king_ring
        if (piece := board.piece_at(square)) and piece.color == chess.BLACK
    )
    black_exposure = sum(
        1
        for square in black_king_ring
        if (piece := board.piece_at(square)) and piece.color == chess.WHITE
    )

    def shield_count(color):
        king_square = board.king(color)
        if king_square is None:
            return 0

        row, col = divmod(king_square, 8)
        row_change = 1 if color == chess.WHITE else -1
        count = 0

        for col_change in (-1, 0, 1):
            new_row = row + row_change
            new_col = col + col_change
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                piece = board.piece_at(new_row * 8 + new_col)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    count += 1
        return count

    white_shield = shield_count(chess.WHITE)
    black_shield = shield_count(chess.BLACK)
    risk_score = (
        black_exposure
        - white_exposure
        + 2 * (white_shield - black_shield)
    )

    return (
        int(aggression * aggression_score)
        + int(positional * positional_score)
        + int(risk * risk_score)
    )
