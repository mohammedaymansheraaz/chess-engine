"""
Play-style shaping -- adds aggression / positional / tempo signals on top
of the engine's leaf evaluation, for BOTH the classic evaluator and the
neural net (since both flow through engine.evaluate()). Style knobs are
additive centipawn adjustments selected at runtime via set_style_weights().

Three named style presets:

    "balanced"    -- all zero (default = current behaviour)
    "aggressive"  -- reward king attacks, sacrifices' tempo, central knights,
                     open lines for rooks/queens; penalise blocked positions
    "positional" -- reward pawn structure, long-term piece activity, central
                    control via pawns, bishop pair, safe king; penalise
                    premature piece development/reckless pawn moves
    "defensive"  -- reward king safety, blocked centre, knight-outposts
                   defending; penalise king exposure

These are heuristic, not learned. They apply cheaply at every leaf --
compute is dominated by board.piece_map() iteration and a couple of
king-ring scans, both of which the existing evaluate() already does.

The weights can also be tuned directly (style_weights dict) for finer
control, e.g. via the API: /api/style?aggression=80&risk=-40
"""

import chess

# Centipawn scale of each style term (tune tradeoffs here)
DEFAULT_WEIGHTS = {
    "balanced": {"aggression": 0, "positional": 0, "risk_avoid": 0},
    "aggressive": {"aggression": 120, "positional": -20, "risk_avoid": -60},
    "positional": {"aggression": -20, "positional": 100, "risk_avoid": 20},
    "defensive": {"aggression": -40, "positional": 20, "risk_avoid": 120},
}

# active style weights, mutable at runtime
_style = {"aggression": 0.0, "positional": 0.0, "risk_avoid": 0.0}


def set_style_weights(
    aggression: float = 0.0, positional: float = 0.0, risk_avoid: float = 0.0
) -> dict:
    """Direct setter. Centipawn magnitudes (~0-200 each work well)."""
    global _style
    _style = {
        "aggression": float(aggression),
        "positional": float(positional),
        "risk_avoid": float(risk_avoid),
    }
    return current_style_weights()


def set_style_preset(name: str) -> dict:
    """Apply a named preset ('balanced', 'aggressive', 'positional', 'defensive')."""
    if name not in DEFAULT_WEIGHTS:
        raise ValueError(
            f"unknown style preset {name!r}; pick one of {list(DEFAULT_WEIGHTS)}"
        )
    return set_style_weights(**DEFAULT_WEIGHTS[name])


def current_style_weights() -> dict:
    return dict(_style)


def _king_ring_squares(board: chess.Board, color: bool) -> set:
    ks = board.king(color)
    if ks is None:
        return set()
    r, c = ks // 8, ks % 8
    ring = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                ring.add(nr * 8 + nc)
    return ring


def style_adjustment(board: chess.Board) -> int:
    """Additive centipawn adjustment (positive = good for white).
    Aggression signal:
      * + (enemy king attacker count - own king attacker count) * W
      * + own attack zones overlap with enemy king ring
    Positional signal:
      * + pawn-structure stability (doubled/isolated pawns -> penalty)
      * + bishop pair bonus (already classic; small augmentation)
      * + central pawn control (e4/d4/e5/d5 with own pawns)
    Risk-avoid signal:
      * - own king exposed (own king ring containing enemy pieces)
      * + own king shielded (own pawns in front of king)
    """
    a = _style["aggression"]
    p = _style["positional"]
    r = _style["risk_avoid"]
    if a == 0 and p == 0 and r == 0:
        return 0

    # Count attackers to each king's ring (attackers = enemy pieces attacking
    # a square in the king's ring).
    w_king_ring = _king_ring_squares(board, chess.WHITE)
    b_king_ring = _king_ring_squares(board, chess.BLACK)
    w_attackers = sum(1 for sq in b_king_ring if board.is_attacked_by(chess.WHITE, sq))
    b_attackers = sum(1 for sq in w_king_ring if board.is_attacked_by(chess.BLACK, sq))
    aggression_cps = w_attackers - b_attackers

    # Positional: central pawn control + pawn-structure penalties
    central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    w_central_pawns = sum(
        1
        for sq in central_squares
        if (pp := board.piece_at(sq))
        and pp.color == chess.WHITE
        and pp.piece_type == chess.PAWN
    )
    b_central_pawns = sum(
        1
        for sq in central_squares
        if (pp := board.piece_at(sq))
        and pp.color == chess.BLACK
        and pp.piece_type == chess.PAWN
    )

    # Doubled / isolated pawn penalty (very rough)
    def pawn_files(color):
        files = [0] * 8
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.PAWN and p.color == color:
                files[sq % 8] += 1
        return files

    wf = pawn_files(chess.WHITE)
    bf = pawn_files(chess.BLACK)
    w_doubled = sum(max(0, n - 1) for n in wf)
    b_doubled = sum(max(0, n - 1) for n in bf)
    positional_cps = (w_central_pawns - b_central_pawns) - (w_doubled - b_doubled) * 2

    # Risk: own king exposure. Count enemy pieces in own king's ring.
    w_exposure = sum(
        1
        for sq in w_king_ring
        if (pp := board.piece_at(sq)) and pp.color == chess.BLACK
    )
    b_exposure = sum(
        1
        for sq in b_king_ring
        if (pp := board.piece_at(sq)) and pp.color == chess.WHITE
    )

    # King pawn shield (own pawns in front of king)
    def shield_count(color):
        ks = board.king(color)
        if ks is None:
            return 0
        r, c = ks // 8, ks % 8
        dr = 1 if color == chess.WHITE else -1
        n = 0
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                pp = board.piece_at(nr * 8 + nc)
                if pp and pp.piece_type == chess.PAWN and pp.color == color:
                    n += 1
        return n

    w_shield = shield_count(chess.WHITE)
    b_shield = shield_count(chess.BLACK)
    risk_cps = (
        (b_exposure - w_exposure)  # own exposure is bad for us
        + (w_shield - b_shield) * 2
    )  # own shield is good for us

    return int(a * aggression_cps) + int(p * positional_cps) + int(r * risk_cps)
