"""
Neural-network value function for the AlphaBeta engine.

Stage-1 architecture:
    ValueNet -- 128 channels x 6 residual blocks, 30 input planes, ~400k params.
    Instantiate with channels=32,res_blocks=2 to get the Stage-0 small arch
    shape (for diagnostic comparison only; old 19-plane checkpoints will NOT
    load because conv_in's weight expects 30 input channels).

Input representation (30 planes of 8x8):
    0-5     white pawn, knight, bishop, rook, queen, king
    6-11    black pawn, knight, bishop, rook, queen, king
    12      side to move (all ones for white)
    13      halfmove clock / 100
    14-17   castling rights (WK, WQ, BK, BQ)
    18      en passant target square
    19      white has bishop pair (0/1)
    20      black has bishop pair (0/1)
    21      white advanced pawns (rank >= 6) -- passer proxy
    22      black advanced pawns (rank <= 3) -- passer proxy
    23      white knight outpost squares (own-pawn-supported squares holding a knight)
    24      black knight outpost squares
    25      white king attack zone (3x3 ring around white king)
    26      black king attack zone
    27      white piece material scalar (broadcast, normalised /39)
    28      black piece material scalar (broadcast, normalised /39)
    29      repetition clamp (0 / 1/3, single scalar broadcast)

Architecture:
    conv 3x3 (30 -> channels, ReLU)
    N residual blocks (conv 3x3, conv 3x3, skip add, ReLU)
    AdaptiveAvgPool2d(1) -> Linear(channels, 256) -> ReLU -> Linear(256, 1) -> Tanh

Trained by self-play + classic-eval imitation (see nn_train.py), then later
by net self-play + policy head (Stage 2). The net plugs into the existing
alpha-beta search through engine.set_network_eval(); the search path is
untouched so the only thing that changes strength is the evaluator.

Usage:
    net = ValueNet()                       # Stage-1 defaults
    v = net.value(board)                    # single position in [-1, 1]
    out = net(batch_of_30plane_tensors)     # batched forward (N, 30, 8, 8) -> (N, 1)
    net.save(path) / ValueNet.load(path)
"""

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
NUM_PLANES = 30


def _piece_value(pt: int) -> int:
    return {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }.get(pt, 0)


def _bishop_pair(board: chess.Board, color: bool) -> bool:
    """True if `color` has >=2 bishops on opposite square colors."""
    bishops = [
        sq
        for sq, p in board.piece_map().items()
        if p.piece_type == chess.BISHOP and p.color == color
    ]
    if len(bishops) < 2:
        return False
    cols = {(bishops[0] % 8 + bishops[0] // 8) % 2}
    return any((sq % 8 + sq // 8) % 2 not in cols for sq in bishops[1:])


def _king_attack_zone(board: chess.Board, color: bool) -> set:
    king_sq = board.king(color)
    if king_sq is None:
        return set()
    r, c = king_sq // 8, king_sq % 8
    zone = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                zone.add(nr * 8 + nc)
    return zone


def _is_outpost(board: chess.Board, square: int, color: bool) -> bool:
    """Own-pawn-supported square. Simplified outpost: a same-color pawn sits
    on a diagonal-adjacent square one rank toward our side."""
    r, c = square // 8, square % 8
    support_dr = -1 if color == chess.WHITE else 1
    for dc in (-1, 1):
        nr, nc = r + support_dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            p = board.piece_at(nr * 8 + nc)
            if p and p.piece_type == chess.PAWN and p.color == color:
                return True
    return False


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a board as a (NUM_PLANES, 8, 8) float tensor."""
    planes = torch.zeros(NUM_PLANES, 8, 8)
    for square, piece in board.piece_map().items():
        r, c = square // 8, square % 8
        channel = PIECE_TYPES.index(piece.piece_type) + (
            0 if piece.color == chess.WHITE else 6
        )
        planes[channel, r, c] = 1.0
    if board.turn == chess.WHITE:
        planes[12].fill_(1.0)
    planes[13].fill_(board.halfmove_clock / 100.0)
    castling = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for i, right in enumerate(castling):
        if right:
            planes[14 + i].fill_(1.0)
    if board.ep_square is not None:
        r, c = board.ep_square // 8, board.ep_square % 8
        planes[18, r, c] = 1.0

    # Bishop pair
    if _bishop_pair(board, chess.WHITE):
        planes[19].fill_(1.0)
    if _bishop_pair(board, chess.BLACK):
        planes[20].fill_(1.0)

    # Advanced pawns + knight outposts
    for square, p in board.piece_map().items():
        if p.piece_type == chess.PAWN:
            r, c = square // 8, square % 8
            if p.color == chess.WHITE and r >= 5:
                planes[21, r, c] = 1.0
            elif p.color == chess.BLACK and r <= 2:
                planes[22, r, c] = 1.0
        elif p.piece_type == chess.KNIGHT and _is_outpost(board, square, p.color):
            r, c = square // 8, square % 8
            chan = 23 if p.color == chess.WHITE else 24
            planes[chan, r, c] = 1.0

    # King attack zones
    for sq in _king_attack_zone(board, chess.WHITE):
        planes[25, sq // 8, sq % 8] = 1.0
    for sq in _king_attack_zone(board, chess.BLACK):
        planes[26, sq // 8, sq % 8] = 1.0

    # Material scalar planes (broadcast over 8x8)
    wmat = (
        sum(
            _piece_value(p.piece_type)
            for _, p in board.piece_map().items()
            if p.color == chess.WHITE
        )
        / 39.0
    )
    bmat = (
        sum(
            _piece_value(p.piece_type)
            for _, p in board.piece_map().items()
            if p.color == chess.BLACK
        )
        / 39.0
    )
    planes[27].fill_(wmat)
    planes[28].fill_(bmat)

    # Repetition clamp
    try:
        rep = 1 if board.is_repetition(2) else 0
    except Exception:
        rep = 0
    planes[29].fill_(rep / 3.0)

    return planes


def fens_to_tensor(fens: list) -> torch.Tensor:
    """Batch of FEN strings -> (N, NUM_PLANES, 8, 8) tensor."""
    return torch.stack([board_to_tensor(chess.Board(fen)) for fen in fens])


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(x + out)


class ValueNet(nn.Module):
    """Residual convnet value function. Stage-1 default: 128ch, 6 blocks,
    30 input planes (~400k params)."""

    def __init__(self, channels: int = 128, res_blocks: int = 6):
        super().__init__()
        self.conv_in = nn.Conv2d(NUM_PLANES, channels, 3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(res_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.res_blocks(x)
        return self.head(x)

    @torch.no_grad()
    def value(self, board: chess.Board) -> float:
        """Score one position from the side to move's perspective, [-1, 1]."""
        x = board_to_tensor(board).unsqueeze(0).to(next(self.parameters()).device)
        return float(self(x).item())

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "ValueNet":
        net = cls(**kwargs)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        net.eval()
        return net
