"""
Neural-network value function for the AlphaBeta engine.

A small residual convolutional network scores a chess position in [-1, 1]
(positive = good for the side to move), AlphaZero-style. It plugs into the
search through engine.set_network_eval(), which routes every leaf and
quiescence evaluation through net.value() -- the search itself is untouched.

Input representation: 19 planes of 8x8:
   0-5   white pawn, knight, bishop, rook, queen, king
   6-11  black pawn, knight, bishop, rook, queen, king
   12    side to move (all ones for white)
   13    halfmove clock / 100
   14-17 castling rights (WK, WQ, BK, BQ)
   18    en passant target square

Architecture (deliberately small so it trains quickly on a laptop GPU and
stays sample-efficient -- a value net for a small engine needs far less
capacity than AlphaZero's 256-filter ResNet):
   conv 3x3 (19 -> channels, ReLU)
   N residual blocks (conv 3x3, conv 3x3, skip, ReLU)
   global average pool -> FC 128 -> FC 1 -> tanh

Usage:
    net = ValueNet()
    v = net.value(board)          # single-position score, [-1, 1]
    out = net(tensors)            # batched forward: (N, 19, 8, 8) -> (N, 1)
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
NUM_PLANES = 19


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
    return planes


def fens_to_tensor(fens: list) -> torch.Tensor:
    """Batch of FEN strings -> (N, NUM_PLANES, 8, 8) tensor."""
    tensors = [board_to_tensor(chess.Board(fen)) for fen in fens]
    return torch.stack(tensors)


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
    """Residual convnet that scores a position from the side to move's
    perspective. Trained by self-play + classic-eval imitation (nn_train.py)."""

    def __init__(self, channels: int = 32, res_blocks: int = 2):
        super().__init__()
        self.conv_in = nn.Conv2d(NUM_PLANES, channels, 3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(res_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        return net
