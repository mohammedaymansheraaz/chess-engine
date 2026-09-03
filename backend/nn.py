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


def _piece_value(piece_type: int) -> int:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    return values.get(piece_type, 0)


def _bishop_pair(board: chess.Board, color: bool) -> bool:
    bishops = [
        square
        for square, piece in board.piece_map().items()
        if piece.piece_type == chess.BISHOP and piece.color == color
    ]

    if len(bishops) < 2:
        return False

    first_color = (bishops[0] % 8 + bishops[0] // 8) % 2
    return any(
        (square % 8 + square // 8) % 2 != first_color
        for square in bishops[1:]
    )


def _king_attack_zone(board: chess.Board, color: bool) -> set:
    king_square = board.king(color)
    if king_square is None:
        return set()

    row, col = divmod(king_square, 8)
    zone = set()

    for row_change in (-1, 0, 1):
        for col_change in (-1, 0, 1):
            if row_change == 0 and col_change == 0:
                continue

            new_row = row + row_change
            new_col = col + col_change
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                zone.add(new_row * 8 + new_col)

    return zone


def _is_outpost(board: chess.Board, square: int, color: bool) -> bool:
    row, col = divmod(square, 8)
    support_row = -1 if color == chess.WHITE else 1

    for col_change in (-1, 1):
        new_row = row + support_row
        new_col = col + col_change
        if 0 <= new_row < 8 and 0 <= new_col < 8:
            piece = board.piece_at(new_row * 8 + new_col)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                return True

    return False


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(NUM_PLANES, 8, 8)

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = PIECE_TYPES.index(piece.piece_type)
        if piece.color == chess.BLACK:
            channel += 6
        planes[channel, row, col] = 1.0

    if board.turn == chess.WHITE:
        planes[12].fill_(1.0)

    planes[13].fill_(board.halfmove_clock / 100.0)

    castling = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for index, has_right in enumerate(castling):
        if has_right:
            planes[14 + index].fill_(1.0)

    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        planes[18, row, col] = 1.0

    if _bishop_pair(board, chess.WHITE):
        planes[19].fill_(1.0)
    if _bishop_pair(board, chess.BLACK):
        planes[20].fill_(1.0)

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)

        if piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE and row >= 5:
                planes[21, row, col] = 1.0
            elif piece.color == chess.BLACK and row <= 2:
                planes[22, row, col] = 1.0

        elif piece.piece_type == chess.KNIGHT and _is_outpost(board, square, piece.color):
            channel = 23 if piece.color == chess.WHITE else 24
            planes[channel, row, col] = 1.0

    for square in _king_attack_zone(board, chess.WHITE):
        row, col = divmod(square, 8)
        planes[25, row, col] = 1.0

    for square in _king_attack_zone(board, chess.BLACK):
        row, col = divmod(square, 8)
        planes[26, row, col] = 1.0

    white_material = sum(
        _piece_value(piece.piece_type)
        for piece in board.piece_map().values()
        if piece.color == chess.WHITE
    ) / 39.0
    black_material = sum(
        _piece_value(piece.piece_type)
        for piece in board.piece_map().values()
        if piece.color == chess.BLACK
    ) / 39.0

    planes[27].fill_(white_material)
    planes[28].fill_(black_material)

    try:
        repeated = 1 if board.is_repetition(2) else 0
    except Exception:
        repeated = 0
    planes[29].fill_(repeated / 3.0)

    return planes


def fens_to_tensor(fens: list) -> torch.Tensor:
    boards = [chess.Board(fen) for fen in fens]
    return torch.stack([board_to_tensor(board) for board in boards])


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class ValueNet(nn.Module):
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
        device = next(self.parameters()).device
        x = board_to_tensor(board).unsqueeze(0).to(device)
        return float(self(x).item())

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "ValueNet":
        net = cls(**kwargs)
        net.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)
        )
        net.eval()
        return net
