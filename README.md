# Alphabeta — a small chess engine that learns

A from-scratch minimax chess engine with a walnut-and-ivory web UI — plus a
**neural network that learns to play by watching games** and can take over
as the engine's evaluator. Train it overnight, then play against it in the
browser or run it in a head-to-head match against the classic engine and
watch its win rate climb.

---

## Two brains, one search

Board representation, legal move generation, and check/mate detection use
the `python-chess` library (correctly implementing chess rules — castling,
en passant, threefold repetition — from scratch is its own project).
Everything that makes this an *engine* rather than a rules library is
hand-written in `backend/engine.py`:

- Minimax search with **alpha-beta pruning** (negamax formulation)
- **Iterative deepening** (search depth 1, 2, 3... within a time budget)
- Move ordering: **MVV-LVA** for captures, **killer moves** for quiet moves,
  and the transposition-table move first
- **Transposition table** (Zobrist hashing via `chess.polyglot.zobrist_hash`)
- **Quiescence search** on captures to avoid the horizon effect

The search is evaluator-agnostic. It scores a leaf position by calling
`engine.evaluate()`, which currently supports two brains:

| Evaluator | What it does |
|-----------|--------------|
| **Classic** | Hand-tuned material + piece-square tables + bishop-pair bonus. Fast, principled, static. |
| **Neural** | A small residual convnet (`backend/nn.py`) that scores any position in [-1, 1] from the side-to-move's perspective. Learned entirely from game data — no hand-coded evaluation terms. |

Switching brains is one function call: `engine.set_network_eval(net.value)`
routes every leaf and quiescence evaluation through the network; `None`
restores the classic evaluator. The search never changes.

---

## The neural network

`backend/nn.py` defines a deliberately small ResNet — a value net for a
small engine needs far less capacity than AlphaZero's 256-filter monster,
and a small net is far more sample-efficient on a single GPU:

```
input  19 planes × 8×8
  ├─ 0–5    white pawn, knight, bishop, rook, queen, king
  ├─ 6–11   black pawn, knight, bishop, rook, queen, king
  ├─ 12     side to move
  ├─ 13     halfmove clock / 100
  ├─ 14–17  castling rights (WK, WQ, BK, BQ)
  └─ 18     en passant target square
        │
conv 3×3 (19 → 32, ReLU)
        │
2 × residual block (conv 3×3, conv 3×3, skip, ReLU)
        │
global average pool → FC 128 → FC 1 → tanh      (~47k params)
```

Output is a scalar in [-1, 1], **positive = good for the side to move**
(the AlphaZero convention). At play time the engine multiplies it by 1000
to fit the centipawn scale the search already uses, so the neural brain
slots into alpha-beta, iterative deepening, quiescence, and the
transposition table with zero search changes.

---

## How it learns (`backend/nn_train.py`)

Training is self-play plus a classic-engine teacher, which the hobby-AI
literature consistently recommends over pure tabula-rasa RL on one GPU
(random self-play produces nothing but draws — no learning signal).

```
classic engine + a little randomness
              │  plays thousands of games
              ▼
   positions tagged with (outcome, classic-eval)
              │
   target = mix·outcome + (1−mix)·classic-eval
   (mix ramps 0 → 1 over the first N games)
              │
   network trained on random minibatches (Adam, MSE)
              │
   every 100 games: head-to-head vs the classic engine
              ▼
   nn_progress.csv  ← win rate per checkpoint, saved best model → nn.pt
```

Key ingredients:

- **Teacher games** (`--gen-policy classic`): the hand-written engine plays
  itself at depth 2 with 15% random moves, producing decisive, chess-like
  games fast. Positions from the last 200k sit in a replay buffer.
- **Bootstrapped targets** (`--mix-ramp`): a brand-new network can't learn
  from a sea of draws, so early targets are mostly a copy of the classic
  engine's own evaluation. The network therefore *starts* by imitating a
  decent evaluator, then the outcome signal (does this position eventually
  win?) gradually takes over — letting it improve beyond its teacher.
- **Capped-game scoring**: games that run long are scored by material, not
  forced to a draw, so even those positions teach something.
- **Online training**: two optimizer steps after every game, keeping the
  network current instead of batching at checkpoints.
- **Weight decay** for a little regularization on a small dataset.

### Train it yourself

```bash
source venv/bin/activate
cd backend

# long run: teacher games, eval match every 100 games, snapshot every 500
python nn_train.py --games 8000 --checkpoint 100 --match-games 14 \
    --depth 2 --mix-ramp 1500 --snapshot-every 500 --out nn.pt

# then let the network take over and keep improving (true self-play)
python nn_train.py --games 4000 --gen-policy net --sp-depth 2 \
    --load nn.pt --checkpoint 100 --out nn.pt
```

### Watch it learn

- **`tail -f /tmp/nn_train.log`** — one line per checkpoint:
  `[game  1200] winrate 42.9% (W6 L5 D3) loss 0.09`
- **`nn_progress.csv`** — `game, winrate, wins, losses, draws, loss, seconds`
  for plotting the improvement curve.
- **`nn.pt`** — the best model so far; `snapshots/model_<game>.pt` — a
  rotating set of checkpoints to compare.
- **`nvidia-smi`** — confirms the GPU is actually working during training.

### Measure the improvement rate

```bash
# neural net vs the classic engine, 40 games at depth 3
python match.py --model nn.pt --games 40 --depth 3

# or pit an early snapshot against a late one — the learning curve, in wins
python match.py --model snapshots/model_1000.pt --model2 snapshots/model_8000.pt --games 20
```

A single 8-hour run on an RTX 3050 6GB generates on the order of a million
positions. Expect the win rate vs the classic engine to climb from ~0%
(random weights) through ~50% (once the network has absorbed the classic
evaluation) and beyond as the outcome signal takes over.

---

## Play against it

```bash
source venv/bin/activate
cd backend

# classic evaluator (default)
uvicorn main:app --reload --port 8000

# neural evaluator — load a trained model at startup
CHESS_MODEL=/path/to/nn.pt uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000. The frontend shows which evaluator is
active ("evaluator · neural net (nn.pt, cuda)") under Engine Telemetry.
Pick a difficulty, start a game, and play. Since both brains share the same
search, they feel identical to play against — only the position-sense
differs.

---

## Run it (classic)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

For the neural brain you also need PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Difficulty levels

| Level    | Search depth |
|----------|--------------|
| Casual   | 2            |
| Club     | 3            |
| Serious  | 4            |
| Ruthless | 5            |

Higher depths take longer per move (there's a 5s time cap per move via
iterative deepening, so it plays the best move found in time rather than
hang).

## API

| Endpoint              | Method | Description                              |
|-----------------------|--------|------------------------------------------|
| `/api/new-game`       | POST   | Start a new game                         |
| `/api/state`          | GET    | Get current board state                  |
| `/api/move`           | POST   | Submit a player move (UCI format)        |
| `/api/engine-move`    | POST   | Request the engine's next move           |
| `/api/undo`           | POST   | Undo last player+engine moves            |
| `/api/eval`           | GET    | Get static evaluation score              |
| `/api/engine`         | GET    | Active evaluator + difficulty mapping    |

All endpoints accept an optional `game_id` query parameter for multi-game
support. Without it, the most recent game is used.

## Features

- **Neural evaluation** plug-in that can replace the hand-written evaluator
  at runtime (`CHESS_MODEL` env var or `engine.set_network_eval`)
- **Pawn promotion picker**: UI prompts for Q/R/B/N on the back rank
- **Draw detection**: threefold repetition, 50-move rule, stalemate, and
  insufficient material surfaced in the UI
- **Multiple concurrent games**: unique game IDs, up to 50 simultaneous
- **Concurrency safety**: per-game locks against simultaneous requests
- **Input validation** and **turn enforcement** with proper HTTP codes

## Project structure

```
backend/
  engine.py      — search (alpha-beta, iterative deepening, TT, quiescence)
  nn.py          — neural value network + board encoding
  nn_train.py    — self-play / teacher training loop
  match.py       — head-to-head: neural net vs classic engine
  train.py       — (bonus) linear TD(0) self-play trainer for material weights
  main.py        — FastAPI server, game state, REST API, model loader
frontend/
  index.html     — board UI, vanilla JS, talks to the API
```

## Design decisions

**One search, swappable evaluator.** Rather than building a separate
"neural engine", the network plugs into `engine.evaluate()` through a
single hook (`engine.set_network_eval`). The alpha-beta search, move
ordering, transposition table, and quiescence are shared by both brains —
so a match between them isolates exactly the quality of the evaluation,
which is what training improves.

**A teacher before pure self-play.** From the AlphaZero paper and every
small-engine follow-up: starting from random weights, self-play produces
only draws for a long time, so there's no learning signal. Pre-seeding the
network with the classic engine's own games (blended targets ramping from
imitation to outcome) gives it a competent starting point, then outcome
learning pushes it past the teacher. This is the difference between "trains
overnight" and "trains for months on 5,000 TPUs."

**Batched match search.** The neural evaluator is ~1000× more expensive per
position than the classic one (a GPU forward vs a table lookup). Running it
unbatched through alpha-beta quiescence made a single eval-match game take
~40 minutes once the network learned real values. Both sides of a match
therefore share a batched full-width minimax — every leaf of the depth-2
tree is scored in one GPU forward — so a game takes ~12 s and the match
still isolates evaluator quality at a fixed depth.

**Small network, big data.** ~47k params is tiny by deep-learning standards
but a good fit for the dataset a single GPU can generate overnight. A
smaller network learns material, piece placement, and win/loss prediction
from a few hundred thousand positions — capacity that would still be
under-trained on this much data.

**Persistence.** Game state is in-memory only. A server restart loses all
games — the right tradeoff for a demo.

**Single-file frontend.** One HTML file with inline CSS and JS: trivially
deployable and easy to understand.

## Benchmarks

### Search (classic evaluator, transposition table on)

Measured from the starting position (your mileage will vary):

| Depth | Nodes     | Time    | Nodes/sec  |
|-------|-----------|---------|------------|
| 2     | 144       | 0.005s  | ~29,000    |
| 3     | 1,155     | 0.025s  | ~46,000    |
| 4     | 3,370     | 0.100s  | ~34,000    |
| 5     | 33,376    | 0.823s  | ~41,000    |

The transposition table is the largest single speedup: 36% fewer nodes and
35% less time at depth 4 on a midgame position, because many positions are
reached via different move orders.

### Neural network (training)

On an RTX 3050 6GB (CUDA 12.4): a batched 19×8×8 forward is ~0.08 ms per
position; teacher games generate at ~2 s each (~150 positions). An
overnight run produces on the order of a million training positions. See
`nn_progress.csv` for the live win-rate curve vs the classic engine.

#### Sample run (this repo, ~1000 games)

The network reaches parity with the classic engine within a few hundred
games, then oscillates around 50% while the target blend is still ramping.
The trend matters more than any single checkpoint (each is only ~14 games):

| Game | Net winrate vs classic (depth 2) | Note                          |
|------|----------------------------------|-------------------------------|
| 100  | 0%   (0W 14L)                    | mix ramp just starting        |
| 300  | 50%  (7W 7L)                     | absorbed classic material     |
| 500  | 50%  (7W 7L)                     | best checkpoint saved         |
| 900  | 64%  (9W 5L)                     | outcome signal taking over    |
| 1000 | 43%  (1W 3L 10D)                 | many draws = play converged   |

Match format: the network plays a real game against the classic evaluator,
both sides using the same depth-2 full-width minimax. Once the mix ramp
completes (game ~1500) the network is trained purely on game outcomes, so
the winrate should climb and stay above the classic baseline as training
continues.

Run a fresh head-to-head any time:

```
python backend/match.py --model backend/nn.pt --games 40 --depth 3
```

### Sample browser game vs the neural engine

Played against `nn.pt` at Club level (depth 3) — the engine answered a live
opponent with a coherent modern opening (white played 1.e4, net replied
1...Nf6 2...Nc6). Telemetry showed the neural evaluator producing a real
score and the search reporting depth 3 / ~2,800 nodes in ~0.7 s.
