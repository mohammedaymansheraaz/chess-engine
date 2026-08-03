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
| **Neural** | A residual convnet (`backend/nn.py`) that scores any position in [-1, 1] from the side-to-move's perspective. Learned entirely from game data — no hand-coded evaluation terms. |

Switching brains is one function call: `engine.set_network_eval(net.value)`
routes every leaf and quiescence evaluation through the network; `None`
restores the classic evaluator. The search never changes.

---

## The neural network (Stage 1)

`backend/nn.py` defines a ResNet value function — deliberately sized for
overnight training on a single laptop GPU (RTX 3050 6GB). The Stage-1
architecture is larger than the original 47k-param net, giving it the
capacity to exceed the classic teacher and climb toward GM strength.

```
input  30 planes × 8×8
  ├─ 0–5      white pawn, knight, bishop, rook, queen, king
  ├─ 6–11     black pawn, knight, bishop, rook, queen, king
  ├─ 12       side to move (all 1s for white)
  ├─ 13       halfmove clock / 100
  ├─ 14–17    castling rights (WK, WQ, BK, BQ)
  ├─ 18       en passant target square
  ├─ 19–20    bishop pair (white / black)
  ├─ 21–22    advanced pawns (passer proxy)
  ├─ 23–24    knight outpost squares
  ├─ 25–26    king attack zones (ring around each king)
  ├─ 27–28    total material scalars (white / black, broadcast)
  └─ 29       repetition clamp
        │
conv 3×3 (30 → 128, ReLU)
        │
6 × residual block (conv 3×3, conv 3×3, skip, ReLU)
        │
global average pool → FC 256 → ReLU → FC 1 → tanh      (~1.8M params)
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
   (mix ramps 0 → mix-max over the first N games; never hits 1.0)
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
  games fast. Positions from the last 1M sit in a replay buffer.
- **Capped mix ramp** (`--mix-max 0.65`): early targets blend the classic
  eval, but the mix **never reaches 1.0** — the network always keeps a
  teacher anchor, preventing the catastrophic forgetting that plagued earlier
  runs. This is the single most important fix for stable long runs.
- **Larger eval matches** (`--match-games 50`): 50 games per checkpoint
  (up from 14) gives a ±7% winrate signal instead of ±20% noise.
- **Capped-game scoring**: games that run long are scored by material, not
  forced to a draw, so even those positions teach something.
- **Online training**: two optimizer steps after every game (`--train-per-game 2`),
  keeping the network current.
- **Weight decay + lower LR** (`--lr 1e-4`) for the larger network.

### Train it yourself

```bash
source venv/bin/activate
cd backend

# Stage 1: teacher games, eval match every 100 games, snapshot every 500
# Uses --mix-max 0.65 so the net never loses its teacher anchor
python nn_train.py --games 5000 --checkpoint 100 --match-games 50 \
    --depth 2 --sp-depth 1 --mix-ramp 1500 --mix-max 0.65 \
    --gen-policy classic --gen-depth 2 --explore 0.15 \
    --buffer 1000000 --train-per-game 2 --batch 256 \
    --lr 1e-4 --snapshot-every 500 --out nn_stage1.pt

# Stage 2 (planned): true self-play, net generates its own games
# python nn_train.py --games 4000 --gen-policy net --sp-depth 2 \
#     --load nn_stage1.pt --checkpoint 100 --out nn_stage1.pt
```

### Watch it learn

- **`tail -f /tmp/nn_stage1.log`** — one line per checkpoint:
  `[game  1200] winrate 42.9% (W6 L5 D3) loss 0.09`
- **`nn_progress.csv`** — `game, winrate, wins, losses, draws, loss, seconds`
  for plotting the improvement curve.
- **`nn_stage1.pt`** — the best model so far; `snapshots/model_<game>.pt` — a
  rotating set of checkpoints to compare.
- **`nvidia-smi`** — confirms the GPU is actually working during training.

### Measure the improvement rate

```bash
# neural net vs the classic engine, 50 games at depth 2
python match.py --model nn_stage1.pt --games 50 --depth 2

# or pit an early snapshot against a late one — the learning curve, in wins
python match.py --model snapshots/model_1000.pt --model2 snapshots/model_8000.pt --games 20
```

A single 6–8 hour nightly run on an RTX 3050 6GB generates on the order of
200–300k positions. Expect the win rate vs the classic engine to climb from
~0% (random weights) through ~50% (once the network has absorbed the classic
evaluation) and beyond as the outcome signal takes over — the `mix-max` cap
lets it climb **past** the teacher instead of collapsing.

---

## Play against it (with style)

```bash
source venv/bin/activate
cd backend

# classic evaluator (default)
uvicorn main:app --reload --port 8000

# neural evaluator — load a trained model at startup
CHESS_MODEL=/path/to/nn_stage1.pt uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000. The frontend shows which evaluator is
active ("evaluator · neural net (nn_stage1.pt, cuda)") under Engine Telemetry.

### Play-style presets

The engine now supports **play-style shaping** without retraining. Style
knobs are additive centipawn adjustments applied at the leaf evaluation
layer — they shift the engine's *feel* instantly for both the classic
and neural evaluators.

```bash
# Set style at runtime via API (works instantly)
curl -X POST "http://localhost:8000/api/style?preset=aggressive"
curl -X POST "http://localhost:8000/api/style?preset=positional"
curl -X POST "http://localhost:8000/api/style?preset=defensive"
curl -X POST "http://localhost:8000/api/style?preset=balanced"
```

| Preset | Behaviour |
|--------|-----------|
| **balanced** | Default (all weights zero) |
| **aggressive** | Rewards king attacks, sacrifices' tempo, central knights, open lines; penalises blocked positions |
| **positional** | Rewards pawn structure, long-term piece activity, central control, bishop pair, safe king |
| **defensive** | Rewards king safety, blocked centre, defensive outposts; penalises king exposure |

Style weights can also be tuned directly:
```bash
curl -X POST "http://localhost:8000/api/style?aggression=80&risk=-40"
```

---

## Stockfish Elo calibration

The repo includes `backend/stockfish_opponent.py` for calibrating real Elo
against Stockfish 16 (installed locally or downloaded via README). This
anchors winrate numbers to a known Elo scale instead of "vs classic at depth 2".

```bash
# Download Stockfish 16 static binary (no sudo, no apt)
mkdir -p stockfish && cd stockfish
curl -L -o sf.tar "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-ubuntu-x86-64-avx2.tar"
tar xf sf.tar && mv stockfish/stockfish-ubuntu-x86-64-avx2 stockfish16 && chmod +x stockfish16
cd ..

# Calibrate: play 20 games vs Stockfish at skill 5 (~1800 Elo anchor)
python backend/stockfish_opponent.py --model backend/nn_stage1.pt \
    --games 20 --depth 2 --sf-skill 5 --sf-elo-anchor 1800
```

Stockfish at full strength is ~3500 Elo. Handicapping via `--sf-skill` (0–20)
or `--sf-time` (seconds per move) gives a calibration ladder.

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
| `/api/style`          | POST   | Set play-style preset or custom weights  |

All endpoints accept an optional `game_id` query parameter for multi-game
support. Without it, the most recent game is used.

### Style API

```bash
# Apply a preset
POST /api/style?preset=aggressive
# or custom weights (centipawns)
POST /api/style?aggression=80&positional=-20&risk=-40
# Get current style
GET  /api/style
```

---

## Features

- **Neural evaluation** plug-in that can replace the hand-written evaluator
  at runtime (`CHESS_MODEL` env var or `engine.set_network_eval`)
- **Play-style shaping** without retraining — aggression/positional/defensive
  presets or custom centipawn weights, applied at the leaf layer
- **Stockfish Elo calibration** — anchors winrate to real Elo scale
- **Pawn promotion picker**: UI prompts for Q/R/B/N on the back rank
- **Draw detection**: threefold repetition, 50-move rule, stalemate, and
  insufficient material surfaced in the UI
- **Multiple concurrent games**: unique game IDs, up to 50 simultaneous
- **Concurrency safety**: per-game locks against simultaneous requests
- **Input validation** and **turn enforcement** with proper HTTP codes

---

## Project structure

```
backend/
  engine.py           — search (alpha-beta, iterative deepening, TT, quiescence)
  nn.py               — neural value network + 30-plane board encoding
  nn_train.py         — self-play / teacher training loop
  match.py            — head-to-head: neural net vs classic engine
  train.py            — (bonus) linear TD(0) self-play trainer for material weights
  main.py             — FastAPI server, game state, REST API, model loader
  stockfish_opponent.py — Stockfish 16 UCI wrapper for Elo calibration
  style.py            — play-style signals (aggression/positional/risk)
frontend/
  index.html          — board UI, vanilla JS, talks to the API
```

---

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

**Capped mix ramp.** The `--mix-max` parameter (default 0.65) caps the
imitation→outcome blend so the network never fully drops the teacher anchor.
This fixes the catastrophic forgetting that caused earlier runs to peak at
100% then collapse to ~5% once `mix` hit 1.0.

**Larger network, bigger buffer.** Stage-1 uses ~1.8M params (128ch × 6 blocks)
with a 1M-position replay buffer — capacity to genuinely exceed the teacher
and climb toward GM strength, not just match it.

**Persistence.** Game state is in-memory only. A server restart loses all
games — the right tradeoff for a demo.

**Single-file frontend.** One HTML file with inline CSS and JS: trivially
deployable and easy to understand.

---

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

On an RTX 3050 6GB (CUDA 12.4): a batched 30×8×8 forward is ~0.1 ms per
position; teacher games generate at ~2 s each (~150 positions). An
overnight run produces on the order of 200–300k training positions. See
`nn_progress.csv` for the live win-rate curve vs the classic engine.

---

## Roadmap

- **Stage 2:** True self-play RL (`--gen-policy net`), policy head + MCTS-lite
  generator, opponent pool of last-8 snapshots.
- **Stage 3:** Opening book (Lichess master book), Syzygy 6-piece tablebases,
  deeper eval-time search (depth 6+), Polyak weight averaging.
- Target: effective **~2800–3000 Elo** on hardware with all boosters stacked.

---

## License

MIT — do whatever you want, but if you make something cool, show me.