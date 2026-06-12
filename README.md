# Player-Centric Ball-Action Spotting (`pcspot`)

A self-contained PyTorch implementation of a player-centric ball-action
spotting model for the SoccerNet 2026 **PCBAS / FOOTPASS** task: who did
what action and when, frame-level, from broadcast soccer video.

The model combines:

- a **Heterogeneous Graph Transformer (HGT)** over per-frame multi-agent
  player graphs. Player-player edges are `self / same_team / opponent /
  near` (kNN) and a variable-degree `radius` edge type that gives the
  block a real cardinality signal alongside the kNN-censored `near`.
- a second **`zone` node type** that bins the pitch into a `6x4`
  absolute grid; players are soft-splatted into zones, occupants are
  SUM-aggregated per team, and zones diffuse to their grid neighbors
  before reporting back into each player's update. Sum aggregation is
  what lets the block perceive physical congestion and local numerical
  superiority, which softmax / mean aggregators structurally cannot.
- an **MS-TCN++** temporal stack run independently per player,
- a player-aware adaptation of the **SoccerNet CALF** loss with an
  objectness side-supervision,
- optional frozen **DINOv2 ViT-S/14** visual features extracted from
  padded player crops, and
- per-player **jersey embedding** + a small bundle of derived scalars
  (distance to own / opp goal, nearest sideline, radius-edge degree
  counts), oriented to attacking direction via the FOOTPASS
  `left_to_right` column.

Optional integrations bundled with `scripts/graph/train.py`:

- per-epoch validation that decodes predictions, runs player-centric
  NMS, and reports Average-mAP / joint Average-mAP / player identity
  accuracy via [`pcspot/eval/metrics.py`](pcspot/eval/metrics.py),
- experiment tracking with **[Weights & Biases](https://wandb.ai/)**
  for step losses, epoch averages, validation metrics, and optional
  checkpoint artifacts. Credentials stay in the environment
  (`WANDB_API_KEY` / `wandb login`); nothing is read from
  `config.toml`. See [§5 “Experiment tracking with Weights &
  Biases”](#experiment-tracking-with-weights--biases) for the full
  setup.

---

## Project layout

```
pcspot/        core library (data, models, losses, features, train, inference, eval)
scripts/       CLIs (mirror, precompute visual features, train, infer, render)
tests/         unittest suite
data/          mirrored PCBAS data (NOT committed)
third_party/   pnlcalib (camera calibration / visible pitch region helpers)
```

---

## 1. Setup

The project targets Python 3.12 and ships with a pre-built CPU-only
`.venv/` for Windows. To rebuild it from scratch on any platform:

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.12.0+cpu` and `torchvision==0.27.0+cpu`.
**For GPU training/inference**, install the CUDA build of torch first
from the appropriate PyTorch index (e.g. `--index-url
https://download.pytorch.org/whl/cu121`), then `pip install -r
requirements.txt` will skip the already-satisfied torch / torchvision.

### Credentials and `config.toml`

The data mirror script reads gated credentials from a local
`config.toml`:

```powershell
copy config.example.toml config.toml
```

Fill in:

- **`huggingface_token`** — a read-scoped token from
  <https://huggingface.co/settings/tokens>. You must also be logged in
  and have accepted the dataset terms at
  <https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026>.
- **`soccernet_password`** — emailed to you after signing the SoccerNet
  NDA (see the FOOTPASS README).

`config.toml` is listed in `.gitignore`. **Never commit it.** If a token
has been committed historically, rotate it on Hugging Face and rewrite
the git history.

### Sanity-check the library

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

All tests are `unittest`-based and run on CPU only — no GPU needed.

---

## 2. Get the data

The PCBAS / FOOTPASS dataset is gated at two independent levels (HF
terms + SoccerNet NDA). Once both are cleared and `config.toml` is
filled in, mirror the smallest split (VAL, 3 matches):

```powershell
.\.venv\Scripts\python.exe scripts\mirror_pcbas_one_match.py --config config.toml --split VAL
```

This downloads HF archives to `data/pcbas_one_match/raw/` and extracts
the password-protected ones into `data/pcbas_one_match/extracted/`. See
[`NOTES_PCBAS_MIRROR.md`](NOTES_PCBAS_MIRROR.md) for the archive layout,
resolution options (`352x640` vs `fullHD`), and troubleshooting.

Expected layout after a successful mirror:

```
data/pcbas_one_match/
  raw/           # archives downloaded verbatim from the HF repo
  extracted/     # contents of any password-protected archives
  manifest.json  # what was downloaded, extracted, or skipped
```

---

## 3. Precompute visual features (DINOv2 ViT-S/14)

The model fuses per-player visual embeddings into the HGT node
embedder. They must be precomputed once per match-half and cached on
disk before training (training reads from the cache rather than the
video):

```powershell
.\.venv\Scripts\python.exe scripts\precompute_visual_features.py `
    --config config.toml --match-id game_18 `
    --backbone dinov2_vits14 --crop-size 224 --pad-factor 1.6 `
    --batch-size 32 --device cuda:0
```

This writes one `.npz` shard per match-half under
`<cache-root>/dinov2_vits14/`. A GPU is strongly recommended; on CPU a
single half can take hours.

For a **smoke test only** (no GPU, no real weights), pass `--use-stub`
to swap DINOv2 for a deterministic statistics-based projection. The
resulting cache is **not** suitable for evaluation.

---

## 4. Define a train/val/test split

The model enforces split discipline at the `(match_id, half_id)` level
via [`pcspot.data.splits.SplitManifest`](pcspot/data/splits.py): every
half belongs to exactly one split. The manifest is a small JSON file,
for example `data/splits.json`:

```json
{
  "train": [
    {"match_id": "game_18", "half_id": "H1"}
  ],
  "val": [
    {"match_id": "game_18", "half_id": "H2"}
  ],
  "test": []
}
```

Each entry can also be a two-element list `["game_18", "H1"]`. The keys
`"train"`, `"val"`, `"test"` are conventional but not enforced — use
whatever split names you want and pass them to the trainer.

---

## 5. Train

The model ships in **three variants** for ablation studies — see
[§7 “Model variants and ablations”](#model-variants-and-ablations) for
what each one isolates. Each lives in its own thin CLI package under
`scripts/<variant>/{train,eval,sweep}.py`; they share the same flags,
data loading, validation, W&B integration, and checkpoint format via
`pcspot.train.{cli_common,runner,trainer}`. The full model
(`graph`) is the default entry point for new experiments:

```powershell
.\.venv\Scripts\python.exe scripts\graph\train.py `
    --config config.toml `
    --splits data/splits.json `
    --output-dir checkpoints/graph/run1 `
    --epochs 5 --batch-size 4 `
    --window-size 128 --stride 96 `
    --hidden-dim 64 `
    --learning-rate 1e-3 --warmup-steps 50 --grad-clip 1.0 `
    --device cuda:0
```

Swap `graph` for `no_zones` or `no_graph` to train the corresponding
ablation — the CLI, flags, and output format are identical (minus the
architecture flags each variant doesn't have, noted in the table
below); only the model that gets constructed changes. A deprecated
`scripts/train.py` shim still exists and always trains the `graph`
variant, kept so existing commands that predate the variant split keep
working.

Useful flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--visual-cache <root>` | none | Enable visual features. Must point at the dir produced in §3. |
| `--visual-backbone` | `dinov2_vits14` | Subdirectory name inside the cache root. |
| `--visual-dim` | `0` | Set to `384` for DINOv2 ViT-S/14 (must match the cache). |
| `--target-cache <dir>` | none | Disk-backed CALF / objectness target cache. Big speedup on multi-epoch runs. |
| `--no-objectness` | off | Disable objectness supervision (sets `objectness_lambda=0`). |
| `--keep-best-metric <name>` | none | Tracked metric name from `validation_fn`; if set, `best.pt` is maintained in `--output-dir`. Use a key produced by `--validation-split`, e.g. `val/map_joint`. |
| `--seed` | `42` | Seeds both numpy and torch. |
| `--validation-split <name>` | none | Split name from the manifest to score each epoch. Enables per-epoch Average-mAP / joint Average-mAP / player identity accuracy. |
| `--validation-stride <int>` | `--window-size` | Non-overlapping windows by default; lower for denser coverage. |
| `--decode-threshold <float>` | `0.5` | Score threshold for `decode_predictions` during validation. |
| `--nms-radius <int>` | `12` | `player_centric_nms` window radius in frames. |
| `--nms-mode <name>` | `per_player_class` | One of `per_player_class`, `per_player`, `per_class`. |
| `--metric-tolerances <csv>` | `3,12,25` | Frame tolerances for `average_map_at_tolerances` (~120ms, ~480ms, 1s at 25fps). |
| `--use-zone-nodes / --no-use-zone-nodes` <sup>`graph` only</sup> | on | Enable the heterogeneous `zone` node type in the HGT (count-aware SUM aggregation; see §7). |
| `--zone-grid <GxxGy>` <sup>`graph` only</sup> | `6x4` | Pitch grid for zone nodes (also accepts `Gx,Gy`). Only used when `--use-zone-nodes` is on. |
| `--use-jersey / --no-use-jersey` | on | Enable the jersey-number embedding branch; more identity-stable than `player_id` across tracklet switches. |
| `--use-goal-distances / --no-use-goal-distances` | on | Add `[dist_own_goal, dist_opp_goal, dist_nearest_sideline]` to the embedder's extra-scalars branch. |
| `--use-radius-edges / --no-use-radius-edges` <sup>`graph` / `no_zones`</sup> | on | Add a variable-degree `radius` edge type + per-player `[n_same_within_r, n_opp_within_r]` degree counts. Not present on `no_graph` (no graph to add edges to). |
| `--radius <float>` <sup>`graph` / `no_zones`</sup> | `0.15` | Pitch distance (normalized) used by `--use-radius-edges`. `0.15` ≈ 9 m on a 60 m-wide pitch. |

The flags marked "`graph` only" / "`graph` / `no_zones`" above are
absent from the other variants' argparsers — they construct a model
with no matching architecture surface, so passing them to
`no_zones`/`no_graph` fails at argument-parsing time rather than
silently no-op-ing.

Per-step and per-epoch logs are printed to stdout. The training run
metadata (CLI args + CALF config) is dumped to `<output-dir>/run.json`;
`scripts/infer.py` reads this file to recover the model construction
kwargs.

Checkpoints land at `<output-dir>/epoch_NNNN.pt` after every epoch. A
`best.pt` is written when `--keep-best-metric` is set and a matching
validation key exists. The bundled CLI provides validation natively via
`--validation-split`; the keys it returns include `val/map`,
`val/map_joint`, `val/player_identity_accuracy`, and per-tolerance
variants such as `val/map_at_t12` and `val/map_joint_at_t25`. You can
also supply your own `validation_fn` from Python by importing
`pcspot.train.trainer.Trainer.fit` directly.

### Reusable training configs (`--train-config`)

Most of the flags above describe **what experiment is being run** and
rarely change between commands. They can be moved to a check-in-able
TOML file and reused via `--train-config`:

```powershell
.\.venv\Scripts\python.exe scripts\graph\train.py `
    --config config.toml `
    --train-config configs\train\graph\baseline.toml `
    --splits data\splits.json `
    --output-dir checkpoints\graph\run1 `
    --wandb-run-name run1
```

Each variant has its own config directory —
[`configs/train/graph/`](configs/train/graph/),
[`configs/train/no_zones/`](configs/train/no_zones/),
[`configs/train/no_graph/`](configs/train/no_graph/) — each containing
a `baseline.toml` starter file plus a `vast_fullhd_dinov2.toml` for
full-resolution DINOv2 runs on rented GPU boxes. Copy one, tweak, and
commit per-experiment variants alongside the code that produced them.

**Precedence is CLI > `--train-config` TOML > built-in defaults.** Any
flag you still pass on the command line wins, so the TOML is a layer of
defaults rather than a hard configuration.

**Where each kind of setting lives:**

| Setting | Where |
| --- | --- |
| Hugging Face token, SoccerNet password, mirror `output_dir` | `config.toml` (gitignored — secrets only) |
| Hyperparams, model dims, validation defaults, W&B project/tags/mode | `--train-config` TOML (check-in-able) |
| Paths and per-run identity (`--splits`, `--output-dir`, `--visual-cache`, `--target-cache`, `--wandb-run-name`, `--wandb-notes`) | CLI only |
| `WANDB_API_KEY` | environment variable / `wandb login` (never in any file) |

Recognised sections in the train-config TOML are `[train]`,
`[validation]`, and `[wandb]`. Unknown sections or keys abort the run
with a clear error so typos never silently change a sweep. See the
header of [`configs/train/graph/baseline.toml`](configs/train/graph/baseline.toml)
(or the `no_zones` / `no_graph` equivalents — each lists only the
`[train]` keys its own variant recognises) for the full key list; it
mirrors the argparse flags 1:1, plus `[train].no_objectness =
true|false` (the negated form of `--objectness/--no-objectness`).

The resolved `train_config` path is echoed into `<output-dir>/run.json`
so a run can always be traced back to the file that defined it.

### Experiment tracking with Weights & Biases

The trainer can stream losses and validation metrics to a [Weights &
Biases](https://wandb.ai/) project. The dependency is already pinned in
[`requirements.txt`](requirements.txt); no `config.toml` changes are
required — credentials are read from the environment so secrets never
land in the repo.

1. **Authenticate once per machine** (any one of these):

   ```powershell
   .\.venv\Scripts\wandb.exe login
   ```

   ```powershell
   $env:WANDB_API_KEY = "your-wandb-api-key"
   ```

   In CI, set `WANDB_API_KEY` as a masked secret. Do **not** add it to
   `config.toml` or commit it anywhere.

2. **Add the W&B flags to the training command.** A typical run with
   validation, best-checkpoint tracking, and artifact upload:

   ```powershell
   .\.venv\Scripts\python.exe scripts\graph\train.py `
       --config config.toml `
       --splits data\splits.json `
       --output-dir checkpoints\graph\run1 `
       --epochs 5 --batch-size 4 `
       --device cuda:0 `
       --validation-split val `
       --keep-best-metric val/map_joint `
       --wandb `
       --wandb-project pcspot `
       --wandb-run-name graph-run1 `
       --wandb-tags pcbas,dinov2,graph `
       --wandb-notes "baseline + DINOv2 ViT-S/14" `
       --wandb-log-artifacts
   ```

   The W&B project, entity, tags, mode, and artifact-upload toggle can
   also be set in a `[wandb]` table inside `--train-config`, leaving
   only run-identity flags (`--wandb-run-name`, `--wandb-notes`) on the
   command line. Use `--no-wandb` to disable tracking for a one-off run
   when the TOML has `enabled = true`.

3. **Run offline or sync later.** On machines without network access:

   ```powershell
   $env:WANDB_MODE = "offline"
   ```

   or pass `--wandb-mode offline`. The run directory under `./wandb` can
   be uploaded later with `wandb sync <run-dir>`. Override the location
   with `--wandb-run-dir`.

What gets logged:

| Phase | Key prefix | Examples |
| --- | --- | --- |
| Every training step | `train/` | `train/loss_total`, `train/loss_bce`, `train/loss_tmse`, `train/loss_objectness`, `train/learning_rate`, `train/step` |
| End of every epoch | `train/avg_*`, `epoch`, `val/*` | `train/avg_loss_total`, `train/last_learning_rate`, plus every key produced by `validation_fn` |
| After training (optional) | W&B artifacts | `run.json` and `*.pt` checkpoints when `--wandb-log-artifacts` is set |

The W&B run config is initialised from the same payload that is written
to `<output-dir>/run.json`, so the W&B UI and the local checkpoint
folder stay in sync. `wandb.init` failures (missing API key, network
error, etc.) are converted to warnings: training continues with stdout
logging only.

Operational notes:

- Never store `WANDB_API_KEY` in `config.toml`, the README, or any
  committed file.
- Use stable, descriptive `--wandb-run-name` plus `--wandb-tags`
  values (e.g. visual backbone, model size, seed) so runs are easy to
  compare in the UI.
- `--output-dir` is still the local source of truth for checkpoints;
  W&B artifacts are an optional tracked copy for sharing.
- `--keep-best-metric val/map_joint` only makes sense when
  `--validation-split` is enabled and the metric tolerance set is
  non-empty.

### Training from Python

`scripts/<variant>/train.py` is a thin argparse wrapper: it defines the
variant's `architecture` flag group and a `build_model(args)` factory,
then delegates everything else — data loading, sampler dispatch,
validation, W&B, checkpointing — to
[`pcspot.train.runner.run`](pcspot/train/runner.py), which in turn
drives [`pcspot.train.trainer.Trainer.fit`](pcspot/train/trainer.py).
To train from a notebook or another script, build a model directly
(e.g. `PlayerCentricSpottingModel(...)`, `NoZonesSpottingModel(...)`,
or `NoGraphSpottingModel(...)` from `pcspot.models`) and call
`Trainer(model, ...).fit(...)`, or call `runner.run(args, build_model)`
with a hand-built `argparse.Namespace` to reuse the full orchestration.

### Standalone evaluation

`scripts/<variant>/eval.py` reconstructs the matching model class from
a checkpoint's `run.json` (written by `scripts/<variant>/train.py`)
and reproduces the validation metrics for a chosen split through the
exact same decode → NMS → `average_map_at_tolerances` pipeline used
during training, so standalone numbers match the training-time
validation log:

```powershell
.\.venv\Scripts\python.exe scripts\graph\eval.py `
    --checkpoint checkpoints\graph\run1\best.pt `
    --config config.toml --splits data\splits.json --split val `
    --visual-cache .cache\visual --visual-dim 384 --device cuda:0
```

### Hyperparameter sweeps with Weights & Biases

`scripts/<variant>/sweep.py` is the W&B sweep-agent entry point for
each variant, with matching sweep definitions in
[`configs/sweeps/`](configs/sweeps/) (`graph.yaml`, `no_zones.yaml`,
`no_graph.yaml`). It supports two modes — create-and-run, or
agent-only against an already-created sweep:

```powershell
# Create the sweep and start an agent in one go
.\.venv\Scripts\python.exe scripts\graph\sweep.py `
    --sweep-config configs\sweeps\graph.yaml `
    --splits data\splits.json `
    --output-dir checkpoints\sweeps\graph `
    --device cuda:0 --epochs 10 --count 20

# ...or attach an agent to a sweep created earlier with `wandb sweep`
.\.venv\Scripts\python.exe scripts\graph\sweep.py `
    --sweep-id <entity/project/sweep_id> `
    --splits data\splits.json `
    --output-dir checkpoints\sweeps\graph `
    --device cuda:0 --epochs 10 --count 20
```

Fixed arguments (paths, device, epoch count, visual cache, validation
settings) stay constant across all trials; per-trial hyperparameters
(`learning_rate`, `hidden_dim`, …) are injected by the W&B agent via
`wandb.config` and applied through the variant's `_build_argparser`
defaults dict. Each trial writes its checkpoints to
`<output-dir>/<run_id>/` so concurrent agents never collide. The
shared agent machinery lives in
[`pcspot.train.sweep_common`](pcspot/train/sweep_common.py); each
`scripts/<variant>/sweep.py` only wires it to that variant's
argparser/`run` pair.

---

## 6. Run inference

Two modes share the same JSON output schema:

### Offline (cached visual features, fast)

```powershell
.\.venv\Scripts\python.exe scripts\infer.py offline `
    --checkpoint checkpoints/run1/epoch_0004.pt `
    --config config.toml --match-id game_18 `
    --visual-cache .cache/visual --visual-backbone dinov2_vits14 `
    --window-size 128 --stride 96 `
    --decode-threshold 0.5 `
    --nms-mode per_player_class --nms-radius 12 `
    --out predictions/game_18.json
```

### Online (streaming, video + tactical HDF5)

```powershell
.\.venv\Scripts\python.exe scripts\infer.py online `
    --checkpoint checkpoints/run1/epoch_0004.pt `
    --config config.toml --match-id game_18 `
    --video data/pcbas_one_match/extracted/.../game_18.mp4 `
    --window-size 128 --stride 32 `
    --device cuda:0 `
    --out predictions/game_18_online.json
```

Use `--use-stub` to run online inference without DINOv2 weights (smoke
test only). The online path is memory-bounded by `window_size` regardless
of video length; see [`pcspot/inference/online.py`](pcspot/inference/online.py)
for the buffer-eviction details.

### Output schema

Predictions are written as a JSON array of positional rows
`[frame, team, jersey_number, class_id, score]`, sorted by `(frame,
team, jersey_number)`:

```json
[
  [12345, 0, 10, 2, 0.91]
]
```

`class_id` is the 1-based PCBAS action class id (see `PCBAS_CLASS_NAMES`;
e.g. `2` = Pass).

### Build a Codabench submission

Submission packaging is bundled. The writer
[`scripts/write_codabench_submission.py`](scripts/write_codabench_submission.py)
takes a directory of per-half prediction JSON files (the format emitted
by `scripts/infer.py`) and produces a `submission.zip` containing a
single `predictions.json` mapping each match id to its flat list of
prediction rows:

```powershell
.\.venv\Scripts\python.exe scripts\write_codabench_submission.py `
    --predictions predictions\ `
    --out submission.zip
```

Pass `--validate-only` to build the per-match documents and run schema
validation without writing the zip (useful in CI); `--report
report.json` writes per-match counts and any validation errors.

---

## 7. Architecture

```
StackedSample
  -> PlayerNodeEmbedder
       (kinematics + role + team + time + optional visual
        + optional jersey + optional [goal distances, degree counts])
  -> HGTEncoder
       player nodes: self / same_team / opponent / near / radius edges
                     + edge features (dist, dx/dy, dv, closing, same_team)
       zone nodes (6x4 grid, optional): SUM-aggregated player->zone,
                     8-neighborhood diffusion, team-oriented zone->player
                     read-back with explicit own/opp counts
  -> PlayerMSTCN (MS-TCN++, dual-dilated stage 1, refinement stages)
  -> PlayerActionHead (class logits + optional objectness)
  -> decode_predictions + player_centric_nms
  -> {time, class, player, score}
```

HGT is implemented in pure PyTorch (no `torch_geometric` dependency)
over dense `(B, T, P, P)` attention masks — `P` is at most ~22 in
soccer, so dense attention is cheap and the implementation stays
self-contained. Zone nodes are **SUM-aggregated** (not averaged) in
physical pitch coordinates rather than the attacking-canonical frame:
sum is the only aggregation that preserves cardinality, which is what
makes "is my area congested?" representable at all, and splatting in
physical coordinates keeps per-team zone counts reflecting actual
on-pitch congestion regardless of attacking direction.

### Model variants and ablations

The diagram above is the **`graph`** variant — the full model. Two
sibling variants exist purely to measure the marginal contribution of
each graph component; train one run of each (everything else equal,
matching `configs/train/<variant>/baseline.toml`) and compare
validation metrics:

| Variant | Class | Pipeline | Isolates |
| --- | --- | --- | --- |
| `graph` | [`PlayerCentricSpottingModel`](pcspot/models/graph_model.py) | embedder → HGT (player + zone nodes) → MS-TCN++ → head | The full model — baseline for the other two. |
| `no_zones` | [`NoZonesSpottingModel`](pcspot/models/no_zones_model.py) | embedder → HGT (player nodes only) → MS-TCN++ → head | The marginal value of **zone nodes** alone. The player-player graph (incl. `radius` edges and degree counts) is retained; `use_zone_nodes` is hard-wired off. |
| `no_graph` | [`NoGraphSpottingModel`](pcspot/models/no_graph_model.py) | embedder → (no HGT) → MS-TCN++ → head | The marginal value of **inter-player message passing** altogether — a pure per-player tower with no edge attention, zone nodes, or team/frame pooling of any kind. |

All three share the embedder, MS-TCN++ stack, action head, output-dict
contract, loss, and validation pipeline — `no_zones` and `no_graph`
are subclasses / siblings of `PlayerCentricSpottingModel` that simply
construct less of the graph, so the `Trainer`, CLI, sweep, and eval
machinery work identically across all three (see [§5
“Train”](#5-train)). Note that turning a variant's architecture flags
off changes the checkpoint shape — each variant is its own checkpoint
family and cannot load another variant's weights.

---

## 8. Known gaps and current limitations

- **Per-class CALF tuning** (`per_class_window`,
  `per_class_positive_weight`) is exposed but not populated. PCBAS
  classes have very different dynamics (Throw-in vs Shot) and tuning
  here is one of the highest-leverage sweeps.
- **No ball-node / temporal-graph edges.** FOOTPASS does not ship
  per-frame ball positions; MS-TCN++ handles temporal propagation for
  now.
- **Single-process trainer.** AMP (`--amp`, bf16/fp16) and gradient
  accumulation (`--grad-accum-steps`) are wired, but DDP / multi-GPU is
  not. The loss and dataset shapes are DDP-friendly; layering
  `torch.distributed` on top is straightforward when needed.
- **Tracklet identity quality.** Identity switches in the FOOTPASS
  tracker will pollute the responsible-player supervision. The jersey
  embedding (`--use-jersey`) partially mitigates this because
  `shirt_number` typically survives an ID swap, but a future iteration
  should also down-weight events whose tracklet is unstable.

---

## 9. Security and data handling

- **Never commit `config.toml`.** It contains a Hugging Face token and
  the SoccerNet NDA password. `config.example.toml` is the safe template.
- **Use a read-scoped Hugging Face token.** Rotate it immediately if it
  has ever been exposed in source control.
- **Do not redistribute extracted videos.** The SoccerNet NDA explicitly
  forbids redistributing broadcast footage; the `extracted/` directory
  is gitignored for that reason.
- The mirror script reports unextractable archives in
  `data/pcbas_one_match/manifest.json` under `skipped`, so a missing
  NDA password fails loudly rather than silently producing a partial
  mirror.
