# PCBAS / FOOTPASS Mirror Notes

This folder mirrors one split from the **SoccerNet Player-Centric
Ball-Action Spotting (PCBAS)** dataset, also known as **FOOTPASS**
(Footovision Play-by-Play Action Spotting in Soccer). The downloader
lives at [scripts/mirror_pcbas_one_match.py](scripts/mirror_pcbas_one_match.py)
and reads its credentials from [config.toml](config.toml).

> **Note on granularity.** The Hugging Face repository
> `SoccerNet/SN-PCBAS-2026` does not expose single matches as discrete
> files. It only ships split-level archives (one set of zips per
> TRAIN/VAL/CHALLENGE). The smallest distributable unit that contains
> at least one match is therefore the **VAL** split, which bundles
> three matches plus their tactical/play-by-play data. The script
> defaults to that split.

## What the dataset is

FOOTPASS / PCBAS supports the SoccerNet 2026 Player-Centric Ball-Action
Spotting challenge. It is a player-centric, multi-modal, multi-agent
benchmark covering 54 full-length men's soccer broadcasts from the
2023/24 season (Ligue 1, Bundesliga, Serie A, La Liga, UEFA Champions
League), totalling roughly 81 hours of video and 102,992 manually
validated on-ball events.

Splits (per FOOTPASS README):

| Split          | Matches | Events |
| -------------- | ------- | ------ |
| Train          | 48      | 91,327 |
| Validation     | 3       | 6,070  |
| Challenge/Test | 3       | 5,595  |

## What the mirrored data contains

At the Hugging Face level the gated repository exposes these top-level
archives (file names confirmed via the script's dry-run):

- `tactical_data_TRAIN.zip`, `tactical_data_VAL.zip`,
  `tactical_data_CHALLENGE.zip` - per-split play-by-play events,
  tracklets, spatiotemporal data, and team/jersey/role metadata.
- `videos_352x640_TRAIN.zip`, `videos_352x640_VAL.zip`,
  `videos_352x640_CHALLENGE.zip` - downsampled broadcast videos at
  352x640.
- `videos_fullHD_TRAIN_01.zip` .. `_05.zip`,
  `videos_fullHD_VAL.zip`, `videos_fullHD_CHALLENGE.zip` - 1920x1080
  broadcast videos. TRAIN is sharded into 5 parts.
- `tactical_data_format.txt` - format specification for the tactical
  data layout. Mirrored as a global metadata file.

Each archive bundles several matches. Once extracted, every match
typically provides a combination of the following modalities:

- **Broadcast video** at 1920x1080, 25 fps. Distributed via the
  SoccerNet NDA channel; archives are password-protected and the
  password belongs in [config.toml](config.toml) under
  `pcbas.soccernet_password`.
- **Play-by-play events** as tuples `(frame, team, jersey, class)`,
  where:
  - `frame` is the 0-based frame index,
  - `team` is `0` (left) or `1` (right),
  - `jersey` is the player shirt number,
  - `class` is one of the action categories below.
- **Action classes** (12 categories used in PCBAS 2024+): Drive, Pass,
  Cross, Shot, Header, Throw-in, Tackle, Block, plus High Pass, Out,
  Ball Player Block, Player Successful Tackle, Free Kick, Goal in the
  extended set.
- **Single-player tracklets** (sequences of bounding boxes per player
  across frames).
- **Spatiotemporal data** (per-frame player positions and velocities,
  typically 22 players per frame).
- **Team / jersey / role metadata** mapping each tracklet to a team,
  jersey number, and one of 13 tactical roles (Goalkeeper, Left Back,
  Left Central Back, Mid Central Back, Right Central Back, Left
  Midfielder, Right Midfielder, Defensive Midfielder, Attacking
  Midfielder, Left Winger, Right Winger, Central Forward, Right Back).
- **Baseline prediction JSONs** (e.g. under `playbyplay_PRED/`) used as
  reference outputs and submission examples.

## Layout after a successful mirror

```
data/pcbas_one_match/
  raw/           # archives downloaded verbatim from the HF repo
  extracted/     # contents of any password-protected archives
  manifest.json  # what was downloaded, extracted, or skipped
```

Defaults:

- Split: `VAL` (3 matches, smallest).
- Video resolution: `352x640` (much smaller than fullHD).
- Override with `--split TRAIN|VAL|CHALLENGE` and
  `--resolution 352x640|fullHD`.

## Access requirements

The dataset is gated at two independent levels and both must be cleared
before the script can actually pull data:

1. **Hugging Face dataset terms.** Visit
   <https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026> while
   logged in, accept the dataset terms, then create a read-scoped
   access token at <https://huggingface.co/settings/tokens> and put
   it into `pcbas.huggingface_token` in [config.toml](config.toml).
2. **SoccerNet NDA.** Sign the SoccerNet NDA form linked from the
   FOOTPASS README. Once you receive the password by email, put it
   into `pcbas.soccernet_password` in [config.toml](config.toml). The
   script uses it to attempt extraction of `.zip` (AES) and `.7z`
   archives.

If either credential is missing or invalid, the script will:

- abort with a clear error during the Hugging Face listing step if
  the dataset is unreachable, or
- download what it can but record archives it could not extract in
  `data/pcbas_one_match/manifest.json` under `skipped`.

## Visualizations

Two renderers are provided once a split has been mirrored and extracted:

- `scripts/render_pcbas_visualizations.py` - CLI entry point.
- `scripts/pcbas_data.py` - loader for the tactical HDF5 + videos.
- `scripts/pcbas_rendering.py` - `supervision` and OpenCV drawing helpers.

### Render commands

List discovered matches:

```powershell
.\.venv\Scripts\python.exe scripts\render_pcbas_visualizations.py --config config.toml --list-matches
```

Short sample render (default 500 frames, both modes):

```powershell
.\.venv\Scripts\python.exe scripts\render_pcbas_visualizations.py --config config.toml --match-id game_18
```

Pick a frame window and only one mode:

```powershell
.\.venv\Scripts\python.exe scripts\render_pcbas_visualizations.py --config config.toml --match-id game_18 --render broadcast --start-frame 50 --max-frames 300
```

Render an entire match (slow):

```powershell
.\.venv\Scripts\python.exe scripts\render_pcbas_visualizations.py --config config.toml --match-id game_18 --full-match
```

### Outputs

Written to `data/pcbas_one_match/renders/`:

- `<match>_broadcast_overlay.mp4` - original broadcast with player
  bounding boxes (`L #10 LW` style labels), thick yellow boxes for
  event actors, and a top banner showing the current frame and any
  active action like `Pass: L #10`.
- `<match>_tactical_radar.mp4` - top-down 900x600 pitch with all 22
  players drawn from the normalized pitch coordinates, team colors
  (blue=left squad ids 1xx, red=right squad ids 2xx), shirt numbers,
  velocity arrows, and event actors highlighted with a yellow ring.
- `<match>_render_manifest.json` - records source paths, frame range,
  event counts, and output paths.

### Tactical data schema used

`tactical_data_<SPLIT>/<split>_tactical_data.h5` holds one dataset per
match half (`game_18_H1`, `game_18_H2`, ...). Each is an `(N, 14)`
float32 matrix:

| Col | Field | Notes |
| --- | ----- | ----- |
| 0 | `frame` | 0-based frame index into the corresponding video |
| 1 | `player_id` | Tracklet id; 1xx = left squad, 2xx = right squad |
| 2 | `left_to_right` | Attack direction in the current half |
| 3 | `shirt_number` | Jersey number |
| 4 | `role_id` | Tactical role (1..13, see role table below) |
| 5-6 | `x`, `y` | Normalized pitch coordinates (~`[0, 1]`) |
| 7-8 | `speed_x`, `speed_y` | Per-axis velocity |
| 9-12 | `roi_x`, `roi_y`, `roi_width`, `roi_height` | Pixel bbox in the original fullHD broadcast (1920x1080), NaN when the player is off-screen |
| 13 | `class` | `0` = no event, `1..8` = on-ball action class |

Roles: 1 GK, 2 LB, 3 LCB, 4 MCB, 5 RCB, 6 LM, 7 RM, 8 DM, 9 AM,
10 LW, 11 RW, 12 CF, 13 RB.

Classes used in the VAL split: 1 Drive, 2 Pass, 3 Cross, 4 Shot,
5 Header, 6 Throw-in, 7 Tackle, 8 Block.

### Limitations

- ROI coordinates are authored against the fullHD broadcast. When
  rendering the downsampled 352x640 video, the renderer rescales
  them automatically by `640/1920` and `352/1080`.
- The broadcast renderer needs the corresponding video file. The
  radar renderer can run from tactical data alone.
- Rendering an entire match is slow and produces large files; use
  `--start-frame` / `--max-frames` for previews and only pass
  `--full-match` when you really want the whole 25 fps output.

## Not covered by this mirror

- Full 500-match SoccerNet-v2 action-spotting corpus.
- Pre-extracted ResNet / Baidu features used by other SoccerNet tasks.
- Codabench evaluation submission flow (challenge ground truth is
  hidden by design).
- Any redistribution of SoccerNet videos: the NDA forbids
  redistributing the broadcast footage, so the contents of
  `extracted/` must not be checked in or shared.
