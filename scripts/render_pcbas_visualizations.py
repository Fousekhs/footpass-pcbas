"""Render PCBAS broadcast overlays and tactical radar videos.

Reads the extracted PCBAS/FOOTPASS split produced by
``mirror_pcbas_one_match.py``, picks a match, and writes annotated
videos under ``data/pcbas_one_match/renders/``.

Usage:
    python scripts/render_pcbas_visualizations.py --config config.toml --list-matches
    python scripts/render_pcbas_visualizations.py --config config.toml --match-id game_18 --max-frames 500
    python scripts/render_pcbas_visualizations.py --config config.toml --match-id game_18 \
        --render broadcast --render radar --start-frame 1000 --max-frames 1500
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import supervision as sv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcbas_data import (  # noqa: E402
    COL,
    MatchAssets,
    PCBASConfig,
    concat_match_array,
    frame_range,
    index_by_frame,
    list_matches,
    select_events,
)
from pcbas_rendering import (  # noqa: E402
    PitchCanvas,
    draw_event_banner,
    event_window_mask,
    make_broadcast_annotators,
    render_pitch_frame,
    rows_to_detections,
    thicken_event_boxes,
)


RENDER_MODES = ("broadcast", "radar")
EVENT_HIGHLIGHT_FRAMES = 12  # +/- frames an event stays highlighted


@dataclass
class RenderPlan:
    match: MatchAssets
    start_frame: int
    end_frame: int
    modes: tuple[str, ...]
    out_dir: Path
    max_frames: Optional[int]
    full_match: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument(
        "--match-id",
        type=str,
        default=None,
        help="Match identifier to render (e.g. game_18). Use --list-matches to see options.",
    )
    p.add_argument(
        "--list-matches",
        action="store_true",
        help="List available matches and exit.",
    )
    p.add_argument(
        "--render",
        action="append",
        choices=RENDER_MODES,
        help="Visualization mode(s) to render. May be passed multiple times. "
        "Default: both broadcast and radar.",
    )
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to render. Default 500 keeps test renders short. "
        "Use --full-match to render an entire match.",
    )
    p.add_argument(
        "--full-match",
        action="store_true",
        help="Render the entire match (overrides --max-frames).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <output_dir>/renders",
    )
    p.add_argument(
        "--no-velocity",
        action="store_true",
        help="Disable velocity arrows on the radar view.",
    )
    return p.parse_args()


def select_match(matches: list[MatchAssets], match_id: Optional[str]) -> MatchAssets:
    if not matches:
        raise SystemExit(
            "No matches discovered. Run scripts/mirror_pcbas_one_match.py first."
        )
    if match_id is None:
        chosen = matches[0]
        print(f"No --match-id provided; using first available: {chosen.match_id}")
        return chosen
    for m in matches:
        if m.match_id == match_id:
            return m
    available = ", ".join(m.match_id for m in matches)
    raise SystemExit(f"Match '{match_id}' not found. Available: {available}")


def resolve_modes(modes: Optional[list[str]]) -> tuple[str, ...]:
    if not modes:
        return RENDER_MODES
    seen: list[str] = []
    for m in modes:
        if m not in seen:
            seen.append(m)
    return tuple(seen)


def build_plan(args: argparse.Namespace, match: MatchAssets, arr: np.ndarray) -> RenderPlan:
    fmin, fmax = frame_range(arr)
    start = args.start_frame if args.start_frame is not None else fmin
    end = args.end_frame if args.end_frame is not None else fmax
    start = max(start, fmin)
    end = min(end, fmax)
    if end < start:
        raise SystemExit(f"Empty frame range after clipping: start={start}, end={end}")
    if not args.full_match and args.max_frames is not None:
        end = min(end, start + args.max_frames - 1)

    out_dir = args.out_dir or (Path("data/pcbas_one_match") / "renders")
    out_dir.mkdir(parents=True, exist_ok=True)

    return RenderPlan(
        match=match,
        start_frame=int(start),
        end_frame=int(end),
        modes=resolve_modes(args.render),
        out_dir=out_dir,
        max_frames=None if args.full_match else args.max_frames,
        full_match=args.full_match,
    )


def render_broadcast(plan: RenderPlan, frame_index: dict[int, np.ndarray]) -> Path:
    src = str(plan.match.video_path)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {src}")
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = plan.out_dir / f"{plan.match.match_id}_broadcast_overlay.mp4"
    info = sv.VideoInfo(width=video_w, height=video_h, fps=int(round(fps)), total_frames=total)
    box_ann, label_ann = make_broadcast_annotators()

    print(
        f"[broadcast] {plan.match.match_id}  {video_w}x{video_h} @ {fps:.2f}fps  "
        f"frames {plan.start_frame}..{plan.end_frame}  ->  {out_path}"
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, plan.start_frame)
    written = 0
    with sv.VideoSink(target_path=str(out_path), video_info=info) as sink:
        for frame_idx in range(plan.start_frame, plan.end_frame + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            rows = frame_index.get(int(frame_idx))
            if rows is None or rows.shape[0] == 0:
                annotated = draw_event_banner(frame, frame_idx, np.empty((0, 14)))
            else:
                detections, labels, is_event = rows_to_detections(rows, video_w, video_h)
                annotated = box_ann.annotate(scene=frame, detections=detections)
                annotated = label_ann.annotate(
                    scene=annotated, detections=detections, labels=labels
                )
                event_rows = rows[rows[:, COL["class"]] != 0]
                if event_rows.shape[0] > 0:
                    annotated = thicken_event_boxes(annotated, event_rows, video_w, video_h)
                annotated = draw_event_banner(annotated, frame_idx, event_rows)

            sink.write_frame(annotated)
            written += 1
    cap.release()
    print(f"[broadcast] wrote {written} frames -> {out_path}")
    return out_path


def render_radar(
    plan: RenderPlan,
    frame_index: dict[int, np.ndarray],
    events_arr: np.ndarray,
    show_velocity: bool,
) -> Path:
    canvas = PitchCanvas(width=900, height=600, margin=30)
    out_path = plan.out_dir / f"{plan.match.match_id}_tactical_radar.mp4"
    info = sv.VideoInfo(width=canvas.width, height=canvas.height, fps=25, total_frames=0)
    print(
        f"[radar] {plan.match.match_id}  pitch {canvas.width}x{canvas.height}  "
        f"frames {plan.start_frame}..{plan.end_frame}  ->  {out_path}"
    )
    events_frames = events_arr[:, COL["frame"]].astype(np.int64) if events_arr.size else np.empty((0,), dtype=np.int64)
    written = 0
    with sv.VideoSink(target_path=str(out_path), video_info=info) as sink:
        for frame_idx in range(plan.start_frame, plan.end_frame + 1):
            rows = frame_index.get(int(frame_idx), np.empty((0, 14), dtype=np.float32))
            mask = np.abs(events_frames - frame_idx) <= EVENT_HIGHLIGHT_FRAMES if events_frames.size else np.zeros((0,), dtype=bool)
            event_rows = events_arr[mask] if events_arr.size else np.empty((0, 14), dtype=np.float32)
            img = render_pitch_frame(
                canvas,
                rows,
                frame_idx=frame_idx,
                event_rows=event_rows,
                show_velocity=show_velocity,
            )
            sink.write_frame(img)
            written += 1
    print(f"[radar] wrote {written} frames -> {out_path}")
    return out_path


def main() -> int:
    args = parse_args()
    cfg = PCBASConfig.load(args.config)

    matches = list_matches(cfg.output_dir)

    if args.list_matches:
        if not matches:
            print("No matches discovered.")
            return 2
        for m in matches:
            arr = concat_match_array(m)
            fmin, fmax = frame_range(arr)
            events = select_events(arr)
            print(
                f"  {m.match_id:<12} halves={len(m.halves)}  rows={arr.shape[0]:>9}  "
                f"events={events.shape[0]:>5}  frames={fmin}..{fmax}  video_ok={m.video_path.exists()}"
            )
        return 0

    match = select_match(matches, args.match_id)
    arr = concat_match_array(match)
    if arr.size == 0:
        raise SystemExit(f"No tactical rows for match {match.match_id}")

    plan = build_plan(args, match, arr)

    sub_mask = (arr[:, COL["frame"]].astype(np.int64) >= plan.start_frame) & (
        arr[:, COL["frame"]].astype(np.int64) <= plan.end_frame
    )
    sub = arr[sub_mask]
    print(
        f"Selected {sub.shape[0]} tactical rows for "
        f"{plan.match.match_id} frames {plan.start_frame}..{plan.end_frame}"
    )
    frame_index = index_by_frame(sub)
    events_arr = select_events(arr)
    events_in_range = events_arr[
        (events_arr[:, COL["frame"]] >= plan.start_frame)
        & (events_arr[:, COL["frame"]] <= plan.end_frame + EVENT_HIGHLIGHT_FRAMES)
    ]
    print(f"Events within window (incl. highlight tail): {events_in_range.shape[0]}")

    outputs: dict[str, str] = {}
    if "broadcast" in plan.modes:
        if not plan.match.video_path.exists():
            print(f"[broadcast] skipped: video file missing at {plan.match.video_path}")
        else:
            outputs["broadcast"] = str(render_broadcast(plan, frame_index))

    if "radar" in plan.modes:
        outputs["radar"] = str(
            render_radar(
                plan,
                frame_index,
                events_arr,
                show_velocity=not args.no_velocity,
            )
        )

    manifest = {
        "match_id": plan.match.match_id,
        "tactical_h5": str(plan.match.tactical_h5),
        "video_path": str(plan.match.video_path),
        "start_frame": plan.start_frame,
        "end_frame": plan.end_frame,
        "frames_rendered": plan.end_frame - plan.start_frame + 1,
        "events_in_range": int(events_in_range.shape[0]),
        "modes": list(plan.modes),
        "outputs": outputs,
    }
    manifest_path = plan.out_dir / f"{plan.match.match_id}_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
