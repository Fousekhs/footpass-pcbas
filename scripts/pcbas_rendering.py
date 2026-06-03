"""Rendering helpers for PCBAS visualizations.

The broadcast overlay relies on the ``supervision`` package for box and
label annotation; the tactical radar uses plain OpenCV drawing on a
custom pitch canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import supervision as sv

from pcbas_data import (
    COL,
    class_label,
    filter_valid_rois,
    role_label,
    scale_roi_to_video,
    team_of_player,
)

TEAM_COLORS_BGR = {
    0: (235, 110, 52),
    1: (60, 60, 220),
    -1: (180, 180, 180),
}

EVENT_HIGHLIGHT_BGR = (0, 255, 255)
PITCH_GREEN_BGR = (60, 130, 60)
PITCH_LINE_BGR = (235, 235, 235)
BALL_BGR = (255, 255, 255)


def make_team_color_palette() -> sv.ColorPalette:
    return sv.ColorPalette(
        colors=[
            sv.Color(b=235, g=110, r=52),
            sv.Color(b=60, g=60, r=220),
            sv.Color(b=180, g=180, r=180),
        ]
    )


def rows_to_detections(rows: np.ndarray, video_w: int, video_h: int):
    """Build a ``sv.Detections`` from PCBAS rows that have valid ROIs.

    Returns (detections, labels, is_event_mask).
    """
    rows = filter_valid_rois(rows)
    if rows.shape[0] == 0:
        return sv.Detections.empty(), [], np.zeros((0,), dtype=bool)

    xyxy = scale_roi_to_video(rows, video_w, video_h)
    teams = np.array([team_of_player(p) for p in rows[:, COL["player_id"]]], dtype=int)
    class_ids = np.where(teams < 0, 2, teams)

    detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_ids.astype(int),
        confidence=np.ones((rows.shape[0],), dtype=np.float32),
    )

    labels: list[str] = []
    for row in rows:
        shirt = row[COL["shirt_number"]]
        role = row[COL["role_id"]]
        team = team_of_player(row[COL["player_id"]])
        team_tag = "L" if team == 0 else ("R" if team == 1 else "?")
        shirt_txt = "?" if np.isnan(shirt) else str(int(shirt))
        labels.append(f"{team_tag} #{shirt_txt} {role_label(role)}")

    is_event = rows[:, COL["class"]] != 0
    return detections, labels, is_event


def draw_event_banner(
    frame: np.ndarray, frame_idx: int, event_rows: np.ndarray
) -> np.ndarray:
    if event_rows.shape[0] == 0:
        text = f"frame {frame_idx}"
    else:
        parts = [f"frame {frame_idx}"]
        for row in event_rows[:3]:
            team = team_of_player(row[COL["player_id"]])
            team_tag = "L" if team == 0 else "R"
            shirt = row[COL["shirt_number"]]
            shirt_txt = "?" if np.isnan(shirt) else str(int(shirt))
            cls = class_label(row[COL["class"]])
            parts.append(f"{cls}: {team_tag} #{shirt_txt}")
        text = "  |  ".join(parts)

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
    blended = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.putText(
        blended,
        text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return blended


@dataclass
class PitchCanvas:
    width: int = 900
    height: int = 600
    margin: int = 30

    @property
    def pitch_w(self) -> int:
        return self.width - 2 * self.margin

    @property
    def pitch_h(self) -> int:
        return self.height - 2 * self.margin

    def to_canvas(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        y_norm = float(np.clip(y_norm, 0.0, 1.0))
        cx = self.margin + int(x_norm * self.pitch_w)
        cy = self.margin + int(y_norm * self.pitch_h)
        return cx, cy

    def blank(self) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), PITCH_GREEN_BGR, dtype=np.uint8)
        m = self.margin
        cv2.rectangle(
            canvas, (m, m), (self.width - m, self.height - m), PITCH_LINE_BGR, 2
        )
        cv2.line(
            canvas,
            (self.width // 2, m),
            (self.width // 2, self.height - m),
            PITCH_LINE_BGR,
            1,
        )
        cv2.circle(
            canvas, (self.width // 2, self.height // 2), 55, PITCH_LINE_BGR, 1
        )
        cv2.circle(
            canvas, (self.width // 2, self.height // 2), 3, PITCH_LINE_BGR, -1
        )
        box_w = int(self.pitch_w * 0.16)
        box_h = int(self.pitch_h * 0.55)
        box_y1 = self.height // 2 - box_h // 2
        box_y2 = self.height // 2 + box_h // 2
        cv2.rectangle(canvas, (m, box_y1), (m + box_w, box_y2), PITCH_LINE_BGR, 1)
        cv2.rectangle(
            canvas,
            (self.width - m - box_w, box_y1),
            (self.width - m, box_y2),
            PITCH_LINE_BGR,
            1,
        )
        small_w = int(self.pitch_w * 0.06)
        small_h = int(self.pitch_h * 0.30)
        s_y1 = self.height // 2 - small_h // 2
        s_y2 = self.height // 2 + small_h // 2
        cv2.rectangle(canvas, (m, s_y1), (m + small_w, s_y2), PITCH_LINE_BGR, 1)
        cv2.rectangle(
            canvas,
            (self.width - m - small_w, s_y1),
            (self.width - m, s_y2),
            PITCH_LINE_BGR,
            1,
        )
        return canvas


def render_pitch_frame(
    canvas: PitchCanvas,
    rows: np.ndarray,
    frame_idx: int,
    event_rows: Optional[np.ndarray] = None,
    show_velocity: bool = True,
) -> np.ndarray:
    img = canvas.blank()
    if rows.shape[0] == 0:
        cv2.putText(
            img,
            f"frame {frame_idx} (no tactical rows)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return img

    event_player_ids: set[int] = set()
    if event_rows is not None and event_rows.shape[0] > 0:
        event_player_ids = {
            int(p) for p in event_rows[:, COL["player_id"]] if not np.isnan(p)
        }

    for row in rows:
        x = row[COL["x"]]
        y = row[COL["y"]]
        if np.isnan(x) or np.isnan(y):
            continue
        cx, cy = canvas.to_canvas(x, y)
        team = team_of_player(row[COL["player_id"]])
        color = TEAM_COLORS_BGR.get(team, TEAM_COLORS_BGR[-1])
        radius = 7
        is_event_actor = (
            not np.isnan(row[COL["player_id"]])
            and int(row[COL["player_id"]]) in event_player_ids
        )
        if is_event_actor:
            cv2.circle(img, (cx, cy), radius + 4, EVENT_HIGHLIGHT_BGR, 2)
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(img, (cx, cy), radius, (0, 0, 0), 1)

        shirt = row[COL["shirt_number"]]
        if not np.isnan(shirt):
            shirt_txt = str(int(shirt))
            cv2.putText(
                img,
                shirt_txt,
                (cx - 6 * len(shirt_txt) + 4, cy - radius - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if show_velocity:
            vx = row[COL["speed_x"]]
            vy = row[COL["speed_y"]]
            if not (np.isnan(vx) or np.isnan(vy)):
                ex = cx + int(vx * 80)
                ey = cy + int(vy * 80)
                cv2.arrowedLine(
                    img, (cx, cy), (ex, ey), (255, 255, 255), 1, tipLength=0.25
                )

    cv2.rectangle(img, (0, 0), (canvas.width, 28), (0, 0, 0), -1)
    if event_rows is not None and event_rows.shape[0] > 0:
        parts: list[str] = [f"frame {frame_idx}"]
        for row in event_rows[:3]:
            team = team_of_player(row[COL["player_id"]])
            team_tag = "L" if team == 0 else "R"
            shirt = row[COL["shirt_number"]]
            shirt_txt = "?" if np.isnan(shirt) else str(int(shirt))
            parts.append(f"{class_label(row[COL['class']])}: {team_tag} #{shirt_txt}")
        text = "  |  ".join(parts)
    else:
        text = f"frame {frame_idx}"
    cv2.putText(
        img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )

    return img


def make_broadcast_annotators() -> tuple[sv.BoxAnnotator, sv.LabelAnnotator]:
    palette = make_team_color_palette()
    box = sv.BoxAnnotator(color=palette, thickness=2)
    label = sv.LabelAnnotator(
        color=palette,
        text_color=sv.Color.WHITE,
        text_scale=0.4,
        text_thickness=1,
        text_padding=2,
    )
    return box, label


def event_window_mask(rows: np.ndarray, frame_idx: int, window: int) -> np.ndarray:
    """Return mask of rows whose absolute frame distance is within ``window``."""
    if rows.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    diff = np.abs(rows[:, COL["frame"]].astype(np.int64) - frame_idx)
    return diff <= window


def thicken_event_boxes(
    frame: np.ndarray, rows: np.ndarray, video_w: int, video_h: int
) -> np.ndarray:
    """Overdraw thick yellow boxes for current event actors."""
    rows = filter_valid_rois(rows)
    if rows.shape[0] == 0:
        return frame
    xyxy = scale_roi_to_video(rows, video_w, video_h)
    out = frame
    for (x1, y1, x2, y2), row in zip(xyxy, rows):
        cv2.rectangle(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            EVENT_HIGHLIGHT_BGR,
            2,
        )
        cls = class_label(row[COL["class"]])
        team = team_of_player(row[COL["player_id"]])
        team_tag = "L" if team == 0 else "R"
        shirt = row[COL["shirt_number"]]
        shirt_txt = "?" if np.isnan(shirt) else str(int(shirt))
        tag = f"{cls} {team_tag}#{shirt_txt}"
        cv2.putText(
            out,
            tag,
            (int(x1), max(int(y1) - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            EVENT_HIGHLIGHT_BGR,
            1,
            cv2.LINE_AA,
        )
    return out
