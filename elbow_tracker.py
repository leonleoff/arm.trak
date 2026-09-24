"""Live elbow angle tracker.

Detects the upper body with MediaPipe's PoseLandmarker, measures the angle
enclosed between upper arm (shoulder -> elbow) and forearm (elbow -> wrist) for
both arms, and draws the result on a live webcam view.

The angle is computed from MediaPipe's *world* landmarks, i.e. in metric 3D
space, so it stays correct even when the arm points towards or away from the
camera. The arc drawn around the elbow is the projection onto the image plane
and is therefore only an illustration -- the number next to it is the real
3D angle.

Keys:
    q / ESC   quit
    r         reset the min/max range-of-motion counters
    p         toggle the angle history plot
    m         toggle mirrored view
"""

from __future__ import annotations

import argparse
import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

PoseLandmark = vision.PoseLandmark

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = MODEL_DIR / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "{variant}/float16/latest/{variant}.task"
)

# BGR colours, one per arm.
COLOR_LEFT = (255, 196, 64)
COLOR_RIGHT = (80, 160, 255)
COLOR_HUD = (240, 240, 240)
COLOR_DIM = (140, 140, 140)

ARMS = {
    "L": (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    "R": (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
}
ARM_COLORS = {"L": COLOR_LEFT, "R": COLOR_RIGHT}
ARM_NAMES = {"L": "Left arm", "R": "Right arm"}


@dataclass
class ArmState:
    """Per-arm smoothed angle, range of motion and angle history."""

    smoothed: float | None = None
    min_angle: float | None = None
    max_angle: float | None = None
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=4096))

    def update(self, angle: float, now: float, alpha: float) -> float:
        if self.smoothed is None:
            self.smoothed = angle
        else:
            self.smoothed = alpha * angle + (1.0 - alpha) * self.smoothed
        value = self.smoothed
        self.min_angle = value if self.min_angle is None else min(self.min_angle, value)
        self.max_angle = value if self.max_angle is None else max(self.max_angle, value)
        self.history.append((now, value))
        return value

    def drop(self) -> None:
        """Called when the arm is not visible: forget the smoothing state."""
        self.smoothed = None

    def reset_rom(self) -> None:
        self.min_angle = None
        self.max_angle = None

    def trim(self, cutoff: float) -> None:
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()


def angle_3d(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float | None:
    """Angle at the elbow in degrees, 0 = fully folded, 180 = fully extended."""
    upper = shoulder - elbow
    fore = wrist - elbow
    norm = float(np.linalg.norm(upper) * np.linalg.norm(fore))
    if norm < 1e-9:
        return None
    cosine = float(np.dot(upper, fore) / norm)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def to_pixels(landmark, width: int, height: int, mirror: bool) -> tuple[int, int]:
    x = 1.0 - landmark.x if mirror else landmark.x
    return int(round(x * width)), int(round(landmark.y * height))


def draw_arm(frame: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    shoulder, elbow, wrist = points
    cv2.line(frame, shoulder, elbow, color, 4, cv2.LINE_AA)
    cv2.line(frame, elbow, wrist, color, 4, cv2.LINE_AA)
    for point, radius in ((shoulder, 6), (elbow, 9), (wrist, 6)):
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, (20, 20, 20), 2, cv2.LINE_AA)


def draw_angle_arc(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    angle: float,
    color: tuple[int, int, int],
) -> None:
    """Draw the elbow arc plus the angle label, both in the image plane."""
    shoulder, elbow, wrist = points
    to_shoulder = np.array(shoulder, dtype=float) - elbow
    to_wrist = np.array(wrist, dtype=float) - elbow
    len_shoulder = float(np.linalg.norm(to_shoulder))
    len_wrist = float(np.linalg.norm(to_wrist))
    if len_shoulder < 1e-3 or len_wrist < 1e-3:
        return

    radius = int(max(22.0, min(len_shoulder, len_wrist) * 0.32))
    start = math.degrees(math.atan2(to_shoulder[1], to_shoulder[0]))
    end = math.degrees(math.atan2(to_wrist[1], to_wrist[0]))
    sweep = (end - start + 180.0) % 360.0 - 180.0
    cv2.ellipse(frame, elbow, (radius, radius), 0.0, start, start + sweep, color, 3, cv2.LINE_AA)

    # Place the label outside the arc, along the bisector of the two arm segments.
    bisector = to_shoulder / len_shoulder + to_wrist / len_wrist
    if float(np.linalg.norm(bisector)) < 1e-3:
        bisector = np.array([-to_shoulder[1], to_shoulder[0]], dtype=float)
    bisector /= float(np.linalg.norm(bisector))
    label_pos = np.array(elbow, dtype=float) + bisector * (radius + 34.0)
    draw_label(frame, f"{angle:.1f}{chr(176)}", (int(label_pos[0]), int(label_pos[1])), color)


def draw_label(
    frame: np.ndarray,
    text: str,
    center: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.75,
) -> None:
    """Draw centred text with a dark backing box so it stays readable."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(np.clip(center[0] - tw // 2, 10, max(10, frame.shape[1] - tw - 10)))
    y = int(np.clip(center[1] + th // 2, th + 10, max(th + 10, frame.shape[0] - baseline - 6)))
    cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + baseline + 4), (25, 25, 25), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    states: dict[str, ArmState],
    visible: dict[str, float],
    pose_found: bool,
) -> None:
    lines: list[tuple[str, tuple[int, int, int]]] = [(f"FPS {fps:4.1f}", COLOR_HUD)]
    for key, state in states.items():
        color = ARM_COLORS[key]
        if key in visible:
            rom = ""
            if state.min_angle is not None and state.max_angle is not None:
                rom = f"   min {state.min_angle:5.1f}   max {state.max_angle:5.1f}"
            lines.append((f"{ARM_NAMES[key]:<10} {visible[key]:5.1f}{chr(176)}{rom}", color))
        else:
            lines.append((f"{ARM_NAMES[key]:<10}   --", COLOR_DIM))
    if not pose_found:
        lines.append(("No pose detected - step back so your torso is in frame", (80, 80, 255)))

    overlay = frame.copy()
    box_height = 18 + 30 * len(lines)
    box_width = min(560, frame.shape[1] - 20)
    cv2.rectangle(overlay, (10, 10), (box_width, 10 + box_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame)
    for index, (text, color) in enumerate(lines):
        position = (24, 42 + 30 * index)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_plot(frame: np.ndarray, states: dict[str, ArmState], now: float, window: float) -> None:
    """Rolling angle-over-time graph in the lower right corner."""
    height, width = frame.shape[:2]
    plot_w, plot_h = min(420, width - 40), 170
    if plot_w < 120 or height < plot_h + 60:
        return
    x0, y0 = width - plot_w - 20, height - plot_h - 40
    x1, y1 = x0 + plot_w, y0 + plot_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0.0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70), 1, cv2.LINE_AA)

    pad_left, pad = 42, 12
    inner_x0, inner_x1 = x0 + pad_left, x1 - pad
    inner_y0, inner_y1 = y0 + pad, y1 - pad

    def to_y(angle: float) -> int:
        return int(inner_y1 - (angle / 180.0) * (inner_y1 - inner_y0))

    for tick in (0, 45, 90, 135, 180):
        y = to_y(tick)
        cv2.line(frame, (inner_x0, y), (inner_x1, y), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(
            frame, f"{tick:3d}", (x0 + 6, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_DIM, 1, cv2.LINE_AA,
        )

    start = now - window
    for key, state in states.items():
        samples = [(t, a) for t, a in state.history if t >= start]
        if len(samples) < 2:
            continue
        points = np.array(
            [
                [inner_x0 + (t - start) / window * (inner_x1 - inner_x0), to_y(a)]
                for t, a in samples
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], False, ARM_COLORS[key], 2, cv2.LINE_AA)

    cv2.putText(
        frame, f"last {window:.0f}s", (inner_x1 - 68, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DIM, 1, cv2.LINE_AA,
    )


def ensure_model(path: Path) -> Path:
    """Download the pose model on first run if it is not present yet."""
    if path.exists():
        return path
    variant = path.stem
    if not variant.startswith("pose_landmarker_"):
        raise SystemExit(
            f"Model file {path} not found and its name does not match a known "
            "MediaPipe variant (pose_landmarker_lite/full/heavy). Download it manually."
        )
    url = MODEL_URL.format(variant=variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model {variant} from {url} ...")
    temporary = path.with_suffix(".download")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)
    print(f"Saved to {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="requested capture width")
    parser.add_argument("--height", type=int, default=720, help="requested capture height")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="pose_landmarker_{lite,full,heavy}.task file; downloaded if missing",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.45,
        help="angle smoothing factor, 1.0 = no smoothing (default: 0.45)",
    )
    parser.add_argument(
        "--min-visibility", type=float, default=0.5,
        help="minimum landmark visibility required to measure an arm (default: 0.5)",
    )
    parser.add_argument(
        "--plot-seconds", type=float, default=10.0,
        help="time window of the angle history plot (default: 10)",
    )
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the view")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # CAP_DSHOW opens noticeably faster than the default backend on Windows.
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {index}. Try a different --camera index.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def build_landmarker(model_path: Path) -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main() -> None:
    args = parse_args()
    model_path = ensure_model(args.model)
    capture = open_camera(args.camera, args.width, args.height)

    states = {key: ArmState() for key in ARMS}
    mirror = not args.no_mirror
    show_plot = True
    fps = 0.0
    start_time = time.perf_counter()
    last_time = start_time
    last_stamp_ms = -1
    window_name = "Elbow angle tracker"

    landmarker = build_landmarker(model_path)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame could not be read, stopping.")
                break

            # Detect on the unmirrored frame so MediaPipe's left/right labels stay
            # anatomically correct, then mirror only for display.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now = time.perf_counter()
            # detect_for_video requires strictly increasing timestamps.
            stamp_ms = max(int((now - start_time) * 1000.0), last_stamp_ms + 1)
            last_stamp_ms = stamp_ms
            result = landmarker.detect_for_video(image, stamp_ms)

            if mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            visible: dict[str, float] = {}
            pose_found = bool(result.pose_landmarks) and bool(result.pose_world_landmarks)

            if pose_found:
                image_landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0]
                for key, joints in ARMS.items():
                    image_points = [image_landmarks[j] for j in joints]
                    if min(p.visibility for p in image_points) < args.min_visibility:
                        states[key].drop()
                        continue

                    world_points = [
                        np.array(
                            [world_landmarks[j].x, world_landmarks[j].y, world_landmarks[j].z],
                            dtype=float,
                        )
                        for j in joints
                    ]
                    raw = angle_3d(*world_points)
                    if raw is None:
                        states[key].drop()
                        continue

                    angle = states[key].update(raw, now, args.smoothing)
                    visible[key] = angle

                    pixels = [to_pixels(p, width, height, mirror) for p in image_points]
                    draw_arm(frame, pixels, ARM_COLORS[key])
                    draw_angle_arc(frame, pixels, angle, ARM_COLORS[key])
            else:
                for state in states.values():
                    state.drop()

            cutoff = now - max(args.plot_seconds, 1.0)
            for state in states.values():
                state.trim(cutoff)

            draw_hud(frame, fps, states, visible, pose_found)
            if show_plot:
                draw_plot(frame, states, now, args.plot_seconds)
            cv2.putText(
                frame, "q quit   r reset min/max   p plot   m mirror",
                (24, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM, 1, cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                for state in states.values():
                    state.reset_rom()
            elif key == ord("p"):
                show_plot = not show_plot
            elif key == ord("m"):
                mirror = not mirror
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        landmarker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
