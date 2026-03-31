"""
Neural-Loom Eye Tracking Module
================================
Uses OpenCV + MediaPipe Face Mesh to detect:
  - Blink rate
  - Eye openness (gaze fatigue)
  - Estimated gaze direction

Classifies engagement as: "focused" | "confused" | "bored"

This module runs locally. RAW FRAMES ARE NEVER STORED OR TRANSMITTED.
Only the classified engagement state string is sent to the server.

Usage (standalone test):
    python models/eye_tracking_model/tracker.py
"""

import time
import math
import threading
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for eyes (MediaPipe Face Mesh)
# Left eye: vertical landmarks 159 (top), 145 (bottom); horizontal 33, 133
# Right eye: vertical landmarks 386 (top), 374 (bottom); horizontal 362, 263
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133

RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

# Iris landmarks for gaze direction
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def eye_aspect_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx, img_w, img_h):
    """Compute Eye Aspect Ratio (EAR) — low EAR means closed eye."""
    top    = landmarks[top_idx]
    bottom = landmarks[bottom_idx]
    left   = landmarks[left_idx]
    right  = landmarks[right_idx]

    vertical   = math.dist((top.x * img_w,    top.y * img_h),
                            (bottom.x * img_w, bottom.y * img_h))
    horizontal = math.dist((left.x * img_w,   left.y * img_h),
                            (right.x * img_w,  right.y * img_h))

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def iris_position_ratio(landmarks, iris_indices, left_idx, right_idx, img_w, img_h):
    """
    Returns normalised iris X position [0..1].
    0 = far left, 0.5 = centre, 1 = far right.
    """
    iris_center_x = np.mean([landmarks[i].x for i in iris_indices]) * img_w
    eye_left_x    = landmarks[left_idx].x * img_w
    eye_right_x   = landmarks[right_idx].x * img_w
    eye_width     = eye_right_x - eye_left_x
    if eye_width == 0:
        return 0.5
    return (iris_center_x - eye_left_x) / eye_width


class EngagementTracker:
    """
    Runs eye tracking in a background thread.
    Exposes `.state` property: "focused" | "confused" | "bored"
    and `.running` flag.
    """

    EAR_BLINK_THRESHOLD   = 0.20   # Below this → eye closed (blink)
    EAR_DROWSY_THRESHOLD  = 0.24   # Consistently low → drowsy/confused
    BLINK_CONSEC_FRAMES   = 2      # Min frames for blink
    GAZE_OFF_THRESHOLD    = 0.30   # Iris ratio far from centre → looking away
    WINDOW_SECONDS        = 10     # Sliding window for metric aggregation
    BORED_BLINK_RATE      = 25     # Blinks/min above this → bored
    CONFUSED_BLINK_RATE   = 5      # Blinks/min below this → intense focus / confused

    def __init__(self):
        self.state   = "focused"
        self.running = False
        self._thread = None

        # Sliding windows
        self._ear_history        = deque()   # (timestamp, ear)
        self._blink_times        = deque()   # timestamps of blinks
        self._gaze_off_history   = deque()   # (timestamp, bool)

        self._blink_counter      = 0         # consecutive low-EAR frames
        self._last_state_change  = time.time()

    # ------------------------------------------------------------------
    def start(self):
        """Start background tracking thread."""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------
    def _run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[EyeTracker] Cannot open webcam — defaulting to 'focused'")
            self.running = False
            return

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,          # needed for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # ---- Process frame (never stored) ----
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm     = results.multi_face_landmarks[0].landmark
                    h, w   = frame.shape[:2]
                    now    = time.time()

                    # EAR
                    left_ear  = eye_aspect_ratio(lm, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                                                  LEFT_EYE_LEFT, LEFT_EYE_RIGHT, w, h)
                    right_ear = eye_aspect_ratio(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                                  RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)
                    avg_ear   = (left_ear + right_ear) / 2.0

                    # Blink detection
                    if avg_ear < self.EAR_BLINK_THRESHOLD:
                        self._blink_counter += 1
                    else:
                        if self._blink_counter >= self.BLINK_CONSEC_FRAMES:
                            self._blink_times.append(now)
                        self._blink_counter = 0

                    # Gaze direction (iris)
                    try:
                        left_iris_ratio  = iris_position_ratio(lm, LEFT_IRIS,
                                                                LEFT_EYE_LEFT, LEFT_EYE_RIGHT, w, h)
                        right_iris_ratio = iris_position_ratio(lm, RIGHT_IRIS,
                                                                RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)
                        avg_iris = (left_iris_ratio + right_iris_ratio) / 2.0
                        gaze_off = abs(avg_iris - 0.5) > self.GAZE_OFF_THRESHOLD
                    except Exception:
                        gaze_off = False

                    # EAR history
                    self._ear_history.append((now, avg_ear))
                    self._gaze_off_history.append((now, gaze_off))

                    # Trim old entries outside sliding window
                    cutoff = now - self.WINDOW_SECONDS
                    while self._ear_history and self._ear_history[0][0] < cutoff:
                        self._ear_history.popleft()
                    while self._blink_times and self._blink_times[0] < cutoff:
                        self._blink_times.popleft()
                    while self._gaze_off_history and self._gaze_off_history[0][0] < cutoff:
                        self._gaze_off_history.popleft()

                    # Classify
                    self.state = self._classify()

        cap.release()
        self.running = False

    # ------------------------------------------------------------------
    def _classify(self) -> str:
        """
        Heuristic classifier using blink rate + gaze + EAR.
        Returns "focused" | "confused" | "bored"
        """
        if not self._ear_history:
            return "focused"

        # Blink rate (blinks per minute)
        blink_rate = len(self._blink_times) * (60.0 / self.WINDOW_SECONDS)

        # Mean EAR over window
        mean_ear = np.mean([e for _, e in self._ear_history])

        # Fraction of time gaze was off-centre
        gaze_off_frac = (
            sum(1 for _, g in self._gaze_off_history if g)
            / max(len(self._gaze_off_history), 1)
        )

        # --- Decision rules ---
        # Bored: high blink rate + frequent gaze aversion
        if blink_rate > self.BORED_BLINK_RATE or gaze_off_frac > 0.5:
            return "bored"

        # Confused: very low blink rate (staring hard) + slightly low EAR
        if blink_rate < self.CONFUSED_BLINK_RATE and mean_ear < self.EAR_DROWSY_THRESHOLD:
            return "confused"

        return "focused"

    # ------------------------------------------------------------------
    @property
    def metrics(self) -> dict:
        """Return raw metrics for debugging."""
        blink_rate = len(self._blink_times) * (60.0 / self.WINDOW_SECONDS)
        mean_ear   = float(np.mean([e for _, e in self._ear_history])) if self._ear_history else 0.0
        gaze_off_frac = (
            sum(1 for _, g in self._gaze_off_history if g)
            / max(len(self._gaze_off_history), 1)
        )
        return {
            "blink_rate_per_min": round(blink_rate, 1),
            "mean_ear":           round(mean_ear, 3),
            "gaze_off_fraction":  round(gaze_off_frac, 2),
            "state":              self.state,
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tracker = EngagementTracker()
    tracker.start()
    print("Tracking... press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(2)
            print(tracker.metrics)
    except KeyboardInterrupt:
        tracker.stop()
        print("Stopped.")
