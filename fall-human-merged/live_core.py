"""Time windows, a bounded inference worker, and camera-level alarm state.

This module has no model, camera, or GUI side effects on import.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import queue
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class LiveConfig:
    window_seconds: float = 2.0
    analysis_interval: float = 1.0
    fall_threshold: float = 0.75
    confirm_windows: int = 2
    recovery_windows: int = 2
    cooldown_seconds: float = 10.0
    max_result_age: float = 3.0
    max_sample_gap: float = 0.5

    def __post_init__(self):
        positive = (self.window_seconds, self.analysis_interval,
                    self.max_result_age, self.max_sample_gap)
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Durasi klip, jeda analisis, dan batas usia hasil harus positif.")
        if not 0 < self.fall_threshold <= 1:
            raise ValueError("Ambang jatuh harus lebih dari 0 hingga 1.")
        if self.confirm_windows < 1 or self.recovery_windows < 1:
            raise ValueError("Jumlah konfirmasi/pemulihan minimal satu.")
        if not math.isfinite(self.cooldown_seconds) or self.cooldown_seconds < 0:
            raise ValueError("Jeda antaralarm tidak boleh negatif.")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Clip:
    sequence: int
    start_s: float
    end_s: float
    end_frame: int
    end_wall: float
    frames: tuple[Any, ...]


class ClipBuffer:
    """Keep real, time-spaced frames; never pad missing frames with duplicates."""
    def __init__(self, num_frames: int, config: LiveConfig):
        if num_frames < 2:
            raise ValueError("Model harus menerima minimal dua frame.")
        self.num_frames = num_frames
        self.config = config
        self.interval = config.window_seconds / (num_frames - 1)
        self.samples = deque(maxlen=num_frames)
        self.next_sample = 0.0
        self.next_analysis = None
        self.previous_time = None
        self.sequence = 0
        self.resets = 0

    def offer(self, timestamp: float, frame_index: int, acquired_wall: float,
              make_image: Callable[[], Any]) -> Clip | None:
        if self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError("Timestamp sumber harus terus meningkat.")
        if self.previous_time is not None and timestamp - self.previous_time > self.config.max_sample_gap:
            self.samples.clear()
            self.next_sample = timestamp
            self.next_analysis = None
            self.resets += 1
        self.previous_time = timestamp
        if timestamp + 1e-7 < self.next_sample:
            return None
        self.samples.append((timestamp, frame_index, acquired_wall, make_image()))
        # Keep a sampling grid despite small capture-time jitter. A long gap is
        # handled above instead of duplicating old images to fill the buffer.
        steps = max(1, math.floor((timestamp - self.next_sample + 1e-7) / self.interval) + 1)
        self.next_sample += steps * self.interval
        if len(self.samples) != self.num_frames:
            return None
        start, end = self.samples[0][0], self.samples[-1][0]
        span = end - start
        if not self.config.window_seconds * 0.85 <= span <= self.config.window_seconds + self.config.max_sample_gap:
            return None
        if self.next_analysis is not None and timestamp + 1e-7 < self.next_analysis:
            return None
        self.next_analysis = timestamp + self.config.analysis_interval
        self.sequence += 1
        return Clip(self.sequence, start, end, frame_index, acquired_wall,
                    tuple(sample[3] for sample in self.samples))


class LatestAnalyzer:
    """At most one active and one pending clip. Replace pending work when slow.

    Offline producers can wait for each sequence, preserving all scheduled
    windows. Live producers never wait for inference.
    """
    def __init__(self, predict: Callable, stop: threading.Event):
        self.predict = predict
        self.stop = stop
        self.condition = threading.Condition()
        self.results = queue.Queue()
        self.pending = None
        self.closing = False
        self.done = threading.Event()
        self.error = None
        self.submitted = 0
        self.replaced = 0
        self.completed = 0
        self.completed_sequence = 0
        self.delivered_sequence = 0
        self.thread = threading.Thread(target=self._run, name="videomae-inference", daemon=True)
        self.thread.start()

    def submit(self, clip: Clip):
        with self.condition:
            if self.closing or self.stop.is_set() or self.error:
                return False
            self.submitted += 1
            self.replaced += int(self.pending is not None)
            self.pending = clip
            self.condition.notify_all()
            return True

    def wait_for(self, sequence: int):
        with self.condition:
            self.condition.wait_for(
                lambda: self.completed_sequence >= sequence or self.done.is_set()
                or self.error is not None or self.stop.is_set())

    def acknowledge(self, sequence: int):
        with self.condition:
            self.delivered_sequence = max(self.delivered_sequence, sequence)
            self.condition.notify_all()

    def wait_for_delivery(self, sequence: int):
        with self.condition:
            self.condition.wait_for(
                lambda: self.delivered_sequence >= sequence or self.error is not None
                or self.stop.is_set())

    def finish(self, cancel=False):
        with self.condition:
            self.closing = True
            if cancel:
                self.pending = None
            self.condition.notify_all()

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.closing
                                            or self.stop.is_set(), timeout=0.2)
                    if self.stop.is_set():
                        self.pending = None
                        break
                    if self.pending is None:
                        if self.closing:
                            break
                        continue
                    clip, self.pending = self.pending, None
                started = time.monotonic()
                prediction = self.predict(clip.frames)
                finished = time.monotonic()
                self.results.put((clip, prediction, (finished - started) * 1000, finished))
                with self.condition:
                    self.completed += 1
                    self.completed_sequence = clip.sequence
                    self.condition.notify_all()
        except Exception as error:
            self.error = error
        finally:
            self.done.set()
            with self.condition:
                self.condition.notify_all()


class AlarmGate:
    """Confirm camera events; stale results never confirm or clear an alarm."""
    RECOVERY_LABELS = {"Standing", "Walking", "Sitting", "normal", "no_fall"}

    def __init__(self, config: LiveConfig):
        self.config = config
        self.fall_streak = 0
        self.recovery_streak = 0
        self.latched = False
        self.last_clip_end = None
        self.last_alarm_end = float("-inf")
        self.event_count = 0

    def update(self, clip: Clip, prediction: dict, stale: bool):
        if stale:
            self.fall_streak = self.recovery_streak = 0
            return None
        if (self.last_clip_end is not None and clip.end_s - self.last_clip_end
                > max(self.config.window_seconds, self.config.analysis_interval * 2.5)):
            self.fall_streak = self.recovery_streak = 0
        self.last_clip_end = clip.end_s
        binary = set(prediction.get("scores", {})) == {"no_fall", "fall"}
        falling = ((binary or prediction["label"] in {"FallDown", "fall"})
                   and prediction["fall_score"] >= self.config.fall_threshold)
        if falling:
            self.fall_streak += 1
            self.recovery_streak = 0
        else:
            self.fall_streak = 0
            if binary or prediction["label"] in self.RECOVERY_LABELS:
                self.recovery_streak += 1
                if self.recovery_streak >= self.config.recovery_windows:
                    self.latched = False
            else:
                self.recovery_streak = 0
        if (falling and not self.latched and self.fall_streak >= self.config.confirm_windows
                and clip.end_s - self.last_alarm_end >= self.config.cooldown_seconds):
            self.event_count += 1
            self.latched = True
            self.last_alarm_end = clip.end_s
            return {"alert_id": self.event_count,
                    "clip_start_ms": round(clip.start_s * 1000),
                    "clip_end_ms": round(clip.end_s * 1000),
                    "label": "fall" if binary else prediction["label"],
                    "confidence": prediction["fall_score"] if binary else prediction["confidence"],
                    "fall_score": prediction["fall_score"]}
        return None
