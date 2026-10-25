"""Bounded, fair per-person inference scheduling; no ML dependencies."""
from collections import OrderedDict
from dataclasses import dataclass
import math
import queue
import threading
import time

from live_core import Clip


@dataclass(frozen=True)
class TrackingConfig:
    detection_fps: float = 10.0
    image_size: int = 416
    max_people: int = 6
    crop_margin: float = 0.2
    lost_roi_seconds: float = 2.0

    def __post_init__(self):
        if not math.isfinite(self.detection_fps) or not 1 <= self.detection_fps <= 30:
            raise ValueError("FPS deteksi harus antara 1 dan 30.")
        if not isinstance(self.max_people, int) or not 1 <= self.max_people <= 16:
            raise ValueError("Batas orang yang dianalisis harus antara 1 dan 16.")
        if self.image_size < 160 or self.image_size % 32:
            raise ValueError("Ukuran YOLO harus kelipatan 32 dan minimal 160.")
        if not math.isfinite(self.crop_margin) or not 0 <= self.crop_margin <= 1:
            raise ValueError("Margin crop harus antara 0 dan 1.")

        if not math.isfinite(self.lost_roi_seconds) or not 0 <= self.lost_roi_seconds <= 5:
            raise ValueError("Durasi ROI terakhir harus antara 0 dan 5 detik.")


@dataclass(frozen=True)
class PersonJob:
    track_id: int
    segment_id: int
    clip: Clip
    bbox: tuple
    evidence_scope: str = "person_crop"

    @property
    def key(self):
        return self.track_id, self.segment_id

    @property
    def token(self):
        return self.key + (self.clip.sequence,)


class PersonAnalyzer:
    """One active job; one replaceable pending job per identity segment.

    Replacing a pending clip preserves its place in line. Expired jobs are
    logged as skipped instead of spending model time on already stale clips.
    """
    def __init__(self, predict, stop, max_pending, max_age=None):
        self.predict, self.stop = predict, stop
        self.max_pending, self.max_age = max_pending, max_age
        self.condition = threading.Condition()
        self.pending = OrderedDict()
        self.results = queue.Queue()
        self.done = threading.Event()
        self.closing = False
        self.error = None
        self.submitted = self.replaced = self.completed = self.expired = self.evicted = 0
        self.delivered = set()
        self.thread = threading.Thread(target=self._run, name="person-videomae", daemon=True)
        self.thread.start()

    def submit(self, job):
        with self.condition:
            if self.closing or self.stop.is_set() or self.error:
                return False
            self.submitted += 1
            if job.key in self.pending:
                self.replaced += 1
            elif len(self.pending) >= self.max_pending:
                self.pending.popitem(last=False)
                self.evicted += 1
            self.pending[job.key] = job
            self.condition.notify_all()
            return True

    def discard(self, key):
        with self.condition:
            if self.pending.pop(key, None) is not None:
                self.evicted += 1

    def acknowledge(self, job):
        with self.condition:
            self.delivered.add(job.token)
            self.condition.notify_all()

    def wait_delivered(self, job):
        with self.condition:
            self.condition.wait_for(lambda: job.token in self.delivered
                                    or self.error is not None or self.stop.is_set())
            self.delivered.discard(job.token)

    def finish(self, cancel=False):
        with self.condition:
            self.closing = True
            if cancel:
                self.evicted += len(self.pending)
                self.pending.clear()
            self.condition.notify_all()

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.closing
                                            or self.stop.is_set(), timeout=0.2)
                    if self.stop.is_set():
                        break
                    if not self.pending:
                        if self.closing:
                            break
                        continue
                    _, job = self.pending.popitem(last=False)
                started = time.monotonic()
                if self.max_age is not None and started - job.clip.end_wall > self.max_age:
                    self.expired += 1
                    self.results.put((job, None, 0.0, started, started))
                    continue
                prediction = self.predict(job.clip.frames)
                finished = time.monotonic()
                self.completed += 1
                self.results.put((job, prediction, (finished - started) * 1000, started, finished))
        except Exception as error:
            self.error = error
        finally:
            self.done.set()
            with self.condition:
                self.condition.notify_all()
