import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from live_core import Clip, LiveConfig
from person_core import PersonAnalyzer, PersonJob, TrackingConfig
from person_engine import PersonDisplay, TrackingWorker, crop_clip, draw_people


def job(identity, sequence):
    return PersonJob(identity, 0, Clip(sequence, 0, 2, sequence, time.monotonic(), (identity,)), (0, 0, 10, 10))


class SchedulingTests(unittest.TestCase):
    def test_replacements_preserve_other_persons_turn(self):
        entered, release, stop = threading.Event(), threading.Event(), threading.Event()
        calls = []
        def predict(frames):
            calls.append(frames[0])
            if len(calls) == 1:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("Test timed out")
            return {"label": "Standing"}
        worker = PersonAnalyzer(predict, stop, 3)
        try:
            worker.submit(job(1, 1))
            self.assertTrue(entered.wait(3))
            worker.submit(job(2, 1))
            worker.submit(job(1, 2))
            worker.submit(job(2, 2))
            worker.finish()
            release.set()
            self.assertTrue(worker.done.wait(3))
            self.assertEqual(calls, [1, 2, 1])
            self.assertEqual(worker.replaced, 1)
        finally:
            stop.set()
            release.set()
            worker.finish(cancel=True)
            worker.thread.join(timeout=3)

    def test_expired_work_is_not_sent_to_model(self):
        stop = threading.Event()
        worker = PersonAnalyzer(lambda frames: self.fail("Expired clip reached model"), stop, 1, max_age=0.1)
        old = job(1, 1)
        old = PersonJob(1, 0, Clip(1, 0, 2, 1, time.monotonic() - 10, (1,)), old.bbox)
        worker.submit(old)
        worker.finish()
        self.assertTrue(worker.done.wait(3))
        worker.thread.join(timeout=3)
        self.assertEqual(worker.expired, 1)
        self.assertIsNone(worker.results.get_nowait()[1])
        self.assertIsNone(worker.error)


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)
    def cpu(self):
        return self
    def numpy(self):
        return self.value


class FakeDetector:
    def __init__(self):
        self.count = 0
    def track(self, frame, **kwargs):
        self.count += 1
        missing = 32 <= self.count <= 50
        boxes = type("Boxes", (), {})()
        boxes.xyxy = Array(np.empty((0, 4)) if missing else [[10, 10, 40, 55]])
        boxes.conf = Array([] if missing else [0.95])
        boxes.id = None if missing else Array([1])
        result = type("Result", (), {"boxes": boxes})()
        return [result]


class PipelineTests(unittest.TestCase):
    def test_complete_pending_clip_survives_lost_detection(self):
        stop, entered, release = threading.Event(), threading.Event(), threading.Event()
        config = LiveConfig(window_seconds=0.8, analysis_interval=0.1)
        def predict(frames):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test timed out")
            return {"label": "Standing"}
        analyzer = PersonAnalyzer(predict, stop, 6)
        with tempfile.TemporaryDirectory() as directory:
            worker = TrackingWorker(FakeDetector(), "cpu", analyzer, PersonDisplay(False, config), stop, True,
                config, TrackingConfig(detection_fps=30, lost_roi_seconds=0), 16, Path(directory))
            worker.start()
            try:
                for index in range(50):
                    worker.offer((index + 1, index / 30, np.zeros((64, 64, 3), np.uint8), time.monotonic()))
                    with worker.condition:
                        self.assertTrue(worker.condition.wait_for(
                            lambda: worker.last_processed >= index + 1 or worker.error, timeout=3))
                    self.assertIsNone(worker.error)
                    if index == 25:
                        self.assertTrue(entered.wait(1))
                self.assertTrue(entered.wait(1))
                self.assertEqual(worker.reset_count, 1)
                self.assertIn((1, 0), analyzer.pending)
                worker.finish()
                release.set()
                self.assertTrue(worker.done.wait(3))
                self.assertTrue(analyzer.done.wait(3))
                self.assertEqual(analyzer.completed, 2)
            finally:
                stop.set()
                release.set()
                worker.finish(cancel=True)
                analyzer.finish(cancel=True)
                worker.thread.join(timeout=3)
                analyzer.thread.join(timeout=3)

    def test_missing_identity_starts_new_clip_segment(self):
        stop = threading.Event()
        config = LiveConfig(window_seconds=0.8, analysis_interval=0.2)
        tracking = TrackingConfig(detection_fps=30, lost_roi_seconds=0)
        analyzer = PersonAnalyzer(lambda frames: {"label": "Standing"}, stop, 6)
        with tempfile.TemporaryDirectory() as directory:
            display = PersonDisplay(False, config)
            worker = TrackingWorker(FakeDetector(), "cpu", analyzer, display, stop, True,
                                    config, tracking, 16, Path(directory))
            worker.start()
            try:
                for index in range(85):
                    value = 20 if index < 32 else 220
                    worker.offer((index + 1, index / 30, np.full((64, 64, 3), value, np.uint8), time.monotonic()))
                    with worker.condition:
                        self.assertTrue(worker.condition.wait_for(
                            lambda: worker.last_processed >= index + 1 or worker.error, timeout=3))
                    self.assertIsNone(worker.error)
                worker.finish()
                self.assertTrue(worker.done.wait(3))
                self.assertTrue(analyzer.done.wait(3))
                results = []
                while not analyzer.results.empty():
                    results.append(analyzer.results.get_nowait()[0])
                self.assertEqual({item.segment_id for item in results}, {0, 1})
                for item in results:
                    expected = 20 if item.segment_id == 0 else 220
                    self.assertTrue(all(np.asarray(frame)[112, 112, 0] == expected for frame in item.clip.frames))
                self.assertEqual(worker.reset_count, 1)
            finally:
                stop.set()
                worker.finish(cancel=True)
                analyzer.finish(cancel=True)
                worker.thread.join(timeout=3)
                analyzer.thread.join(timeout=3)

    def test_delayed_result_cannot_attach_to_new_segment(self):
        display = PersonDisplay(False, LiveConfig())
        display.observations([dict(track_id=1, segment_id=1)], time.monotonic(), 10, 1, 0, 0)
        old = job(1, 1)
        self.assertFalse(display.is_current(old))
        display.result(old, dict(label="FallDown", confidence=0.9), 10, True, False)
        self.assertNotIn("result", display.snapshot()["rows"][0])

    def test_historical_alarm_stays_visible_without_a_current_box(self):
        display = PersonDisplay(False, LiveConfig())
        old = job(1, 1)
        display.result(old, dict(label="FallDown", confidence=0.95), 10, False, True)
        display.observations([], time.monotonic(), 10, 0, 0, 0)
        state = display.snapshot()
        self.assertEqual(state["rows"], [])
        self.assertIn((1, 0), state["alarms"])
        rendered = draw_people(np.zeros((240, 320, 3), np.uint8), state, "Test")
        self.assertEqual(rendered[2, 2].tolist(), [0, 0, 255])

    def test_fixed_crop_keeps_motion_and_square_images(self):
        frame = np.zeros((100, 140, 3), dtype=np.uint8)
        frame[20:40, 10:30] = 255
        clip = Clip(1, 0, 2, 16, 0, ((frame, (10, 20, 30, 40)), (frame, (70, 50, 90, 70))))
        result = crop_clip(clip, 0.2)
        self.assertEqual([image.size for image in result.frames], [(224, 224), (224, 224)])
        self.assertTrue(np.array_equal(np.asarray(result.frames[0]), np.asarray(result.frames[1])))


if __name__ == "__main__":
    unittest.main()
