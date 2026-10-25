"""Optional tests of scheduling and alarm state; no camera/model is loaded.

Prepared for the user to run. These tests are not a model-accuracy benchmark.
"""
import threading
import unittest

from live_core import AlarmGate, Clip, ClipBuffer, LatestAnalyzer, LiveConfig


def clip(sequence, end):
    return Clip(sequence, end - 2, end, sequence, 0.0, tuple(range(16)))


def prediction(label="FallDown", score=0.9):
    return {"label": label, "confidence": score,
            "fall_score": score if label == "FallDown" else 0.01, "scores": {label: score}}


class LiveWindowTests(unittest.TestCase):
    def test_video_rate_is_separate_from_sixteen_sample_window(self):
        buffer = ClipBuffer(16, LiveConfig())
        jobs = []
        for index in range(121):
            item = buffer.offer(index / 30, index, index / 30, lambda value=index: value)
            if item:
                jobs.append(item)
        self.assertTrue(jobs)
        self.assertEqual(len(jobs[0].frames), 16)
        self.assertEqual(len(set(jobs[0].frames)), 16)
        self.assertAlmostEqual(jobs[0].end_s - jobs[0].start_s, 2.0)
        self.assertGreaterEqual(jobs[0].end_frame, 59)

    def test_gap_requires_a_new_window(self):
        buffer = ClipBuffer(16, LiveConfig())
        for index in range(61):
            buffer.offer(index / 30, index, 0, lambda: object())
        self.assertIsNone(buffer.offer(4.0, 120, 0, lambda: object()))
        self.assertEqual(len(buffer.samples), 1)
        self.assertEqual(buffer.resets, 1)

    def test_missing_sampling_rate_does_not_duplicate_frames(self):
        buffer = ClipBuffer(16, LiveConfig())
        jobs = [buffer.offer(index / 4, index, 0, lambda value=index: value)
                for index in range(20)]
        self.assertFalse(any(jobs))


class AlarmTests(unittest.TestCase):
    def test_stale_result_does_not_confirm_a_fall(self):
        gate = AlarmGate(LiveConfig())
        self.assertIsNone(gate.update(clip(1, 2), prediction(), False))
        self.assertIsNone(gate.update(clip(2, 3), prediction(), True))
        self.assertIsNone(gate.update(clip(3, 4), prediction(), False))
        self.assertIsNotNone(gate.update(clip(4, 5), prediction(), False))

    def test_alarm_is_latched_until_recovery_and_cooldown(self):
        gate = AlarmGate(LiveConfig())
        gate.update(clip(1, 2), prediction(), False)
        self.assertIsNotNone(gate.update(clip(2, 3), prediction(), False))
        for number in range(4, 8):
            self.assertIsNone(gate.update(clip(number, number), prediction(), False))
        gate.update(clip(8, 8), prediction("Standing"), False)
        gate.update(clip(9, 9), prediction("Walking"), False)
        gate.update(clip(10, 10), prediction(), False)
        self.assertIsNone(gate.update(clip(11, 11), prediction(), False))
        gate.update(clip(12, 12), prediction(), False)
        self.assertIsNotNone(gate.update(clip(13, 13), prediction(), False))


class QueueTests(unittest.TestCase):
    def test_slow_inference_replaces_only_pending_work(self):
        started, release, stop = threading.Event(), threading.Event(), threading.Event()
        calls = []
        def predict(frames):
            calls.append(frames[0])
            if len(calls) == 1:
                started.set()
                if not release.wait(2):
                    raise RuntimeError("Test synchronization timed out")
            return prediction()
        worker = LatestAnalyzer(predict, stop)
        try:
            worker.submit(Clip(1, 0, 2, 60, 0, (1,)))
            self.assertTrue(started.wait(2))
            worker.submit(Clip(2, 1, 3, 90, 0, (2,)))
            worker.submit(Clip(3, 2, 4, 120, 0, (3,)))
            worker.finish()
            release.set()
            self.assertTrue(worker.done.wait(2))
            self.assertEqual(calls, [1, 3])
            self.assertEqual(worker.replaced, 1)
            self.assertIsNone(worker.error)
        finally:
            stop.set()
            release.set()
            worker.finish(cancel=True)
            worker.thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
