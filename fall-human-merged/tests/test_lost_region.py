import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
from live_core import LiveConfig
from person_core import TrackingConfig
from person_engine import TrackingWorker, PersonDisplay, last_known_region, draw_people
import person_engine


class Array:
    def __init__(self, data): self.data = np.asarray(data)
    def cpu(self): return self
    def numpy(self): return self.data


class Detector:
    names = {0: 'person'}

    def __init__(self, missing):
        self.index = 0
        self.missing = missing
        self.predictor = SimpleNamespace(trackers=[])
    def track(self, frame, **kwargs):
        missing = self.index in self.missing
        self.index += 1
        boxes = type('Boxes', (), {})()
        boxes.xyxy = Array(np.empty((0, 4)) if missing else [[20, 10, 30, 45]])
        boxes.conf = Array([] if missing else [.95])
        boxes.id = None if missing else Array([1])
        return [type('Result', (), {'boxes': boxes})()]


class Collector:
    error = None
    def __init__(self): self.jobs = []
    def submit(self, job): self.jobs.append(job); return True
    def finish(self, **kwargs): pass


class LostRegionTests(unittest.TestCase):
    def run_sequence(self, missing, count=85, retention=2):
        config = LiveConfig(window_seconds=.8, analysis_interval=.2)
        display, collector, stop = PersonDisplay(False, config), Collector(), threading.Event()
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            worker = TrackingWorker(Detector(missing), 'cpu', collector, display, stop, True,
                config, TrackingConfig(detection_fps=30, lost_roi_seconds=retention), 16, Path(directory))
            worker.start()
            try:
                for i in range(count):
                    worker.offer((i + 1, i / 30, np.full((64, 64, 3), i, np.uint8), time.monotonic()))
                    with worker.condition:
                        self.assertTrue(worker.condition.wait_for(
                            lambda: worker.last_processed >= i + 1 or worker.error, timeout=3))
                    self.assertIsNone(worker.error)
                    rows.append(display.snapshot())
                worker.finish()
                self.assertTrue(worker.done.wait(3))
                logs = list(csv.DictReader((Path(directory) / 'tracking.csv').open()))
            finally:
                stop.set(); worker.finish(cancel=True); worker.thread.join(timeout=3)
        return worker, collector.jobs, rows, logs

    def test_fresh_samples_complete_clip_through_detection_dropout(self):
        worker, jobs, states, logs = self.run_sequence(set(range(15, 35)))
        region_jobs = [job for job in jobs if job.evidence_scope == 'last_known_region']
        self.assertTrue(region_jobs)
        crossing = next(job for job in region_jobs if job.clip.start_s < .5 < job.clip.end_s)
        values = [int(np.asarray(frame)[112, 112, 0]) for frame in crossing.clip.frames]
        self.assertEqual(len(set(values)), 16, 'No frozen frame or duplicate padding')
        self.assertGreater(max(values), 15, 'Contains actual fresh frames after detector loss')
        self.assertEqual(worker.reset_count, 0)
        self.assertEqual({job.segment_id for job in jobs}, {0})
        lost_state = states[20]
        self.assertEqual(lost_state['detections'], 0)
        self.assertEqual(lost_state['rows'][0]['observation_status'], 'lost_roi')
        self.assertGreater(lost_state['rows'][0]['last_seen_age_ms'], 0)
        display = PersonDisplay(False, LiveConfig())
        display.observations(lost_state['rows'], time.monotonic(), 1, 0, 0, 0)
        self.assertFalse(display.is_current(crossing))
        render = draw_people(np.zeros((240, 320, 3), np.uint8), lost_state, 'Test')
        self.assertGreater(np.count_nonzero(render), 0)
        self.assertTrue(any(row['observation_status'] == 'lost_roi' for row in logs))

    def test_region_expires_without_renewing_real_last_seen_time(self):
        worker, jobs, states, logs = self.run_sequence(set(range(15, 60)), retention=.8)
        self.assertEqual(states[45]['rows'], [])
        self.assertEqual(states[60]['rows'][0]['segment_id'], 1)
        self.assertEqual(worker.reset_count, 1)
        self.assertTrue(all(float(row['last_seen_age_ms']) <= 800.001
                            for row in logs if row['observation_status'] == 'lost_roi'))
        resumed = [job for job in jobs if job.segment_id == 1]
        self.assertTrue(resumed)
        self.assertTrue(all(np.asarray(frame)[112, 112, 0] >= 60
                            for job in resumed for frame in job.clip.frames))

    def test_region_clipped_and_large_enough_for_horizontal_body(self):
        region = last_known_region((20, 10, 30, 45), 64, 64)
        self.assertLess(region[0], 20)
        self.assertGreater(region[2], 30)
        self.assertGreater(region[3], 45)
        self.assertTrue(all(0 <= value <= 64 for value in region))
        for invalid in (-1, 6, float('nan')):
            with self.assertRaises(ValueError): TrackingConfig(lost_roi_seconds=invalid)

    def test_full_pipeline_logs_region_alarm_and_does_not_clear_on_bystander_normal(self):
        class Predictor:
            device, source, revision, num_frames = 'cpu', 'fake-local', 'test', 16
            count = 0
            def predict(self, frames):
                self.count += 1
                label = 'FallDown' if self.count == 1 else 'Standing'
                return dict(label=label, confidence=.99, fall_score=.99 if self.count == 1 else .01,
                            scores={label:.99})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            video = base/'fixture.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64,64))
            self.assertTrue(writer.isOpened())
            for i in range(48): writer.write(np.full((64,64,3), i, np.uint8))
            writer.release()
            detector = Detector(set(range(15,100)))
            fake_ultralytics = SimpleNamespace(YOLO=lambda *a, **kw: detector)
            events = []
            with patch.dict('sys.modules', {'ultralytics':fake_ultralytics}), \
                 patch.object(person_engine, 'load_model', return_value=Predictor()):
                output, summary = person_engine.process_people(video, save_video=False, realtime=False,
                    config=LiveConfig(window_seconds=.8, analysis_interval=.2, confirm_windows=1),
                    tracking=TrackingConfig(detection_fps=30), output_root=base/'outputs',
                    on_event=lambda kind, value: events.append((kind,value)))
            self.assertEqual(summary['status'], 'completed')
            alerts = list(csv.DictReader((output/'alerts.csv').open()))
            predictions = list(csv.DictReader((output/'predictions.csv').open()))
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]['evidence_scope'], 'last_known_region')
            self.assertGreater(len(predictions), 2)
            self.assertTrue(all(row['evidence_scope'] == 'last_known_region' for row in predictions))
            self.assertEqual(predictions[-1]['label'], 'Standing')
            # Normal in an ambiguous ROI cannot prove the fallen patient recovered.
            preview = cv2.imread(str(output/'preview.jpg'))
            self.assertGreater(int(preview[2,2,2]), 150)


if __name__ == '__main__': unittest.main()
