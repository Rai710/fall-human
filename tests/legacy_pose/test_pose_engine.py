"""Data integrity checks for per-person motion histories."""
import unittest

import numpy as np

from pose_engine import LANDMARK_NAMES, Person, TrackRegistry, landmark_rows


def person(identity, confidence=0.9):
    points = np.tile([100.0, 120.0, confidence], (17, 1))
    return Person(identity, (20, 30, 200, 220), 0.95, points)


class TrackHistoryTests(unittest.TestCase):
    def test_reordered_people_keep_separate_histories(self):
        registry = TrackRegistry()
        registry.update([person(1), person(2)], 0, 320, 240)
        records = registry.update([person(2), person(1)], 100, 320, 240)
        self.assertEqual([record.track_id for record in records], [1, 2])
        self.assertEqual([record.segment_id for record in records], [0, 0])
        self.assertEqual(registry.totals[1]["observed_frames"], 2)

    def test_missing_pose_exports_blanks_and_recovery_starts_new_segment(self):
        registry = TrackRegistry()
        registry.update([person(1)], 0, 320, 240)
        missing = registry.update([], 100, 320, 240)[0]
        rows = list(landmark_rows(1, 100, missing, 320, 240))
        self.assertEqual(len(rows), 17)
        self.assertEqual([row["landmark_name"] for row in rows], LANDMARK_NAMES)
        self.assertTrue(all(row["observed"] == 0 and row["x"] == row["confidence"] == "" for row in rows))
        self.assertEqual(registry.update([person(1)], 200, 320, 240)[0].segment_id, 1)

    def test_long_absence_expires_but_does_not_reuse_sequence_segment(self):
        registry = TrackRegistry()
        registry.update([person(1)], 0, 320, 240)
        self.assertEqual(registry.update([], 2000, 320, 240), [])
        self.assertEqual(registry.update([person(1)], 2100, 320, 240)[0].segment_id, 1)

    def test_low_confidence_is_not_usable_and_recovery_splits_segment(self):
        registry = TrackRegistry()
        low = registry.update([person(1, 0)], 0, 320, 240)[0]
        self.assertFalse(low.pose_usable)
        rows = list(landmark_rows(0, 0, low, 320, 240))
        self.assertTrue(all(row["keypoint_valid"] == 0 and row["x"] == "" for row in rows))
        self.assertEqual(registry.update([person(1)], 100, 320, 240)[0].segment_id, 1)

    def test_unassigned_detection_is_not_given_a_fake_identity(self):
        registry = TrackRegistry()
        self.assertEqual(registry.update([person(None)], 0, 320, 240), [])
        self.assertEqual(registry.totals, {})

    def test_normalized_coordinates_keep_the_person_identity(self):
        record = TrackRegistry().update([person(7)], 0, 320, 240)[0]
        row = list(landmark_rows(0, 0, record, 320, 240))[11]
        self.assertEqual(row["track_id"], 7)
        self.assertEqual(row["landmark_name"], "left_hip")
        self.assertAlmostEqual(row["x"], 100 / 320)
        self.assertAlmostEqual(row["y"], 0.5)


if __name__ == "__main__":
    unittest.main()
