"""Behavior checks for the per-person, time-based collapse rule."""
import unittest

import numpy as np

from fall_rules import CollapseDetector
from pose_engine import Person, TrackFrame


WIDTH, HEIGHT = 1000, 600


def record(identity=1, posture="upright", quality=True):
    points = np.tile([450.0, 400.0, 0.99 if quality else 0.0], (17, 1))
    if posture == "upright":
        bbox = (400, 80, 500, 550)
        points[0, :2] = (450, 130)
        points[5, :2] = (420, 200)
        points[6, :2] = (480, 200)
        points[11, :2] = (430, 350)
        points[12, :2] = (470, 350)
    elif posture == "falling":
        bbox = (300, 370, 630, 550)
        points[0, :2] = (545, 415)
        points[5, :2] = (420, 420)
        points[6, :2] = (470, 420)
        points[11, :2] = (560, 430)
        points[12, :2] = (590, 430)
    elif posture == "floor":
        bbox = (280, 405, 660, 550)
        points[0, :2] = (560, 450)
        points[5, :2] = (420, 455)
        points[6, :2] = (470, 455)
        points[11, :2] = (570, 460)
        points[12, :2] = (600, 460)
    elif posture == "bed":
        bbox = (280, 200, 660, 350)
        points[0, :2] = (560, 240)
        points[5, :2] = (420, 250)
        points[6, :2] = (470, 250)
        points[11, :2] = (570, 260)
        points[12, :2] = (600, 260)
    else:
        raise ValueError(posture)
    return TrackFrame(identity, 0, Person(identity, bbox, 0.95, points), quality)


class CollapseRulesTests(unittest.TestCase):
    def update(self, detector, moment, records):
        return detector.update(records, moment, WIDTH, HEIGHT, moment // 100)

    def test_fast_descent_to_floor_then_hold_alerts_once(self):
        detector = CollapseDetector()
        for moment in (0, 100, 200, 300):
            self.update(detector, moment, [record()])
        status, alerts = self.update(detector, 800, [record(posture="falling")])
        self.assertEqual(status[1], "candidate")
        self.assertEqual(alerts, [])
        all_alerts = []
        for moment in range(900, 3200, 100):
            _, alerts = self.update(detector, moment, [record(posture="floor")])
            all_alerts.extend(alerts)
        self.assertEqual(len(all_alerts), 1)
        self.assertEqual(all_alerts[0]["track_id"], 1)
        self.assertGreaterEqual(all_alerts[0]["horizontal_hold_ms"], 1500)

    def test_bed_landing_is_rejected(self):
        detector = CollapseDetector()
        for moment in (0, 100, 200, 300):
            self.update(detector, moment, [record()])
        for moment in range(800, 3300, 100):
            status, alerts = self.update(detector, moment, [record(posture="bed")])
            self.assertEqual(status[1], "normal")
            self.assertFalse(alerts)

    def test_short_lying_or_missing_evidence_does_not_alert(self):
        detector = CollapseDetector()
        self.update(detector, 0, [record()])
        self.update(detector, 600, [record(posture="falling")])
        for moment in (700, 800, 900, 1000):
            self.update(detector, moment, [record(posture="floor")])
        for moment in range(1100, 2300, 100):
            status, alerts = self.update(detector, moment, [TrackFrame(1, 0, None, False)])
            self.assertFalse(alerts)
        self.assertEqual(status[1], "normal")

    def test_pose_can_be_missing_after_landing_when_box_remains(self):
        detector = CollapseDetector()
        self.update(detector, 0, [record()])
        self.update(detector, 650, [record(posture="falling")])
        alerts = []
        for moment in range(750, 3000, 100):
            _, new = self.update(detector, moment, [record(posture="floor", quality=False)])
            alerts.extend(new)
        self.assertEqual(len(alerts), 1)

    def test_unassigned_box_can_confirm_existing_fall_candidate(self):
        detector = CollapseDetector()
        self.update(detector, 0, [record()])
        self.update(detector, 650, [record(posture="falling")])
        alerts = []
        for moment in range(750, 3000, 100):
            floor_person = record(posture="floor", quality=False).person
            floor_person.track_id = None
            statuses, new = detector.update(
                [TrackFrame(1, 0, None, False)], moment, WIDTH, HEIGHT,
                moment // 100, unassigned_people=[floor_person])
            self.assertEqual(detector.provisional_matches[id(floor_person)], 1)
            alerts.extend(new)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["evidence_source"], "provisional_bbox")
        self.assertEqual(statuses[1], "alert")

    def test_distant_unassigned_box_is_not_borrowed(self):
        detector = CollapseDetector()
        self.update(detector, 0, [record()])
        self.update(detector, 650, [record(posture="falling")])
        for moment in range(750, 3000, 100):
            other = record(posture="floor", quality=False).person
            other.track_id = None
            other.bbox = (0, 405, 200, 550)
            _, alerts = detector.update([TrackFrame(1, 0, None, False)], moment,
                                        WIDTH, HEIGHT, moment // 100, unassigned_people=[other])
            self.assertFalse(alerts)
            self.assertEqual(detector.provisional_matches, {})

    def test_one_person_falls_other_stays_upright(self):
        detector = CollapseDetector()
        self.update(detector, 0, [record(1), record(2)])
        self.update(detector, 650, [record(1, "falling"), record(2)])
        alerts = []
        for moment in range(750, 3000, 100):
            _, new = self.update(detector, moment, [record(2), record(1, "floor")])
            alerts.extend(new)
        self.assertEqual([alert["track_id"] for alert in alerts], [1])

    def test_already_horizontal_has_no_descent_history(self):
        detector = CollapseDetector()
        for moment in range(0, 3000, 100):
            _, alerts = self.update(detector, moment, [record(posture="floor")])
            self.assertFalse(alerts)


if __name__ == "__main__":
    unittest.main()
