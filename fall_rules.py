"""Conservative, per-track collapse rule for a fixed corridor/waiting-room camera.

This is a suspicion signal for staff review, not a medical diagnosis. All time
thresholds use source timestamps; playback or CPU speed does not change the rule.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from types import SimpleNamespace


@dataclass(frozen=True)
class Observation:
    timestamp_ms: int
    shoulder_y: float | None
    nose_y: float | None
    torso_angle_deg: float | None
    bbox_height: float
    bbox_bottom: float
    bbox_aspect: float

    @property
    def upright(self):
        return self.torso_angle_deg is not None and self.torso_angle_deg >= 48 and self.bbox_aspect <= 1.1

    @property
    def horizontal(self):
        return self.bbox_aspect >= 1.2 and (
            self.torso_angle_deg is None or self.torso_angle_deg <= 45 or self.bbox_aspect >= 1.7
        )


@dataclass
class TrackState:
    history: deque = field(default_factory=deque)
    mode: str = "normal"
    last_seen_ms: int = -1
    candidate_start_ms: int = 0
    last_evidence_ms: int = 0
    evidence_ms: int = 0
    evidence_frames: int = 0
    baseline_bottom: float = 0
    baseline_height: float = 0
    drop_ratio: float = 0
    descent_ms: int = 0
    recovery_start_ms: int | None = None
    last_bbox: tuple | None = None


def _point(person, index, width, height):
    x, y, confidence = (float(value) for value in person.points[index])
    if (math.isfinite(x) and math.isfinite(y) and math.isfinite(confidence)
            and confidence >= 0.4 and 0 <= x < width and 0 <= y < height):
        return x, y
    return None


def observation(record, timestamp_ms, width, height):
    """Use anatomy only when pose quality passed; bounding boxes remain usable later."""
    person = record.person
    if person is None:
        return None
    x1, y1, x2, y2 = (float(value) for value in person.bbox)
    box_height = max(1, y2 - y1)
    box_width = max(1, x2 - x1)
    shoulder_y = nose_y = torso_angle = None
    if record.pose_usable:
        shoulders = [_point(person, index, width, height) for index in (5, 6)]
        hips = [_point(person, index, width, height) for index in (11, 12)]
        shoulders = [point for point in shoulders if point is not None]
        hips = [point for point in hips if point is not None]
        if shoulders:
            shoulder_y = sum(point[1] for point in shoulders) / len(shoulders)
        nose = _point(person, 0, width, height)
        if nose is not None:
            nose_y = nose[1]
        if shoulders and hips:
            sx = sum(point[0] for point in shoulders) / len(shoulders)
            sy = sum(point[1] for point in shoulders) / len(shoulders)
            hx = sum(point[0] for point in hips) / len(hips)
            hy = sum(point[1] for point in hips) / len(hips)
            if math.hypot(hx - sx, hy - sy) >= 0.08 * box_height:
                torso_angle = math.degrees(math.atan2(abs(hy - sy), abs(hx - sx)))
    return Observation(timestamp_ms, shoulder_y, nose_y, torso_angle,
                       box_height, y2, box_width / box_height)


def bbox_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0


class CollapseDetector:
    """Fast upright-to-horizontal descent, floor proximity, then sustained evidence.

    It deliberately abstains if the camera cannot establish an upright pose and
    a fast descent. Missing pose after landing may be bridged by a tracked box.
    """

    LOOKBACK_MS = 1500
    MIN_DESCENT_MS = 250
    MIN_DROP_BODY_RATIO = 0.27
    MIN_DROP_RATE_PER_SECOND = 0.32
    MAX_FLOOR_RISE_BODY_RATIO = 0.10
    CONFIRM_MS = 1500
    MIN_EVIDENCE_MS = 800
    MIN_EVIDENCE_FRAMES = 8
    MAX_EVIDENCE_GAP_MS = 600
    RECOVERY_MS = 1500
    EXPIRE_MS = 5000
    MAX_PROVISIONAL_GAP_MS = 600
    MIN_PROVISIONAL_IOU = 0.25

    def __init__(self):
        self.tracks = {}
        self.provisional_matches = {}

    @classmethod
    def settings(cls):
        return {name.lower(): getattr(cls, name) for name in (
            "LOOKBACK_MS", "MIN_DESCENT_MS", "MIN_DROP_BODY_RATIO",
            "MIN_DROP_RATE_PER_SECOND", "MAX_FLOOR_RISE_BODY_RATIO",
            "CONFIRM_MS", "MIN_EVIDENCE_MS", "MIN_EVIDENCE_FRAMES",
            "MAX_EVIDENCE_GAP_MS", "RECOVERY_MS", "EXPIRE_MS",
            "MAX_PROVISIONAL_GAP_MS", "MIN_PROVISIONAL_IOU")}

    def _associate_unassigned(self, records, unassigned_people, timestamp_ms):
        """Bridge a candidate with a nearby box after ByteTrack drops its ID.

        This is local rule evidence only. It never writes an invented ID to the
        pose CSV or changes ByteTrack's identity assignments.
        """
        by_id = {record.track_id: record for record in records}
        observed_ids = {identity for identity, record in by_id.items() if record.person is not None}
        pair_scores = []
        for identity, state in self.tracks.items():
            if (identity in observed_ids or state.mode not in ("candidate", "alert")
                    or state.last_bbox is None
                    or timestamp_ms - state.last_seen_ms > self.MAX_PROVISIONAL_GAP_MS):
                continue
            for person in unassigned_people:
                x1, y1, x2, y2 = person.bbox
                if (x2 - x1) / max(1, y2 - y1) < 1.2:
                    continue
                overlap = bbox_iou(state.last_bbox, person.bbox)
                if overlap >= self.MIN_PROVISIONAL_IOU:
                    pair_scores.append((overlap, identity, person))
        used_ids = set()
        used_people = set()
        for _overlap, identity, person in sorted(pair_scores, key=lambda pair: pair[0], reverse=True):
            if identity in used_ids or id(person) in used_people:
                continue
            used_ids.add(identity)
            used_people.add(id(person))
            self.provisional_matches[id(person)] = identity
            by_id[identity] = SimpleNamespace(track_id=identity, person=person, pose_usable=False)
        return [by_id[identity] for identity in sorted(by_id)]

    def _motion(self, state, current):
        if not current.horizontal:
            return None
        best = None
        for previous in state.history:
            elapsed = current.timestamp_ms - previous.timestamp_ms
            if not self.MIN_DESCENT_MS <= elapsed <= self.LOOKBACK_MS or previous.horizontal:
                continue
            floor_rise = previous.bbox_bottom - current.bbox_bottom
            if floor_rise > self.MAX_FLOOR_RISE_BODY_RATIO * previous.bbox_height:
                continue
            ratios = []
            if previous.shoulder_y is not None and current.shoulder_y is not None:
                ratios.append((current.shoulder_y - previous.shoulder_y) / previous.bbox_height)
            if previous.nose_y is not None and current.nose_y is not None:
                ratios.append((current.nose_y - previous.nose_y) / previous.bbox_height)
            if not ratios:
                continue
            drop = max(ratios)
            if drop < self.MIN_DROP_BODY_RATIO or drop * 1000 / elapsed < self.MIN_DROP_RATE_PER_SECOND:
                continue
            if best is None or drop > best[0]:
                best = (drop, elapsed, previous)
        return best

    def update(self, records, timestamp_ms, width, height, frame_index, unassigned_people=()):
        statuses = {}
        alerts = []
        self.provisional_matches = {}
        effective_records = self._associate_unassigned(records, unassigned_people, timestamp_ms)
        for record in effective_records:
            identity = record.track_id
            state = self.tracks.setdefault(identity, TrackState())
            current = observation(record, timestamp_ms, width, height)
            if current is None:
                if state.mode == "candidate" and timestamp_ms - state.last_evidence_ms > self.MAX_EVIDENCE_GAP_MS:
                    state.mode = "normal"
                statuses[identity] = state.mode
                continue
            state.last_seen_ms = timestamp_ms
            state.last_bbox = tuple(float(value) for value in record.person.bbox)
            if state.mode == "alert":
                if record.pose_usable and current.upright:
                    if state.recovery_start_ms is None:
                        state.recovery_start_ms = timestamp_ms
                    elif timestamp_ms - state.recovery_start_ms >= self.RECOVERY_MS:
                        state.mode = "normal"
                        state.history.clear()
                        state.recovery_start_ms = None
                else:
                    state.recovery_start_ms = None
            elif state.mode == "candidate":
                near_floor = (state.baseline_bottom - current.bbox_bottom
                              <= self.MAX_FLOOR_RISE_BODY_RATIO * state.baseline_height)
                if current.horizontal and near_floor:
                    gap = timestamp_ms - state.last_evidence_ms
                    if gap <= self.MAX_EVIDENCE_GAP_MS:
                        state.evidence_ms += min(gap, 150)
                        state.evidence_frames += 1
                        state.last_evidence_ms = timestamp_ms
                    else:
                        state.mode = "normal"
                elif timestamp_ms - state.last_evidence_ms > self.MAX_EVIDENCE_GAP_MS:
                    state.mode = "normal"
                if (state.mode == "candidate" and timestamp_ms - state.candidate_start_ms >= self.CONFIRM_MS
                        and state.evidence_ms >= self.MIN_EVIDENCE_MS
                        and state.evidence_frames >= self.MIN_EVIDENCE_FRAMES
                        and current.horizontal and near_floor):
                    state.mode = "alert"
                    alerts.append({
                        "frame_index": frame_index, "timestamp_ms": timestamp_ms,
                        "track_id": identity, "event": "suspected_collapse",
                        "evidence_source": ("provisional_bbox" if record.person.track_id is None
                                            else "tracked_pose_or_bbox"),
                        "drop_body_ratio": round(state.drop_ratio, 3),
                        "descent_ms": state.descent_ms,
                        "horizontal_hold_ms": timestamp_ms - state.candidate_start_ms,
                        "evidence_ms": state.evidence_ms,
                        "bbox_aspect": round(current.bbox_aspect, 3),
                        "floor_rise_body_ratio": round(
                            (state.baseline_bottom - current.bbox_bottom) / state.baseline_height, 3),
                    })
            if state.mode == "normal" and record.pose_usable:
                motion = self._motion(state, current)
                if motion is not None:
                    drop, elapsed, previous = motion
                    state.mode = "candidate"
                    state.candidate_start_ms = timestamp_ms
                    state.last_evidence_ms = timestamp_ms
                    state.evidence_ms = 0
                    state.evidence_frames = 1
                    state.baseline_bottom = previous.bbox_bottom
                    state.baseline_height = previous.bbox_height
                    state.drop_ratio = drop
                    state.descent_ms = elapsed
            if record.pose_usable and (current.shoulder_y is not None or current.nose_y is not None):
                state.history.append(current)
            while state.history and timestamp_ms - state.history[0].timestamp_ms > self.LOOKBACK_MS:
                state.history.popleft()
            statuses[identity] = state.mode
        for identity, state in list(self.tracks.items()):
            if timestamp_ms - state.last_seen_ms > self.EXPIRE_MS:
                del self.tracks[identity]
        return statuses, alerts