"""Verify binary semantics and the API-to-engine contract for both analysis scopes."""
from pathlib import Path
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps
from fastapi.testclient import TestClient

import backend.main as main
import backend.runner as runner_module
import person_engine
import pose_engine
from action_config import action_model_path, describe_action
from live_core import AlarmGate, Clip, LiveConfig


def test_config_exposes_actual_binary_model_and_detector():
    response = TestClient(main.app).get('/api/config')
    assert response.status_code == 200
    data = response.json()
    assert data['detector']['label'] == 'YoloDetector.pt'
    assert data['detector']['ready'] is True
    assert data['action_model']['ready'] is True
    assert data['action_model']['labels'] == {'0': 'no_fall', '1': 'fall'}
    assert data['action_model']['threshold'] == .70
    assert data['action_model']['export_threshold'] == .58


def test_binary_threshold_does_not_require_argmax_fall_and_recovers():
    gate = AlarmGate(LiveConfig(fall_threshold=.4, confirm_windows=1, cooldown_seconds=0))
    def offer(index, score):
        return gate.update(Clip(index, index - 2, index, index, 0, ()),
                           dict(label='fall' if score >= .5 else 'no_fall',
                                confidence=max(score, 1-score), fall_score=score,
                                scores={'no_fall': 1-score, 'fall': score}), False)
    event = offer(2, .45)
    assert event and event['label'] == 'fall'
    assert event['confidence'] == .45
    offer(3, .1)
    offer(4, .1)
    assert not gate.latched
    assert offer(5, .9)


def test_preprocessing_matches_training_rgb_letterbox():
    rgb = np.random.default_rng(42).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    expected = ImageOps.pad(Image.fromarray(rgb), (224, 224),
                            method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    actual = pose_engine._mixed_image(rgb[:, :, ::-1].copy())
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize('scope', ['scene', 'person'])
def test_runner_routes_model_settings_and_preserves_alarm_identity(scope, tmp_path, monkeypatch):
    received = []
    def engine(source, **options):
        received.append(options)
        options['on_event']('session', tmp_path)
        options['on_event']('alarm', {'alert_id': 1, 'track_id': 3 if scope == 'person' else '',
                                     'label': 'fall', 'confidence': .9, 'fall_score': .9,
                                     'clip_end_ms': 2000})
        options['on_frame'](np.zeros((32, 32, 3), np.uint8),
                            {'active_alarms': 1, 'analyses': 1, 'capture_fps': 15})
        return tmp_path, {'status': 'completed', 'model_labels': {'0': 'no_fall', '1': 'fall'}}
    monkeypatch.setattr(person_engine, 'process_people', engine if scope == 'person' else
                        lambda *a, **k: pytest.fail('wrong engine'))
    monkeypatch.setattr(pose_engine, 'process_source', engine if scope == 'scene' else
                        lambda *a, **k: pytest.fail('wrong engine'))
    runner = runner_module.SessionRunner(Path('upload.mp4'), save_video=False, analysis_scope=scope)
    runner._run()
    assert runner.state == 'completed', runner.error
    options = received[0]
    assert options['local_only'] is True
    assert options['model_path'] == action_model_path()
    assert options['config'].fall_threshold == .70
    assert options['config'].analysis_interval == .5
    assert options['config'].confirm_windows == 1
    assert options['realtime'] is False
    assert runner.snapshot()['analysis_scope'] == scope
    assert runner.stats['is_fall'] is True
    assert runner.alerts[0]['track_id'] == ('3' if scope == 'person' else '')
    assert runner.alerts[0]['evidence_scope'] == ('person_crop' if scope == 'person' else 'whole_scene')


def test_api_defaults_to_scene_and_validates_scope(tmp_path, monkeypatch):
    calls = []
    def start(source, **options):
        calls.append(options)
        return SimpleNamespace(snapshot=lambda: {'id': 'test', 'state': 'starting'})
    monkeypatch.setattr(main, 'UPLOADS_DIR', tmp_path)
    monkeypatch.setattr(main, 'manager', SimpleNamespace(start=start))
    client = TestClient(main.app)
    response = client.post('/api/sessions/start', files={'file': ('sample.mp4', b'video', 'video/mp4')})
    assert response.status_code == 200
    assert calls[0]['analysis_scope'] == 'scene'
    assert calls[0]['realtime'] is False
    response = client.post('/api/sessions/start', data={'analysis_scope': 'unknown'},
                           files={'file': ('sample.mp4', b'video', 'video/mp4')})
    assert response.status_code == 422


def test_unfinished_stop_does_not_admit_another_inference(monkeypatch):
    manager = runner_module.SessionManager()
    runner = runner_module.SessionRunner(Path('upload.mp4'))
    runner.state = 'stopping'
    runner.thread = SimpleNamespace(is_alive=lambda: True)
    manager.sessions[runner.id] = runner
    manager.active_id = runner.id
    with pytest.raises(RuntimeError, match='Masih ada sesi'):
        manager.start(Path('another.mp4'))
