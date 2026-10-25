"""Run an uploaded local video through the real API/models, without a camera or downloads."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from backend.main import app
from action_config import action_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--scope', choices=['scene', 'person', 'both'], default='both')
    parser.add_argument('--report', type=Path, default=Path('outputs/integration_real_smoke.json'))
    args = parser.parse_args()
    scopes = ['scene', 'person'] if args.scope == 'both' else [args.scope]
    records = []
    with TestClient(app) as client:
        assert client.get('/api/health').status_code == 200
        assert client.get('/').status_code == 200
        assert '<div id="root">' in client.get('/').text
        for scope in scopes:
            print(f'START scope={scope} video={args.video.name}', flush=True)
            with args.video.open('rb') as handle:
                response = client.post('/api/sessions/start',
                                       data={'analysis_scope': scope, 'realtime': 'false', 'save_video': 'true'},
                                       files={'file': (args.video.name, handle, 'video/mp4')})
            assert response.status_code == 200, response.text
            session_id = response.json()['id']
            start = last_update = time.monotonic()
            while True:
                snapshot = client.get(f'/api/sessions/{session_id}').json()
                if snapshot['state'] not in {'starting', 'running', 'stopping'}:
                    break
                if time.monotonic() - last_update >= 5:
                    print(scope, snapshot['state'], snapshot.get('stats', {}).get('analyses', 0),
                          snapshot.get('message', ''), flush=True)
                    last_update = time.monotonic()
                if time.monotonic() - start > 600:
                    client.post(f'/api/sessions/{session_id}/stop')
                    raise TimeoutError('Real inference did not complete within ten minutes')
                time.sleep(.5)
            assert snapshot['state'] == 'completed', snapshot
            summary = snapshot['summary']
            assert summary['model_kind'] == 'finetuned2', summary
            assert summary['model_labels'] == {'0': 'no_fall', '1': 'fall'}, summary
            assert summary['config']['fall_threshold'] == action_settings()['threshold']
            assert summary['analyses'] > 0, summary
            assert summary['recorded_frames'] == summary['frames_processed']
            if scope == 'person':
                assert Path(summary['detector_model']).name == 'YoloDetector.pt'
                assert summary['detector_classes'] == [0]
            folder = snapshot['output']
            detail = client.get(f'/api/sessions/{folder}').json()
            assert detail['live'] is False
            assert detail['predictions']
            assert client.get(f'/api/sessions/{folder}/files/annotated.mp4').status_code == 200
            records.append(dict(scope=scope, video=args.video.name, output=folder,
                                elapsed_seconds=round(time.monotonic()-start, 2), summary=summary,
                                alerts=snapshot['alerts']))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
            print('RESULT', json.dumps({key: summary.get(key) for key in
                  ['status', 'model_kind', 'analyses', 'alerts_count', 'peak_people', 'analysis_mean_ms',
                   'frames_processed', 'recorded_frames', 'warnings']}), flush=True)
    print('REPORT', args.report.resolve(), flush=True)


if __name__ == '__main__':
    main()
