import json

import pytest

import detector_config as dc


def write_cfg(tmp_path, **override):
    path = tmp_path / "model_config.json"
    path.write_text(json.dumps(override), encoding="utf-8")
    return path


def test_default_is_pretrained_pose_class_zero(tmp_path):
    cfg = dc.load_config(tmp_path / "tidak-ada.json", env={})
    assert cfg["source"] == "pretrained"
    assert cfg["pretrained"]["name"] == "yolo11n-pose.pt"
    assert cfg["person_class_ids"] == [0]


def test_env_overrides_file(tmp_path):
    path = write_cfg(tmp_path, source="pretrained")
    cfg = dc.load_config(path, env={"DETECTOR_SOURCE": "LOCAL", "DETECTOR_LOCAL_PATH": "x/best.pt",
                                    "DETECTOR_CLASSES": "1,2"})
    assert cfg["source"] == "local"
    assert cfg["local"]["path"] == "x/best.pt"
    assert cfg["person_class_ids"] == [1, 2]


def test_classes_can_be_all(tmp_path):
    assert dc.load_config(write_cfg(tmp_path, person_class_ids=None), env={})["person_class_ids"] is None
    assert dc.load_config(write_cfg(tmp_path), env={"DETECTOR_CLASSES": "all"})["person_class_ids"] is None


def test_invalid_source_and_json(tmp_path):
    with pytest.raises(ValueError):
        dc.load_config(write_cfg(tmp_path, source="roboflow"), env={})
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError):
        dc.load_config(bad, env={})


def test_local_resolves_existing_file(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    cfg = dc.load_config(write_cfg(tmp_path, source="local", local={"path": str(weights)}), env={})
    resolved = dc.resolve_detector(cfg)
    assert resolved.path == weights and resolved.source == "local"
    assert dc.describe_detector(cfg)["ready"] is True


def test_local_missing_gives_actionable_error(tmp_path):
    cfg = dc.load_config(write_cfg(tmp_path, source="local", local={"path": str(tmp_path / "nope.pt")}), env={})
    with pytest.raises(FileNotFoundError, match="best.pt lokal"):
        dc.resolve_detector(cfg)


def test_huggingface_requires_repo_id(tmp_path):
    cfg = dc.load_config(write_cfg(tmp_path, source="huggingface"), env={})
    with pytest.raises(ValueError, match="repo_id"):
        dc.resolve_detector(cfg)


def test_huggingface_downloads_with_token(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setenv("HF_TOKEN", "hf_rahasia")
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        target = tmp_path / "models" / "hf" / "user__repo" / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w")
        return str(target)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    cfg = dc.load_config(write_cfg(tmp_path, source="huggingface",
                                   huggingface={"repo_id": "user/repo", "filename": "best.pt"}), env={})
    resolved = dc.resolve_detector(cfg)
    assert resolved.path.name == "best.pt"
    assert calls["repo_id"] == "user/repo" and calls["token"] == "hf_rahasia"


def test_huggingface_falls_back_to_previous_copy_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "MODELS_DIR", tmp_path / "models")
    copy = tmp_path / "models" / "hf" / "user__repo" / "best.pt"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"w")
    import huggingface_hub

    def offline(**_):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    cfg = dc.load_config(write_cfg(tmp_path, source="huggingface",
                                   huggingface={"repo_id": "user/repo"}), env={})
    assert dc.resolve_detector(cfg).path == copy


def test_pretrained_without_auto_download_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "MODELS_DIR", tmp_path / "models")
    cfg = dc.load_config(write_cfg(tmp_path, pretrained={"name": "yolo11n-pose.pt", "auto_download": False}), env={})
    with pytest.raises(FileNotFoundError, match="download_detector.py"):
        dc.resolve_detector(cfg)


def test_pretrained_auto_download_places_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "MODELS_DIR", tmp_path / "models")

    def fake_download(name, target, on_status):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w")

    monkeypatch.setattr(dc, "_download_pretrained", fake_download)
    resolved = dc.resolve_detector(dc.load_config(tmp_path / "x.json", env={}))
    assert resolved.path == tmp_path / "models" / "yolo11n-pose.pt"
