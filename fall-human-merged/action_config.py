"""The web dashboard uses the complete local two-class export, without downloads."""
from pathlib import Path

from mixed_model_config import MIXED_LABEL, mixed_model_path, read_mixed_settings

ROOT = Path(__file__).resolve().parent
APPLICATION_FALL_THRESHOLD = 0.70


def action_model_path():
    return mixed_model_path(ROOT)


def action_settings():
    settings = read_mixed_settings(action_model_path())
    settings["export_threshold"] = settings["threshold"]
    settings["threshold"] = APPLICATION_FALL_THRESHOLD
    return settings


def describe_action():
    result = dict(label=MIXED_LABEL, ready=False, labels={"0": "no_fall", "1": "fall"},
                  threshold=None, export_threshold=None, clip_seconds=None, stride_seconds=None)
    try:
        settings = action_settings()
        result.update(threshold=settings["threshold"], export_threshold=settings["export_threshold"], clip_seconds=settings["clip_seconds"],
                      stride_seconds=settings["stride_seconds"])
        result["ready"] = all((action_model_path() / name).is_file() for name in
                              ("model.safetensors", "config.json", "preprocessor_config.json",
                               "inference_config.json", "export_complete.json"))
    except (OSError, ValueError, KeyError):
        pass
    return result
