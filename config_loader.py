from __future__ import annotations

import os
from typing import Any


def _deep_get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def load_config(path: str | None) -> dict[str, Any]:
    path = str(path or "").strip()
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("pyyaml is required to load config.yaml") from e

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        data = None

    return data if isinstance(data, dict) else {}


def apply_env_defaults_from_config(cfg: dict[str, Any]) -> None:
    """Apply config values to environment variables only if the env var is not already set.

    This keeps current behavior (env wins) while allowing a config.yaml to drive defaults.
    """

    def set_if_missing(key: str, value: Any) -> None:
        if value is None:
            return
        if os.environ.get(key) is not None:
            return
        os.environ[key] = str(value)

    # Recognition thresholds
    set_if_missing("FACE_SERVICE_MIN_SIMILARITY", _deep_get(cfg, ["recognition", "min_similarity"]))
    set_if_missing("FACE_SERVICE_TOP2_MARGIN", _deep_get(cfg, ["recognition", "min_top2_margin"]))
    set_if_missing("FACE_SERVICE_TOP2_HIGH_CONF", _deep_get(cfg, ["recognition", "top2_high_conf_sim"]))

    # Quality thresholds
    set_if_missing("FACE_QUALITY_BLUR_MIN", _deep_get(cfg, ["quality", "min_blur"]))
    set_if_missing("FACE_QUALITY_FACE_RATIO_MIN", _deep_get(cfg, ["quality", "min_face_ratio"]))
    set_if_missing("FACE_QUALITY_BRIGHTNESS_MIN", _deep_get(cfg, ["quality", "min_brightness"]))
    set_if_missing("FACE_QUALITY_BRIGHTNESS_MAX", _deep_get(cfg, ["quality", "max_brightness"]))
    set_if_missing("FACE_QUALITY_LANDMARK_MIN", _deep_get(cfg, ["quality", "min_landmark_conf"]))
    set_if_missing("FACE_QUALITY_MAX_ABS_YAW", _deep_get(cfg, ["quality", "max_abs_yaw"]))
    set_if_missing("FACE_QUALITY_MAX_ABS_PITCH", _deep_get(cfg, ["quality", "max_abs_pitch"]))
    set_if_missing("FACE_MIN_RESOLUTION", _deep_get(cfg, ["quality", "min_resolution"]))

    # SFTP defaults (used by poller)
    set_if_missing("SFTP_HOST", _deep_get(cfg, ["sftp", "host"]))
    set_if_missing("SFTP_PORT", _deep_get(cfg, ["sftp", "port"]))
    set_if_missing("SFTP_USERNAME", _deep_get(cfg, ["sftp", "username"]))
    set_if_missing("SFTP_PASSWORD", _deep_get(cfg, ["sftp", "password"]))
    set_if_missing("SFTP_BASE_PATH", _deep_get(cfg, ["sftp", "base_path"]))
    set_if_missing("SFTP_POLL_INTERVAL_SEC", _deep_get(cfg, ["sftp", "poll_interval_sec"]))
    set_if_missing("SFTP_PROCESSED_DIRNAME", _deep_get(cfg, ["sftp", "processed_dirname"]))

    # API base used by poller
    set_if_missing("API_BASE_URL", _deep_get(cfg, ["api", "base_url"]))
