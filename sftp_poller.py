from __future__ import annotations

import io
import os
import posixpath
import time
from typing import Any

from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

import httpx
import paramiko

from config_loader import apply_env_defaults_from_config, load_config


def _env(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    return default if v is None else str(v)


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _parse_human_dt(s: str, *, tz_name: str) -> float | None:
    raw = str(s or "").strip()
    if not raw:
        return None

    tz = None
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo(str(tz_name or "UTC"))
        except Exception:
            tz = None

    # Supported inputs (examples):
    # - 07-02-2026:07 pm
    # - 07-02-2026:07:30 pm
    # - 07-02-2026 07 pm
    # - 07-02-2026 07:30 pm
    s2 = " ".join(raw.replace("/", "-").split())
    fmts = (
        "%d-%m-%Y:%I %p",
        "%d-%m-%Y:%I:%M %p",
        "%d-%m-%Y %I %p",
        "%d-%m-%Y %I:%M %p",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(s2, fmt)
            if tz is not None:
                dt = dt.replace(tzinfo=tz)
                return float(dt.timestamp())
            # If timezone support is unavailable, treat as local time.
            return float(dt.timestamp())
        except Exception:
            continue
    return None


def _is_image_name(name: str) -> bool:
    n = str(name or "").lower()
    return n.endswith(".jpg") or n.endswith(".jpeg") or n.endswith(".png") or n.endswith(".webp")


def _split_csv(v: str) -> list[str]:
    out: list[str] = []
    for p in str(v or "").split(","):
        p = p.strip()
        if p:
            out.append(p)
    return out


def _norm_name(s: str) -> str:
    return str(s or "").strip()


def _should_skip_name(fn: str, *, include_any: list[str], exclude_any: list[str]) -> tuple[bool, str]:
    s = str(fn or "")
    if include_any:
        ok = any(sub in s for sub in include_any)
        if not ok:
            return True, "not_in_include"
    if exclude_any:
        hit = next((sub for sub in exclude_any if sub in s), None)
        if hit:
            return True, f"excluded:{hit}"
    return False, ""


def _ensure_remote_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = [p for p in str(path).split("/") if p]
    cur = "/"
    for p in parts:
        cur = posixpath.join(cur, p)
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except Exception:
                pass


def _list_dirs(sftp: paramiko.SFTPClient, base_path: str) -> list[str]:
    out: list[str] = []
    for it in sftp.listdir_attr(base_path):
        try:
            name = it.filename
            mode = int(it.st_mode)
        except Exception:
            continue
        # paramiko SFTPAttributes doesn't expose isdir directly without stat; use S_ISDIR
        try:
            import stat

            if stat.S_ISDIR(mode):
                out.append(name)
        except Exception:
            continue
    return sorted(out)


def main() -> None:
    cfg_path = _env("CONFIG_PATH", "/app/config.yaml")
    cfg = load_config(cfg_path)
    apply_env_defaults_from_config(cfg)

    host = _env("SFTP_HOST")
    port = _as_int(_env("SFTP_PORT", "22"), 22)
    username = _env("SFTP_USERNAME")
    password = _env("SFTP_PASSWORD")
    base_path = _env("SFTP_BASE_PATH", "/")
    poll_interval = _as_int(_env("SFTP_POLL_INTERVAL_SEC", "60"), 60)
    processed_dirname = str(_env("SFTP_PROCESSED_DIRNAME", "") or "").strip()
    max_files_per_cam = _as_int(_env("SFTP_MAX_FILES_PER_CAMERA_PER_TICK", "50"), 50)
    if max_files_per_cam < 1:
        max_files_per_cam = 1

    # Optional time filter for remote files (use server-reported st_mtime)
    # Examples:
    # - SFTP_SINCE_HOURS=24  (only last 24 hours)
    # - SFTP_SINCE_DAYS=7    (only last 7 days)
    # - SFTP_SINCE_TS=1700000000 (unix seconds)
    # - SFTP_UNTIL_TS=1700003600 (unix seconds)
    now_ts = time.time()
    tz_name = _env("SFTP_TIMEZONE", "Asia/Kolkata")
    since_dt = _env("SFTP_SINCE_DATETIME", "")
    until_dt = _env("SFTP_UNTIL_DATETIME", "")
    since_dt_ts = _parse_human_dt(since_dt, tz_name=tz_name) if since_dt else None
    until_dt_ts = _parse_human_dt(until_dt, tz_name=tz_name) if until_dt else None

    since_ts_env = _as_float(_env("SFTP_SINCE_TS", ""), 0.0)
    until_ts_env = _as_float(_env("SFTP_UNTIL_TS", ""), 0.0)
    since_hours = _as_float(_env("SFTP_SINCE_HOURS", ""), 0.0)
    since_days = _as_float(_env("SFTP_SINCE_DAYS", ""), 0.0)
    allow_no_mtime = _as_bool(_env("SFTP_TIMEFILTER_ALLOW_NO_MTIME", "1"), True)

    # Precedence: human datetime window > explicit ts > hours/days
    since_ts: float | None = None
    until_ts: float | None = None

    if since_dt and since_dt_ts is None:
        print(f"[sftp_poller] WARN invalid SFTP_SINCE_DATETIME='{since_dt}' (expected e.g. 07-02-2026:07 pm)")
    if until_dt and until_dt_ts is None:
        print(f"[sftp_poller] WARN invalid SFTP_UNTIL_DATETIME='{until_dt}' (expected e.g. 07-02-2026:10 pm)")

    if since_dt_ts is not None or until_dt_ts is not None:
        since_ts = float(since_dt_ts) if since_dt_ts is not None else None
        until_ts = float(until_dt_ts) if until_dt_ts is not None else None
    else:
        if since_ts_env > 0:
            since_ts = float(since_ts_env)
        elif since_hours > 0:
            since_ts = float(now_ts - (since_hours * 3600.0))
        elif since_days > 0:
            since_ts = float(now_ts - (since_days * 86400.0))

        until_ts = float(until_ts_env) if until_ts_env > 0 else None

    # Optional filters (comma-separated)
    # Example: SFTP_CAMERAS_ALLOWLIST=BEWELL-CHN-Entrance,BEWELL-CHN-Entry
    cameras_allowlist = set(_norm_name(x) for x in _split_csv(_env("SFTP_CAMERAS_ALLOWLIST", "")) if _norm_name(x))
    filename_include_any = _split_csv(_env("SFTP_FILENAME_INCLUDE_ANY", ""))
    # Default: skip obvious background frames if present in your naming convention
    filename_exclude_any = _split_csv(_env("SFTP_FILENAME_EXCLUDE_ANY", "FACE_BACKGROUND"))

    api_base = _env("API_BASE_URL", "http://localhost:8000").rstrip("/")
    top_k = _as_int(_env("SFTP_TOP_K", _env("FACE_SERVICE_TOP_K", "5")), 5)
    min_similarity = _as_float(_env("SFTP_MIN_SIMILARITY", _env("FACE_SERVICE_MIN_SIMILARITY", "0.25")), 0.25)

    if not host or not username:
        raise RuntimeError("SFTP_HOST and SFTP_USERNAME are required")

    print(
        f"[sftp_poller] connecting host={host} port={port} base_path={base_path} api_base={api_base} "
        f"allowlist={sorted(cameras_allowlist) if cameras_allowlist else 'ALL'} "
        f"include_any={filename_include_any or 'NONE'} exclude_any={filename_exclude_any or 'NONE'} "
        f"max_files_per_cam={max_files_per_cam} "
        f"since_ts={since_ts if since_ts is not None else 'NONE'} until_ts={until_ts if until_ts is not None else 'NONE'} tz={tz_name}"
    )

    transport = paramiko.Transport((host, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    client = httpx.Client(timeout=60.0)

    try:
        while True:
            try:
                cameras_raw = _list_dirs(sftp, base_path)
            except Exception as e:
                print(f"[sftp_poller] list base_path failed: {e}")
                cameras_raw = []

            cameras = [_norm_name(c) for c in cameras_raw if _norm_name(c)]
            if cameras_raw:
                print(f"[sftp_poller] base_path list raw={cameras_raw}")
            else:
                print(f"[sftp_poller] base_path list raw=EMPTY (base_path={base_path})")

            if cameras_allowlist:
                before = list(cameras)
                cameras = [c for c in cameras if c in cameras_allowlist]
                missing = sorted(cameras_allowlist.difference(set(before)))
                if missing:
                    print(f"[sftp_poller] allowlist folders missing under {base_path}: {missing}")
                    for m in missing:
                        p = posixpath.join(base_path, m)
                        try:
                            sftp.stat(p)
                            print(f"[sftp_poller] stat ok for missing allowlist folder path={p} (listed? no)")
                        except Exception as e:
                            print(f"[sftp_poller] stat failed for missing allowlist folder path={p}: {e}")

            print(f"[sftp_poller] scan tick cameras={cameras}")

            for cam in cameras:
                cam_dir = posixpath.join(base_path, cam)
                try:
                    items = sftp.listdir_attr(cam_dir)
                except Exception:
                    continue

                print(f"[sftp_poller] scanning cam={cam} dir={cam_dir} n_items={len(items)}")

                processed_this_cam = 0

                for it in items:
                    fn = getattr(it, "filename", "")
                    if not fn or (processed_dirname and fn == processed_dirname):
                        continue
                    if not _is_image_name(fn):
                        continue

                    mtime = None
                    try:
                        mtime = float(getattr(it, "st_mtime", None))
                    except Exception:
                        mtime = None
                    if mtime is None and not allow_no_mtime and (since_ts is not None or until_ts is not None):
                        print(f"[sftp_poller] skip cam={cam} file={fn} reason=no_mtime")
                        continue
                    if mtime is not None and since_ts is not None and float(mtime) < float(since_ts):
                        print(f"[sftp_poller] skip cam={cam} file={fn} reason=too_old mtime={int(mtime)}")
                        continue
                    if mtime is not None and until_ts is not None and float(mtime) > float(until_ts):
                        print(f"[sftp_poller] skip cam={cam} file={fn} reason=too_new mtime={int(mtime)}")
                        continue

                    skip, why = _should_skip_name(fn, include_any=filename_include_any, exclude_any=filename_exclude_any)
                    if skip:
                        print(f"[sftp_poller] skip cam={cam} file={fn} reason={why}")
                        continue

                    remote_path = posixpath.join(cam_dir, fn)
                    # Skip already-processed subfolders
                    if processed_dirname and f"/{processed_dirname}/" in remote_path:
                        continue

                    print(f"[sftp_poller] pickup cam={cam} remote_path={remote_path}")

                    try:
                        with sftp.open(remote_path, "rb") as f:
                            data = f.read()
                    except Exception as e:
                        print(f"[sftp_poller] download failed {remote_path}: {e}")
                        continue

                    try:
                        files = {"file": (fn, io.BytesIO(data), "image/jpeg")}
                        resp = client.post(
                            f"{api_base}/v1/events/recognition",
                            data={
                                "camera": cam,
                                "source_path": fn,
                                "top_k": str(top_k),
                                "min_similarity": str(min_similarity),
                                "process_all_faces": "1",
                            },
                            files=files,
                        )
                        ok = resp.status_code < 300
                        if not ok:
                            print(f"[sftp_poller] api failed {remote_path}: {resp.status_code} {resp.text[:200]}")
                        else:
                            j = resp.json()
                            decision = str(j.get("decision") or "")
                            faces_processed = None
                            try:
                                meta = j.get("meta") or {}
                                faces_processed = meta.get("faces_processed")
                            except Exception:
                                faces_processed = None
                            suffix = f" faces_processed={faces_processed}" if faces_processed is not None else ""
                            print(f"[sftp_poller] processed {remote_path} decision={decision}{suffix}")
                    except Exception as e:
                        print(f"[sftp_poller] api error {remote_path}: {e}")
                        ok = False

                    # Move to processed/ regardless of match/no_match/rejected if API call succeeded
                    if ok and processed_dirname:
                        processed_dir = posixpath.join(cam_dir, processed_dirname)
                        _ensure_remote_dir(sftp, processed_dir)
                        dst = posixpath.join(processed_dir, fn)
                        try:
                            sftp.rename(remote_path, dst)
                        except Exception:
                            # fallback: avoid overwrite
                            dst2 = posixpath.join(processed_dir, f"{int(time.time())}_{fn}")
                            try:
                                sftp.rename(remote_path, dst2)
                            except Exception as e:
                                print(f"[sftp_poller] move failed {remote_path} -> {dst}: {e}")

                    if ok:
                        processed_this_cam += 1
                        if processed_this_cam >= max_files_per_cam:
                            print(f"[sftp_poller] cam cap reached cam={cam} cap={max_files_per_cam} (remaining will be processed next tick)")
                            break

            time.sleep(max(5, int(poll_interval)))
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            sftp.close()
        except Exception:
            pass
        try:
            transport.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
