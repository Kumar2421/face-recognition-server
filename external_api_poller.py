import os
import time
import json
import base64
import logging
import datetime
import httpx
import yaml
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("external_api_poller")

BENCHMARK_FILE = "poller_benchmark.json"

def save_benchmark(data: Dict[str, Any]):
    try:
        with open(BENCHMARK_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save benchmark: {e}")

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_sftp_link(link: str) -> Dict[str, str]:
    """
    Parses camera and timestamp from sftpLink:
    /BR=TMJ-CBE__ZN=FirstFloor__IO=ENTRY__CAM=FF-Stair-Entry/10.122.15.58_01_20260427123420875_FACE_SNAP_32640.jpg
    """
    out = {"camera": "unknown", "ts_str": ""}
    if not link:
        return out
    
    # Extract Camera
    import re
    cam_match = re.search(r"CAM=([^/]+)", link)
    if cam_match:
        out["camera"] = cam_match.group(1)
    
    # Extract Timestamp (YYYYMMDDHHMMSS)
    ts_match = re.search(r"_(\d{14})", link)
    if ts_match:
        out["ts_str"] = ts_match.group(1)
        
    return out

def ts_to_epoch(ts_str: str) -> float:
    """Converts YYYYMMDDHHMMSS (IST) to UTC Unix epoch."""
    if not ts_str or len(ts_str) < 14:
        return time.time()
    try:
        # The timestamps in filenames (e.g. 20260419235728) are in local IST time.
        # We must parse them as IST and convert to UTC epoch.
        dt = datetime.datetime.strptime(ts_str[:14], "%Y%m%d%H%M%S")
        ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        dt = dt.replace(tzinfo=ist)
        return dt.timestamp()
    except Exception:
        return time.time()

def wait_for_service(url: str, timeout: int = 60):
    """Wait for the face_service to be ready."""
    start_time = time.time()
    logger.info(f"Waiting for face_service at {url}...")
    while time.time() - start_time < timeout:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{url}/health", timeout=2.0)
                if resp.status_code == 200:
                    logger.info("face_service is ready!")
                    return True
        except Exception:
            pass
        time.sleep(2)
    logger.error(f"face_service at {url} not ready after {timeout}s")
    return False

STATE_FILE = "poller_state.json"

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    return {}

def save_state(state: Dict[str, Any]):
    try:
        logger.info(f"Saving state to {STATE_FILE}: {state}")
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        # Force write to disk
        os.sync() if hasattr(os, 'sync') else f.flush()
        logger.info("State saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

def process_single_event(event: Dict[str, Any], internal_api: str, client: httpx.Client) -> Dict[str, Any]:
    """Processes a single event and returns metrics if successful, None if failed."""
    ext_id = str(event.get("id"))
    filename = (
        event.get("filename") or 
        event.get("eventId") or
        ext_id
    )
    
    # metrics structure
    result = {
        "success": False,
        "skipped": False,
        "model_ms": 0,
        "processing_ms": 0
    }

    # 1. Check internal face_service for existence
    try:
        check_url = f"{internal_api}/v1/events/recognition?source_path={filename}&limit=1"
        check_resp = client.get(check_url)
        if check_resp.status_code == 200:
            existing_data = check_resp.json()
            existing = existing_data.get("items", [])
            if existing:
                result["skipped"] = True
                result["success"] = True
                item = existing[0]
                result["model_ms"] = item.get("model_ms", 0)
                result["processing_ms"] = item.get("processing_ms", 0)
                return result
    except Exception as e:
        logger.warning(f"Existence check failed for {filename}: {e}")

    # ... [Data extraction logic remains the same] ...
    # (Trimming for brevity in thought, but tool call will include full content)

    # 2. Extract Data
    face_resp = event.get("faceResponse") or {}
    if isinstance(face_resp, str):
        try: face_resp = json.loads(face_resp)
        except: face_resp = {}

    ev_data = event.get("data") or {}
    if isinstance(ev_data, str):
        try: ev_data = json.loads(ev_data)
        except: ev_data = {}

    sftp_link = (
        event.get("sftpLink") or 
        face_resp.get("sftpLink") or 
        ev_data.get("sftpLink")
    )
    
    img_base64 = (
        event.get("image") or 
        event.get("base64") or
        face_resp.get("image") or 
        face_resp.get("base64") or
        ev_data.get("image") or
        ev_data.get("base64")
    )
    
    img_url = None
    if not img_base64:
        box_data = face_resp.get("boxData") or {}
        if isinstance(box_data, dict):
            img_url = box_data.get("imageUrl")
            inner_fr = box_data.get("faceResponse") or {}
            if isinstance(inner_fr, dict):
                sftp_link = sftp_link or inner_fr.get("sftpLink")
                img_base64 = img_base64 or inner_fr.get("image") or inner_fr.get("base64")

    logger.debug(f"DEBUG: Data extracted for {filename}: sftp_link={'YES' if sftp_link else 'NO'}, img_base64={'YES' if img_base64 else 'NO'}, img_url={'YES' if img_url else 'NO'}")

    if not sftp_link and not img_base64 and not img_url:
        logger.warning(f"Skipping event {ext_id}: No sftpLink or image data found.")
        result["skipped"] = True
        result["success"] = True
        return result
        
    parsed = parse_sftp_link(sftp_link)
    camera = event.get("camera")
    if not camera or camera == "unknown":
        camera = parsed["camera"]
    
    ts_epoch = ts_to_epoch(parsed["ts_str"])
    if not parsed["ts_str"] and event.get("timestamp"):
        try:
            dt = datetime.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            ts_epoch = dt.timestamp()
        except Exception:
            pass
    
    # 3. Handle Image
    img_bytes = None
    if img_base64:
        try: 
            if "," in img_base64:
                img_base64 = img_base64.split(",")[1]
            img_bytes = base64.b64decode(img_base64)
            logger.debug(f"DEBUG: Successfully decoded base64 image for {filename} ({len(img_bytes)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to decode base64 for {filename}: {e}")
    
    if not img_bytes and img_url:
        try:
            logger.debug(f"DEBUG: Fetching image from URL: {img_url}")
            img_resp = client.get(img_url, timeout=10.0)
            if img_resp.status_code == 200:
                img_bytes = img_resp.content
                logger.debug(f"DEBUG: Successfully fetched image from URL for {filename} ({len(img_bytes)} bytes)")
            else:
                logger.warning(f"Failed to fetch image URL for {filename}: {img_resp.status_code}")
        except Exception as e:
            logger.warning(f"Failed to fetch image URL for {filename}: {e}")

    if not img_bytes:
        logger.warning(f"No image data found for {filename}, skipping.")
        result["skipped"] = True
        result["success"] = True
        return result

    # 4. Ingest
    try:
        files = {"file": (filename or "event.jpg", img_bytes, "image/jpeg")}
        data = {
            "camera": camera,
            "ts": str(ts_epoch),
            "source_path": filename,
            "process_all_faces": "1"
        }
        logger.info(f"Ingesting {filename}: size={len(img_bytes)} bytes, camera={camera}, ts={ts_epoch}")
        
        ingest_resp = client.post(f"{internal_api}/v1/events/recognition", data=data, files=files)
        if ingest_resp.status_code == 200:
            logger.info(f"Successfully ingested {filename}")
            resp_json = ingest_resp.json()
            # If the API returns multiple faces, we might get a list or a single object depending on the implementation
            # Assuming it returns the created event details
            result["success"] = True
            if isinstance(resp_json, list) and len(resp_json) > 0:
                # Average metrics if multiple faces detected? Or just take the first.
                result["model_ms"] = sum(f.get("model_ms", 0) for f in resp_json) / len(resp_json)
                result["processing_ms"] = sum(f.get("processing_ms", 0) for f in resp_json) / len(resp_json)
            elif isinstance(resp_json, dict):
                result["model_ms"] = resp_json.get("model_ms", 0)
                result["processing_ms"] = resp_json.get("processing_ms", 0)
            return result
        else:
            logger.error(f"Failed to ingest {filename}: {ingest_resp.status_code} {ingest_resp.text}")
            return result
    except Exception as e:
        logger.error(f"Ingest error {filename}: {e}")
        return result

def process_external_events(api_base: str, jwt: str, internal_api: str, date_str: str = None, workers: int = 5):
    if not date_str:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
    if not wait_for_service(internal_api):
        logger.error("Skipping sync because face_service is unavailable.")
        return

    state = load_state()
    last_processed_id = state.get(f"last_id_{date_str}")

    logger.info(f"Syncing events (ASC) for date: {date_str}, starting from last_id: {last_processed_id}, workers: {workers}")
    
    headers = {"Authorization": f"Bearer {jwt}"}
    page = 0
    limit = 50
    
    client = httpx.Client(timeout=60.0)
    
    new_last_id = None
    stop_polling = False
    
    # Benchmark stats
    benchmark_stats = {
        "date": date_str,
        "start_time": datetime.datetime.now().isoformat(),
        "total_api_events": 0,
        "processed_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "skipped_count": 0,
        "pages_processed": 0,
        "last_id_seen": last_processed_id,
        "avg_model_ms": 0.0,
        "avg_processing_ms": 0.0,
        "total_model_ms": 0.0,
        "total_processing_ms": 0.0
    }

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while not stop_polling:
                # Switched to sort=asc for historical backfill
                url = f"{api_base}/api/v2/retail/event/all?date={date_str}&page={page}&limit={limit}&all=true&sort=asc"
                try:
                    resp = client.get(url, headers=headers)
                    if resp.status_code != 200:
                        logger.error(f"Failed to fetch external events: {resp.status_code} {resp.text}")
                        break
                        
                    body = resp.json()
                    events = body.get("list", [])
                    total_count = body.get("total", 0)
                    benchmark_stats["total_api_events"] = total_count
                    
                    if not events:
                        logger.info(f"No more events found on page {page}. Total available: {total_count}")
                        break
                        
                    logger.info(f"Processing page {page} ({len(events)} events). Date: {date_str}, Total API Events: {total_count}")

                    current_batch = []
                    for event in events:
                        ext_id = str(event.get("id"))
                        
                        # In ASC mode, new_last_id should be updated as we go or at the end of successful pages
                        # For simplicity, we track the latest ID in the current run
                        new_last_id = ext_id
                        
                        if last_processed_id and int(ext_id) <= int(last_processed_id):
                            # In ASC mode, if we see an ID <= last_processed, we've seen it before
                            benchmark_stats["skipped_count"] += 1
                            continue
                        
                        current_batch.append(event)

                    if not current_batch:
                        # If entire page was already processed, just move to next page
                        page += 1
                        continue

                    # Process batch in parallel
                    futures = [executor.submit(process_single_event, ev, internal_api, client) for ev in current_batch]
                    for future in futures:
                        benchmark_stats["processed_count"] += 1
                        try:
                            res = future.result()
                            if res and res["success"]:
                                benchmark_stats["success_count"] += 1
                                benchmark_stats["total_model_ms"] += res["model_ms"]
                                benchmark_stats["total_processing_ms"] += res["processing_ms"]
                                
                                # Update averages
                                total_metrics_count = benchmark_stats["success_count"]
                                benchmark_stats["avg_model_ms"] = benchmark_stats["total_model_ms"] / total_metrics_count
                                benchmark_stats["avg_processing_ms"] = benchmark_stats["total_processing_ms"] / total_metrics_count
                            else:
                                benchmark_stats["failure_count"] += 1
                        except Exception as e:
                            logger.error(f"Worker thread error: {e}")
                            benchmark_stats["failure_count"] += 1

                    benchmark_stats["pages_processed"] += 1
                    benchmark_stats["end_time"] = datetime.datetime.now().isoformat()
                    benchmark_stats["last_id_seen"] = new_last_id
                    save_benchmark(benchmark_stats)

                    if len(events) < limit:
                        logger.info(f"Reached end of data at page {page} for {date_str}.")
                        break
                    
                    page += 1
                    
                    # Update state periodically or after each page in ASC mode
                    state[f"last_id_{date_str}"] = new_last_id
                    save_state(state)
                        
                except Exception as e:
                    logger.error(f"Pagination loop error on page {page}: {e}")
                    break
    finally:
        benchmark_stats["end_time"] = datetime.datetime.now().isoformat()
        save_benchmark(benchmark_stats)
        
        if new_last_id:
            state[f"last_id_{date_str}"] = new_last_id
            save_state(state)
            logger.info(f"Final state update with last_id: {new_last_id} for date: {date_str}")
        
        client.close()

def main():
    config_path = _env("CONFIG_PATH", "/app/config.yaml")
    config = load_config(config_path)
    
    ext_cfg = config.get("external_api", {})
    if not ext_cfg.get("enabled"):
        logger.info("External API sync is disabled.")
        return
        
    api_base = ext_cfg.get("base_url", "https://live.thefusionapps.com")
    jwt = ext_cfg.get("jwt_token", "")
    poll_interval = int(ext_cfg.get("poll_interval_sec", 300))
    workers = int(ext_cfg.get("workers", 5))
    internal_api = _env("API_BASE_URL", "http://localhost:8000").rstrip("/")
    
    if not jwt:
        logger.error("JWT token not found in config.yaml under external_api.jwt_token")
        return

    while True:
        try:
            sync_date = _env("SYNC_DATE") or ext_cfg.get("sync_date")
            process_external_events(api_base, jwt, internal_api, date_str=sync_date, workers=workers)
        except Exception as e:
            logger.error(f"Global sync error: {e}")
            
        logger.info(f"Sleeping for {poll_interval}s...")
        time.sleep(poll_interval)

if __name__ == "__main__":
    main()
