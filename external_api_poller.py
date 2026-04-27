import os
import time
import json
import base64
import logging
import datetime
import httpx
import yaml
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("external_api_poller")

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

def process_external_events(api_base: str, jwt: str, internal_api: str, date_str: str = None):
    if not date_str:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
    # Ensure service is ready before starting sync
    if not wait_for_service(internal_api):
        logger.error("Skipping sync because face_service is unavailable.")
        return

    logger.info(f"Syncing events for date: {date_str}")
    
    headers = {"Authorization": f"Bearer {jwt}"}
    page = 0
    limit = 50
    
    client = httpx.Client(timeout=60.0)
    
    # Track the latest event processed to avoid re-syncing redundant data
    # In a production environment, this should be stored in a persistent state file or database
    
    while True:
        url = f"{api_base}/api/v2/retail/event/all?date={date_str}&page={page}&limit={limit}&all=true&sort=desc"
        try:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.error(f"Failed to fetch external events: {resp.status_code} {resp.text}")
                break
                
            body = resp.json()
            events = body.get("list", [])
            
            if not events:
                logger.info(f"No more events found on page {page}. Total data for {date_str} has been fetched.")
                break
                
            logger.info(f"Processing page {page} with {len(events)} events...")

            for event in events:
                # 1. Unique ID check
                ext_id = str(event.get("id"))
                
                # Check if this specific event_id already exists in our system to avoid redundant work
                # We check by source_path/filename which contains the external event ID
                filename = (
                    event.get("filename") or 
                    event.get("eventId") or
                    ext_id
                )
                
                # Check internal face_service for existence
                try:
                    # We can use the source_path filter on recognition events
                    check_url = f"{internal_api}/v1/events/recognition?source_path={filename}&limit=1"
                    check_resp = client.get(check_url)
                    if check_resp.status_code == 200:
                        existing_data = check_resp.json()
                        existing = existing_data.get("items", [])
                        if existing:
                            logger.info(f"Event {filename} already exists, skipping.")
                            continue
                        else:
                            logger.info(f"Event {filename} NOT found in face_service, proceeding with ingestion.")
                except Exception as e:
                    logger.warning(f"Existence check failed for {filename}: {e}")

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
                    face_resp.get("image") or 
                    ev_data.get("image")
                )
                
                img_url = None
                if not img_base64:
                    box_data = face_resp.get("boxData") or {}
                    if isinstance(box_data, dict):
                        img_url = box_data.get("imageUrl")
                        inner_fr = box_data.get("faceResponse") or {}
                        if isinstance(inner_fr, dict):
                            sftp_link = sftp_link or inner_fr.get("sftpLink")
                            img_base64 = img_base64 or inner_fr.get("image")

                if not sftp_link:
                    continue
                    
                parsed = parse_sftp_link(sftp_link)
                camera = parsed["camera"]
                ts_epoch = ts_to_epoch(parsed["ts_str"])
                
                # 3. Handle Image
                img_bytes = None
                if img_base64:
                    try: 
                        if "," in img_base64:
                            img_base64 = img_base64.split(",")[1]
                        img_bytes = base64.b64decode(img_base64)
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 for {filename}: {e}")
                
                if not img_bytes and img_url:
                    try:
                        img_resp = client.get(img_url, timeout=10.0)
                        if img_resp.status_code == 200:
                            img_bytes = img_resp.content
                    except Exception as e:
                        logger.warning(f"Failed to fetch image URL for {filename}: {e}")

                if not img_bytes:
                    logger.warning(f"No image data found for {filename}, skipping.")
                    continue

                # 4. Ingest
                try:
                    files = {"file": (filename or "event.jpg", img_bytes, "image/jpeg")}
                    data = {
                        "camera": camera,
                        "ts": str(ts_epoch),
                        "source_path": filename,
                        "process_all_faces": "1"
                    }
                    # DEBUG: verify payload
                    logger.info(f"Ingesting {filename}: size={len(img_bytes)} bytes, camera={camera}, ts={ts_epoch}")
                    
                    ingest_resp = client.post(f"{internal_api}/v1/events/recognition", data=data, files=files)
                    if ingest_resp.status_code == 200:
                        logger.info(f"Successfully ingested {filename}")
                    else:
                        logger.error(f"Failed to ingest {filename}: {ingest_resp.status_code} {ingest_resp.text}")
                except Exception as e:
                    logger.error(f"Ingest error {filename}: {e}")
            
            # Pagination Logic
            page += 1
            if len(events) < limit:
                logger.info(f"Reached last page ({page-1}) for {date_str}.")
                break
                
        except Exception as e:
            logger.error(f"Pagination loop error on page {page}: {e}")
            break
    
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
    internal_api = _env("API_BASE_URL", "http://localhost:8000").rstrip("/")
    
    if not jwt:
        logger.error("JWT token not found in config.yaml under external_api.jwt_token")
        return

    while True:
        try:
            # Sync date from env or config
            sync_date = _env("SYNC_DATE") or ext_cfg.get("sync_date")
            process_external_events(api_base, jwt, internal_api, date_str=sync_date)
        except Exception as e:
            logger.error(f"Global sync error: {e}")
            
        logger.info(f"Sleeping for {poll_interval}s...")
        time.sleep(poll_interval)

if __name__ == "__main__":
    main()
