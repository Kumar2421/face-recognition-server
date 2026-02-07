#!/usr/bin/env python3
"""
Download employee images from CSV into folders named by employeeName.
CSV columns expected:
  id,employeeName,employeeId,jobRole,jobShift,employeeImages,...
The employeeImages field contains a JSON-like list of URLs.
"""

import csv
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import List

USER_AGENT = "Mozilla/5.0 (compatible; face-service-downloader/1.0)"

def download_image(url: str, dest_path: Path) -> bool:
    """Download a single image URL to dest_path, overwrite if exists."""
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=20) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        print(f"Downloaded: {dest_path.name}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}", file=sys.stderr)
        return False

def parse_image_urls(cell: str) -> List[str]:
    """Parse the employeeImages cell which may be a JSON list or single URL."""
    cell = cell.strip()
    if not cell:
        return []
    # Strip surrounding brackets/braces if present
    if cell.startswith("{") or cell.startswith("["):
        cell = cell[1:]
    if cell.endswith("}") or cell.endswith("]"):
        cell = cell[:-1]
    # Replace curly braces with square brackets for JSON parsing
    cell = cell.replace("{", "[").replace("}", "]")
    try:
        urls = json.loads(cell)
        if isinstance(urls, list):
            return [u.strip() for u in urls if isinstance(u, str) and u.strip()]
        elif isinstance(urls, str):
            return [urls.strip()]
        else:
            return []
    except json.JSONDecodeError:
        # Fallback: split by commas and strip whitespace/brackets
        parts = [p.strip().strip("[]{}") for p in cell.split(",") if p.strip()]
        return [p for p in parts if p]

def main(csv_path: str, out_dir: str = "employee_images"):
    out_root = Path(out_dir)
    out_root.mkdir(exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("employeeName", "").strip()
            if not name:
                continue
            # Sanitize folder name (remove path separators and extra spaces)
            folder_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("_")
            folder_path = out_root / folder_name
            folder_path.mkdir(exist_ok=True)

            urls = parse_image_urls(row.get("employeeImages", ""))
            if not urls:
                print(f"No image URLs for employee: {name}")
                continue

            for idx, url in enumerate(urls, start=1):
                # Determine file extension from URL or default to .jpg
                parsed = urlparse(url)
                ext = (os.path.splitext(parsed.path)[1] or ".jpg").lower()
                # Ensure extension is common image type
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    ext = ".jpg"
                filename = f"{idx:02d}{ext}"
                dest_path = folder_path / filename
                download_image(url, dest_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_employee_images.py <path_to_csv> [output_dir]")
        sys.exit(1)
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "employee_images"
    main(csv_file, output_dir)
