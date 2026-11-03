"""
Image Downloader for Disaster Coordination Hub.
Downloads attached disaster images locally for vision classifiers to inspect.
"""

import logging
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from collectors.models import Report

logger = logging.getLogger("DisasterHub.ImageDownloader")

IMAGE_EXTENSIONS = {
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG\r\n\x1a\n": ".png",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
    b"RIFF": ".webp",
}

class ImageDownloader:
    """Downloads remote media URLs to local storage for computer vision classification."""

    def __init__(self, output_dir: Optional[Path] = None, timeout: int = 6, max_workers: int = 4):
        if output_dir is None:
            # Default to data/images relative to project
            base_dir = Path(__file__).resolve().parent.parent
            self.output_dir = base_dir / "data" / "images"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_workers = max_workers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DisasterHub/1.0"
        }

    def download_reports_images(self, reports: List[Report], max_images_per_report: int = 2) -> int:
        """
        Download images for all given reports concurrently.
        Populates report.local_image_paths for successful downloads.
        Returns total number of downloaded images.
        """
        download_tasks = []

        for rep in reports:
            if not rep.media_urls:
                continue

            for idx, img_url in enumerate(rep.media_urls[:max_images_per_report]):
                download_tasks.append((rep, idx, img_url))

        if not download_tasks:
            logger.info("No media URLs found across reports to download.")
            return 0

        logger.info(f"Downloading {len(download_tasks)} candidate images to {self.output_dir}...")
        downloaded_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._download_single_image, rep.id, idx, url): (rep, url)
                for rep, idx, url in download_tasks
            }

            for future in as_completed(future_to_task):
                rep, url = future_to_task[future]
                try:
                    local_path = future.result()
                    if local_path and os.path.exists(local_path):
                        rep.local_image_paths.append(str(local_path))
                        downloaded_count += 1
                except Exception as ex:
                    logger.debug(f"Failed to download image {url}: {ex}")

        logger.info(f"Downloaded {downloaded_count} local images successfully.")
        return downloaded_count

    def _download_single_image(self, report_id: str, img_idx: int, url: str) -> Optional[str]:
        """Download a single image from URL and save to local disk."""
        # Sanitize report id for filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in report_id)
        
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read(10 * 1024 * 1024)  # 10MB max limit
                if len(data) < 100:  # Too small to be an image
                    return None

                # Determine file extension from magic bytes
                ext = ".jpg"
                for magic, ext_candidate in IMAGE_EXTENSIONS.items():
                    if data.startswith(magic):
                        ext = ext_candidate
                        break

                filename = f"{safe_id}_{img_idx}{ext}"
                target_path = self.output_dir / filename

                with open(target_path, "wb") as f:
                    f.write(data)

                return str(target_path)
        except Exception:
            return None
