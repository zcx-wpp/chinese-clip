"""
Common utility helpers.
"""

import logging
import os
import uuid
from urllib.parse import urlparse

import oss2
import requests
from oss2.credentials import EnvironmentVariableCredentialsProvider
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
endpoint = os.getenv("OSS_ENDPOINT", "https://oss-cn-shenzhen-internal.aliyuncs.com")
region = os.getenv("OSS_REGION", "cn-shenzhen")
bucket_name = os.getenv("OSS_BUCKET_NAME", "gbzmkj")
bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_extension_from_content_type(content_type):
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
    }
    return mapping.get(content_type.split(";")[0].strip().lower(), ".bin")


def upload_file_oss(local_file_path, object_name, bucket=bucket):
    if not os.path.exists(local_file_path):
        logging.error("Local file not found: %s", local_file_path)
        return None

    logging.info("Uploading file '%s' to OSS path '%s'...", local_file_path, object_name)

    try:
        bucket.put_object_from_file(object_name, local_file_path)
        logging.info("File upload succeeded.")
        logging.info("OSS path: %s", object_name)
        return object_name
    except oss2.exceptions.OssError as exc:
        logging.error("File upload failed: %s", exc)
        return None


def download_file_oss(object_name, local_file_path, bucket=bucket):
    local_dir = os.path.dirname(local_file_path)
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        logging.info("Created local directory: %s", local_dir)

    logging.info("Downloading OSS object '%s' to '%s'...", object_name, local_file_path)

    try:
        bucket.get_object_to_file(object_name, local_file_path)
        logging.info("File download succeeded.")
        return True
    except oss2.exceptions.OssError as exc:
        logging.error("File download failed: %s", exc)
        return False


def download_file(url, output_dir=None, filename=None):
    try:
        if os.path.exists(url):
            return url

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "temp")
        ensure_dir(output_dir)

        if not url.startswith(("http://", "https://")):
            if filename is None:
                filename = os.path.basename(url)
                if not filename or "." not in filename:
                    filename = f"file_{str(uuid.uuid4())[:8]}.bin"

            output_path = os.path.join(output_dir, filename)
            success = download_file_oss(object_name=url, local_file_path=output_path)
            if success:
                return output_path
            raise Exception("Failed to download file from OSS. Check configuration or permissions.")

        head = requests.head(url, allow_redirects=True, timeout=10)
        content_type = head.headers.get("Content-Type", "")

        if filename is None:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or "." not in filename:
                ext = get_extension_from_content_type(content_type)
                filename = f"file_{str(uuid.uuid4())[:8]}{ext}"

        output_path = os.path.join(output_dir, filename)

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        logging.info("Downloading file: %s -> %s", url, output_path)

        with open(output_path, "wb") as file_obj, tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
                    bar.update(len(chunk))

        logging.info("Download complete: %s", output_path)
        return output_path
    except Exception as exc:
        logging.error("Download failed: %s, error: %s", url, exc)
        raise Exception(f"Failed to download file {url}: {exc}") from exc
