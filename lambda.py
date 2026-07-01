import io
import json
import os
import shutil
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote_plus

import boto3
import psycopg2
import psycopg2.extras
from PIL import Image, ImageOps


S3_BUCKET = os.environ.get("S3_BUCKET", "nikhith-ai-photo-gallery-dev")
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

RDS_HOST = os.environ.get("RDS_HOST", "photo-gallery-postgres-dev.c7o2u4ouqyim.us-east-1.rds.amazonaws.com")
RDS_PORT = int(os.environ.get("RDS_PORT", "5432"))
RDS_DB = os.environ.get("RDS_DB", "postgres")
RDS_USER = os.environ.get("RDS_USER", "photo_worker")
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")
RDS_SSLMODE = os.environ.get("RDS_SSLMODE", "require")

PREVIEW_MAX_SIDE = int(os.environ.get("PREVIEW_MAX_SIDE", "2048"))
PREVIEW_JPEG_QUALITY = int(os.environ.get("PREVIEW_JPEG_QUALITY", "85"))

RAW_IMAGE_EXTS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2"}
WEB_RENDERABLE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".gif", ".avif", ".bmp"}

s3 = boto3.client("s3", region_name=AWS_REGION)


def _log(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print("[RAW_PREVIEW] " + json.dumps(payload, default=str, sort_keys=True), flush=True)


def _event_summary(event: Dict[str, Any]) -> Dict[str, Any]:
    headers = event.get("headers") or {}
    return {
        "top_level_keys": sorted(str(key) for key in event.keys()),
        "is_http": _is_http_event(event),
        "is_s3_event": "Records" in event,
        "body_type": type(event.get("body")).__name__,
        "is_base64_encoded": bool(event.get("isBase64Encoded")),
        "header_names": sorted(str(key).lower() for key in headers.keys()),
        "request_context_keys": sorted(str(key) for key in (event.get("requestContext") or {}).keys()),
    }


def _json_response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    _log("http_response", status_code=status_code, payload=payload)
    return {
        "statusCode": status_code,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload, default=str),
    }


def _is_http_event(event: Dict[str, Any]) -> bool:
    return "body" in event or "headers" in event or "requestContext" in event


def _headers(event: Dict[str, Any]) -> Dict[str, str]:
    headers = event.get("headers") or {}
    return {str(k).lower(): str(v) for k, v in headers.items() if v is not None}


def _parse_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        _log("payload_parse_empty", reason="event_not_dict", event_type=type(event).__name__)
        return {}

    if "Records" in event:
        s3_keys = _s3_keys_from_event(event)
        _log("payload_parse_s3_event", s3_key_count=len(s3_keys), s3_keys=s3_keys[:25])
        return {"s3_keys": s3_keys}

    body = event.get("body")
    if body is None:
        _log("payload_parse_direct_event", payload=event)
        return event

    if event.get("isBase64Encoded"):
        import base64
        body = base64.b64decode(body).decode("utf-8")
        _log("payload_decode_base64", decoded_length=len(body))

    if isinstance(body, str) and body.strip():
        payload = json.loads(body)
        _log("payload_parse_json_body", payload=payload)
        return payload
    if isinstance(body, dict):
        _log("payload_parse_dict_body", payload=body)
        return body
    _log("payload_parse_empty", reason="unsupported_body", body_type=type(body).__name__)
    return {}


def _authorized(event: Dict[str, Any]) -> bool:
    expected = (
        os.environ.get("PHOTO_WORKER_ADMIN_KEY")
        or os.environ.get("RAW_PREVIEW_WORKER_ADMIN_KEY")
        or os.environ.get("ADMIN_KEY")
        or ""
    ).strip()
    if not expected:
        _log("auth_check", admin_key_configured=False, authorized=True)
        return True

    provided = _headers(event).get("x-admin-key", "").strip()
    authorized = bool(provided) and provided == expected
    _log(
        "auth_check",
        admin_key_configured=True,
        provided_admin_key=bool(provided),
        authorized=authorized,
    )
    return authorized


def _require_env() -> None:
    missing = []
    if not S3_BUCKET:
        missing.append("S3_BUCKET")
    if not RDS_HOST:
        missing.append("RDS_HOST")
    if not RDS_DB:
        missing.append("RDS_DB")
    if not RDS_USER:
        missing.append("RDS_USER")
    if not RDS_PASSWORD:
        missing.append("RDS_PASSWORD")
    if missing:
        _log("env_missing", missing=missing)
        raise RuntimeError(f"Missing required environment variables: {missing}")
    _log(
        "env_ready",
        s3_bucket=S3_BUCKET,
        aws_region=AWS_REGION,
        rds_host=RDS_HOST,
        rds_port=RDS_PORT,
        rds_db=RDS_DB,
        rds_user=RDS_USER,
        rds_sslmode=RDS_SSLMODE,
        preview_max_side=PREVIEW_MAX_SIDE,
        preview_jpeg_quality=PREVIEW_JPEG_QUALITY,
    )


def _connect():
    _require_env()
    _log("db_connect_start", host=RDS_HOST, port=RDS_PORT, db=RDS_DB, user=RDS_USER)
    return psycopg2.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        dbname=RDS_DB,
        user=RDS_USER,
        password=RDS_PASSWORD,
        sslmode=RDS_SSLMODE,
        cursor_factory=psycopg2.extras.RealDictCursor,
        application_name=os.environ.get("DB_APPLICATION_NAME", "photo_worker_lambda"),
        connect_timeout=int(os.environ.get("DB_CONNECT_TIMEOUT", "10")),
    )


def _s3_keys_from_event(event: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for record in event.get("Records") or []:
        s3_record = record.get("s3") or {}
        obj = s3_record.get("object") or {}
        key = obj.get("key")
        if key:
            keys.append(unquote_plus(str(key)))
    _log("s3_event_keys_extracted", count=len(keys), keys=keys[:25])
    return keys


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _fetch_rows(photo_ids: Iterable[str], s3_keys: Iterable[str]) -> List[Dict[str, Any]]:
    ids = list(dict.fromkeys(photo_ids))
    keys = list(dict.fromkeys(s3_keys))
    started = time.time()
    _log("fetch_rows_start", photo_ids=ids, s3_keys=keys, photo_id_count=len(ids), s3_key_count=len(keys))
    if not ids and not keys:
        _log("fetch_rows_skipped", reason="empty_ids_and_keys")
        return []

    clauses = []
    params: List[Any] = []

    if ids:
        clauses.append("id = ANY(%s::uuid[])")
        params.append(ids)
    if keys:
        clauses.append("(original_s3_key = ANY(%s::text[]) OR source_s3_key = ANY(%s::text[]))")
        params.extend([keys, keys])

    sql = f"""
        SELECT
          id,
          photo_uuid,
          storage_album_slug,
          storage_event_slug,
          file_name,
          original_s3_key,
          source_s3_key,
          clean_preview_s3_key
        FROM photos
        WHERE COALESCE(is_deleted, false) = false
          AND ({' OR '.join(clauses)})
        ORDER BY created_at
    """

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = list(cur.fetchall())
            _log(
                "fetch_rows_done",
                rows=len(rows),
                duration_ms=round((time.time() - started) * 1000, 2),
                sample=[
                    {
                        "id": str(row.get("id")),
                        "file_name": row.get("file_name"),
                        "original_s3_key": row.get("original_s3_key"),
                        "source_s3_key": row.get("source_s3_key"),
                        "clean_preview_s3_key": row.get("clean_preview_s3_key"),
                    }
                    for row in rows[:10]
                ],
            )
            return rows
    finally:
        conn.close()
        _log("db_connection_closed", operation="fetch_rows")


def _source_key(row: Dict[str, Any]) -> Optional[str]:
    return row.get("original_s3_key") or row.get("source_s3_key")


def _source_ext(row: Dict[str, Any]) -> str:
    key_ext = Path(_source_key(row) or "").suffix.lower()
    file_ext = Path(str(row.get("file_name") or "")).suffix.lower()
    if file_ext in RAW_IMAGE_EXTS:
        return file_ext
    return key_ext or file_ext


def _base_prefix_from_row(row: Dict[str, Any]) -> str:
    album_slug = row.get("storage_album_slug")
    event_slug = row.get("storage_event_slug")
    if album_slug and event_slug:
        return f"albums/{album_slug}/events/{event_slug}"

    key = _source_key(row) or ""
    marker = "/originals/"
    if marker in key:
        return key.split(marker, 1)[0]

    return str(Path(key).parent.parent).replace(".", "").strip("/")


def _preview_key_for_row(row: Dict[str, Any]) -> str:
    photo_uuid = row.get("photo_uuid") or row.get("id")
    return f"{_base_prefix_from_row(row)}/preview/{photo_uuid}.jpg"


def _s3_key_exists(key: str) -> bool:
    started = time.time()
    _log("s3_head_start", key=key, bucket=S3_BUCKET)
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        _log("s3_head_done", key=key, exists=True, duration_ms=round((time.time() - started) * 1000, 2))
        return True
    except Exception as error:
        _log("s3_head_done", key=key, exists=False, duration_ms=round((time.time() - started) * 1000, 2), error=repr(error))
        return False


def _download(key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _log("s3_download_start", key=key, local_path=str(path), bucket=S3_BUCKET)
    s3.download_file(S3_BUCKET, key, str(path))
    _log(
        "s3_download_done",
        key=key,
        local_path=str(path),
        local_bytes=path.stat().st_size if path.exists() else None,
        duration_ms=round((time.time() - started) * 1000, 2),
    )


def _upload(path: Path, key: str) -> None:
    started = time.time()
    _log(
        "s3_upload_start",
        key=key,
        local_path=str(path),
        local_bytes=path.stat().st_size if path.exists() else None,
        bucket=S3_BUCKET,
    )
    s3.upload_file(
        str(path),
        S3_BUCKET,
        key,
        ExtraArgs={"ContentType": "image/jpeg", "CacheControl": "public, max-age=31536000, immutable"},
    )
    _log("s3_upload_done", key=key, duration_ms=round((time.time() - started) * 1000, 2))


def _read_with_rawpy(path: Path, prefer_thumb: bool) -> Image.Image:
    import rawpy

    started = time.time()
    _log("rawpy_read_start", path=str(path), prefer_thumb=prefer_thumb)
    with rawpy.imread(str(path)) as raw:
        if prefer_thumb:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
                    _log("rawpy_thumb_used", path=str(path), size=img.size, duration_ms=round((time.time() - started) * 1000, 2))
                    return img
                img = Image.fromarray(thumb.data).convert("RGB")
                _log("rawpy_thumb_used", path=str(path), size=img.size, duration_ms=round((time.time() - started) * 1000, 2))
                return img
            except Exception as error:
                _log("rawpy_thumb_failed", path=str(path), error=repr(error))

        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
        img = Image.fromarray(rgb).convert("RGB")
        _log("rawpy_postprocess_done", path=str(path), size=img.size, duration_ms=round((time.time() - started) * 1000, 2))
        return img


def _read_image(path: Path) -> Image.Image:
    ext = path.suffix.lower()
    _log("read_image_start", path=str(path), ext=ext)

    if ext in RAW_IMAGE_EXTS:
        img = ImageOps.exif_transpose(_read_with_rawpy(path, prefer_thumb=True)).convert("RGB")
        _log("read_image_done", path=str(path), decoder="rawpy", size=img.size)
        return img

    try:
        img = Image.open(path)
        img.load()
        result = ImageOps.exif_transpose(img).convert("RGB")
        _log("read_image_done", path=str(path), decoder="pillow", size=result.size)
        return result
    except Exception as pil_error:
        _log("read_image_pillow_failed", path=str(path), error=repr(pil_error))
        try:
            result = ImageOps.exif_transpose(_read_with_rawpy(path, prefer_thumb=True)).convert("RGB")
            _log("read_image_done", path=str(path), decoder="rawpy_fallback", size=result.size)
            return result
        except Exception as raw_error:
            _log("read_image_failed", path=str(path), pil_error=repr(pil_error), raw_error=repr(raw_error))
            raise RuntimeError(f"Could not decode image. PIL={pil_error!r}; rawpy={raw_error!r}") from raw_error


def _write_preview(source: Path, output: Path) -> Tuple[int, int]:
    started = time.time()
    _log("preview_write_start", source=str(source), output=str(output), max_side=PREVIEW_MAX_SIDE, quality=PREVIEW_JPEG_QUALITY)
    img = _read_image(source)
    original_size = img.size
    img.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output, "JPEG", quality=PREVIEW_JPEG_QUALITY, optimize=True)
    _log(
        "preview_write_done",
        source=str(source),
        output=str(output),
        original_size=original_size,
        preview_size=img.size,
        output_bytes=output.stat().st_size if output.exists() else None,
        duration_ms=round((time.time() - started) * 1000, 2),
    )
    return img.size


def _needs_preview(row: Dict[str, Any]) -> bool:
    source_key = _source_key(row)
    if not source_key:
        _log("needs_preview", photo_id=str(row.get("id")), needs_preview=False, reason="missing_source_key")
        return False

    ext = _source_ext(row)
    if ext in RAW_IMAGE_EXTS:
        _log("needs_preview", photo_id=str(row.get("id")), needs_preview=True, reason="raw_ext", ext=ext, source_key=source_key)
        return True
    needs_preview = ext not in WEB_RENDERABLE_EXTS
    _log("needs_preview", photo_id=str(row.get("id")), needs_preview=needs_preview, reason="web_renderable_check", ext=ext, source_key=source_key)
    return needs_preview


def _record_preview(photo_id: str, preview_key: str, size: Optional[Tuple[int, int]] = None) -> None:
    started = time.time()
    _log("record_preview_start", photo_id=photo_id, preview_key=preview_key, size=size)
    set_parts = ["clean_preview_s3_key = %s", "updated_at = now()"]
    params: List[Any] = [preview_key]

    if size:
        set_parts.extend(["width = COALESCE(width, %s)", "height = COALESCE(height, %s)"])
        params.extend([size[0], size[1]])

    params.append(photo_id)

    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE photos SET {', '.join(set_parts)} WHERE id = %s::uuid",
                    params,
                )
                _log("record_preview_done", photo_id=photo_id, preview_key=preview_key, rowcount=cur.rowcount, duration_ms=round((time.time() - started) * 1000, 2))
    finally:
        conn.close()
        _log("db_connection_closed", operation="record_preview", photo_id=photo_id)


def _process_row(row: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    photo_id = str(row.get("id"))
    source_key = _source_key(row)
    _log(
        "process_row_start",
        photo_id=photo_id,
        file_name=row.get("file_name"),
        source_key=source_key,
        source_ext=_source_ext(row),
        existing_preview_key=row.get("clean_preview_s3_key"),
    )
    if not source_key:
        _log("process_row_skipped", photo_id=photo_id, reason="missing_source_key")
        return {"photo_id": photo_id, "status": "skipped", "reason": "missing_source_key"}

    if not _needs_preview(row):
        _log("process_row_skipped", photo_id=photo_id, reason="web_renderable", duration_ms=round((time.time() - started) * 1000, 2))
        return {"photo_id": photo_id, "status": "skipped", "reason": "web_renderable"}

    preview_key = _preview_key_for_row(row)
    existing_key = row.get("clean_preview_s3_key")
    if existing_key and existing_key == preview_key and _s3_key_exists(existing_key):
        _log("process_row_skipped", photo_id=photo_id, reason="preview_exists", preview_key=existing_key, duration_ms=round((time.time() - started) * 1000, 2))
        return {"photo_id": photo_id, "status": "skipped", "reason": "preview_exists", "preview_key": existing_key}

    if _s3_key_exists(preview_key):
        _record_preview(photo_id, preview_key)
        _log("process_row_done", photo_id=photo_id, reason="recorded_existing", preview_key=preview_key, duration_ms=round((time.time() - started) * 1000, 2))
        return {"photo_id": photo_id, "status": "ok", "reason": "recorded_existing", "preview_key": preview_key}

    tmpdir = Path(tempfile.mkdtemp(prefix=f"raw-preview-{photo_id}-"))
    _log("tempdir_created", photo_id=photo_id, tmpdir=str(tmpdir))
    try:
        suffix = _source_ext(row) or ".img"
        original = tmpdir / f"original{suffix}"
        preview = tmpdir / "preview.jpg"
        _log("process_row_render_start", photo_id=photo_id, source_key=source_key, preview_key=preview_key, suffix=suffix)
        _download(source_key, original)
        size = _write_preview(original, preview)
        _upload(preview, preview_key)
        _record_preview(photo_id, preview_key, size)
        _log("process_row_done", photo_id=photo_id, status="ok", preview_key=preview_key, width=size[0], height=size[1], duration_ms=round((time.time() - started) * 1000, 2))
        return {"photo_id": photo_id, "status": "ok", "preview_key": preview_key, "width": size[0], "height": size[1]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _log("tempdir_removed", photo_id=photo_id, tmpdir=str(tmpdir))


# --------------------------------------------------------------------------- #
# Face-detection trigger (RunPod serverless + Vercel redeploy)
#
# When the incoming payload carries a video_url / target_urls, this Lambda
# reuses (or creates) a RunPod serverless endpoint running the latest
# face-occurrence worker image, submits the detection job asynchronously, then
# upserts a Vercel env var with the job reference and triggers a redeploy.
# --------------------------------------------------------------------------- #
FACE_GITHUB_REPO = os.environ.get("FACE_GITHUB_REPO", "NikhithVasa/face-occurance-detector")
FACE_GITHUB_WORKFLOW_FILE = os.environ.get("FACE_GITHUB_WORKFLOW_FILE", "build-and-push.yml")
FACE_GITHUB_TOKEN = os.environ.get("FACE_GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN", "")).strip()

FACE_DOCKER_IMAGE = os.environ.get("FACE_DOCKER_IMAGE", "dsfsdfnikhith/face-occurrence-worker")
FACE_IMAGE_TAG_OVERRIDE = os.environ.get("FACE_IMAGE_TAG", "").strip()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
RUNPOD_ENDPOINT_NAME = os.environ.get("RUNPOD_ENDPOINT_NAME", "face-occurrence-worker")
RUNPOD_GPU_IDS = os.environ.get("RUNPOD_GPU_IDS", "NVIDIA GeForce RTX 4090")
RUNPOD_WORKERS_MAX = int(os.environ.get("RUNPOD_WORKERS_MAX", "1"))
RUNPOD_CONTAINER_DISK_GB = int(os.environ.get("RUNPOD_CONTAINER_DISK_GB", "20"))
RUNPOD_IDLE_TIMEOUT = int(os.environ.get("RUNPOD_IDLE_TIMEOUT", "5"))

# Public URL of this Lambda (Function URL / API Gateway). RunPod POSTs the job
# result here on completion so results are written to the DB. If empty, the job
# is still submitted but results must be fetched by polling RunPod /status.
LAMBDA_WEBHOOK_BASE = os.environ.get("LAMBDA_WEBHOOK_BASE", "").strip()
# Optional shared secret appended to the webhook URL and validated on callback.
WEBHOOK_SECRET = os.environ.get("FACE_WEBHOOK_SECRET", "").strip()

# Everything except video_url / target_urls is hardcoded for the detection job.
FACE_DETECTION_DEFAULTS: Dict[str, Any] = {
    "fps": 4,
    "similarity_threshold": 0.4,
    "merge_gap_sec": 2,
    "min_interval_sec": 0,
}

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"


def _http_json(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {error.code} {method} {url}: {detail}") from error


def _face_resolve_image_tag() -> str:
    if FACE_IMAGE_TAG_OVERRIDE:
        return FACE_IMAGE_TAG_OVERRIDE
    url = (
        f"https://api.github.com/repos/{FACE_GITHUB_REPO}/actions/workflows/"
        f"{FACE_GITHUB_WORKFLOW_FILE}/runs?branch=main&status=success&per_page=1"
    )
    headers = {"User-Agent": "face-trigger-lambda"}
    if FACE_GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {FACE_GITHUB_TOKEN}"
    data = _http_json(url, headers=headers)
    runs = data.get("workflow_runs") or []
    if not runs or not runs[0].get("run_number"):
        raise RuntimeError("Could not resolve latest image tag from GitHub Actions.")
    tag = f"v{runs[0]['run_number']}"
    _log("face_image_tag_resolved", tag=tag, run_id=runs[0].get("id"))
    return tag


def _runpod_graphql(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY is not set.")
    result = _http_json(
        f"{RUNPOD_GRAPHQL_URL}?api_key={RUNPOD_API_KEY}",
        method="POST",
        payload={"query": query, "variables": variables or {}},
    )
    if result.get("errors"):
        raise RuntimeError(f"RunPod GraphQL error: {json.dumps(result['errors'])}")
    return result.get("data") or {}


def _image_name_no_tag(image_ref: str) -> str:
    return image_ref.rsplit(":", 1)[0] if ":" in image_ref else image_ref


def _face_save_template(image_ref: str, template_id: Optional[str] = None, name: Optional[str] = None) -> str:
    template_input: Dict[str, Any] = {
        "name": name or f"{RUNPOD_ENDPOINT_NAME}-template",
        "imageName": image_ref,
        "dockerArgs": "",
        "containerDiskInGb": RUNPOD_CONTAINER_DISK_GB,
        "volumeInGb": 0,
        "ports": "",
        "env": [],
        "isServerless": True,
    }
    if template_id:
        template_input["id"] = template_id
    data = _runpod_graphql(
        "mutation ($input: SaveTemplateInput!) { saveTemplate(input: $input) { id imageName name } }",
        {"input": template_input},
    )
    return str((data.get("saveTemplate") or {}).get("id"))


def _face_save_endpoint(template_id: str) -> str:
    endpoint_input = {
        "name": RUNPOD_ENDPOINT_NAME,
        "templateId": template_id,
        "gpuIds": RUNPOD_GPU_IDS,
        "idleTimeout": RUNPOD_IDLE_TIMEOUT,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "workersMin": 0,
        "workersMax": RUNPOD_WORKERS_MAX,
        "networkVolumeId": "",
        "locations": "",
    }
    data = _runpod_graphql(
        "mutation ($input: EndpointInput!) { saveEndpoint(input: $input) { id name templateId } }",
        {"input": endpoint_input},
    )
    return str((data.get("saveEndpoint") or {}).get("id"))


def _face_ensure_endpoint(image_ref: str) -> str:
    """Reuse a serverless endpoint running image_ref (by image name), else create one."""
    data = _runpod_graphql(
        """
        query { myself {
          endpoints { id name templateId }
          podTemplates { id name imageName isServerless }
        } }
        """
    )
    myself = data.get("myself") or {}
    templates_by_id = {t.get("id"): t for t in (myself.get("podTemplates") or [])}
    target_name = _image_name_no_tag(image_ref)

    for endpoint in myself.get("endpoints") or []:
        template = templates_by_id.get(endpoint.get("templateId"))
        if not template:
            continue
        if _image_name_no_tag(str(template.get("imageName", ""))) != target_name:
            continue
        if template.get("imageName") != image_ref:
            _log("face_endpoint_template_bump", endpoint_id=endpoint.get("id"),
                 old_image=template.get("imageName"), new_image=image_ref)
            _face_save_template(image_ref, template_id=template.get("id"), name=template.get("name"))
        _log("face_endpoint_reused", endpoint_id=endpoint.get("id"), name=endpoint.get("name"))
        return str(endpoint.get("id"))

    template_id = _face_save_template(image_ref)
    endpoint_id = _face_save_endpoint(template_id)
    _log("face_endpoint_created", endpoint_id=endpoint_id, template_id=template_id, image=image_ref)
    return endpoint_id


def _face_submit_job(
    endpoint_id: str, video_url: str, target_urls: List[str], webhook: Optional[str] = None
) -> str:
    body: Dict[str, Any] = {
        "input": {"video_url": video_url, "target_urls": target_urls, **FACE_DETECTION_DEFAULTS}
    }
    if webhook:
        body["webhook"] = webhook
    result = _http_json(
        f"https://api.runpod.ai/v2/{endpoint_id}/run",
        method="POST",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        payload=body,
        timeout=60.0,
    )
    job_id = result.get("id")
    if not job_id:
        raise RuntimeError(f"RunPod /run returned no job id: {json.dumps(result)}")
    _log("face_job_submitted", endpoint_id=endpoint_id, job_id=job_id, status=result.get("status"))
    return str(job_id)


# --------------------------------------------------------------------------- #
# DB writes (same psycopg2 path as the rest of this Lambda)
# --------------------------------------------------------------------------- #
def _face_insert_video(meta: Dict[str, Any], target_urls: List[str]) -> str:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO videos (
                      album_id, album_event_id, customer_id, file_name,
                      original_s3_key, storage_album_slug, storage_event_slug,
                      detection_params, target_person_id, target_s3_keys,
                      detection_status, created_at, updated_at
                    ) VALUES (
                      %(album_id)s, %(album_event_id)s, %(customer_id)s, %(file_name)s,
                      %(original_s3_key)s, %(storage_album_slug)s, %(storage_event_slug)s,
                      %(detection_params)s, %(target_person_id)s, %(target_s3_keys)s,
                      'processing', now(), now()
                    ) RETURNING id
                    """,
                    {
                        "album_id": meta.get("album_id"),
                        "album_event_id": meta.get("album_event_id"),
                        "customer_id": meta.get("customer_id"),
                        "file_name": meta.get("file_name"),
                        "original_s3_key": meta.get("original_s3_key"),
                        "storage_album_slug": meta.get("storage_album_slug"),
                        "storage_event_slug": meta.get("storage_event_slug"),
                        "detection_params": psycopg2.extras.Json(FACE_DETECTION_DEFAULTS),
                        "target_person_id": meta.get("target_person_id"),
                        "target_s3_keys": psycopg2.extras.Json(target_urls),
                    },
                )
                video_id = str(cur.fetchone()["id"])
                _log("face_video_inserted", video_id=video_id, album_id=meta.get("album_id"))
                return video_id
    finally:
        conn.close()


def _face_set_video_job(video_id: str, endpoint_id: str, job_id: str) -> None:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE videos SET runpod_endpoint_id=%s, runpod_job_id=%s, updated_at=now() "
                    "WHERE id=%s::uuid",
                    (endpoint_id, job_id, video_id),
                )
    finally:
        conn.close()


def _face_write_result(video_id: str, output: Dict[str, Any]) -> Dict[str, Any]:
    matches = output.get("matches") or []
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT album_id, album_event_id, target_person_id FROM videos WHERE id=%s::uuid",
                    (video_id,),
                )
                row = cur.fetchone() or {}
                album_id = row.get("album_id")
                album_event_id = row.get("album_event_id")
                person_id = row.get("target_person_id")

                cur.execute(
                    """
                    UPDATE videos SET
                      detection_status='completed',
                      result_json=%s,
                      duration_sec=%s,
                      model=%s,
                      match_count=%s,
                      detection_error=NULL,
                      completed_at=now(),
                      updated_at=now()
                    WHERE id=%s::uuid
                    """,
                    (
                        psycopg2.extras.Json(output),
                        output.get("duration_sec"),
                        output.get("model"),
                        len(matches),
                        video_id,
                    ),
                )

                # Idempotent: clear any prior matches for this video, then insert.
                cur.execute("DELETE FROM video_face_matches WHERE video_id=%s::uuid", (video_id,))
                for m in matches:
                    cur.execute(
                        """
                        INSERT INTO video_face_matches (
                          video_id, album_id, album_event_id, person_id,
                          start_sec, end_sec, start_time, end_time,
                          max_similarity, avg_similarity, frames_matched, verified
                        ) VALUES (
                          %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            video_id, album_id, album_event_id, person_id,
                            m.get("start_sec"), m.get("end_sec"),
                            m.get("start_time"), m.get("end_time"),
                            m.get("max_similarity"), m.get("avg_similarity"),
                            m.get("frames_matched"), m.get("verified"),
                        ),
                    )
        _log("face_result_written", video_id=video_id, match_count=len(matches))
        return {"ok": True, "video_id": video_id, "match_count": len(matches)}
    finally:
        conn.close()


def _face_fail_video(video_id: str, error: str) -> None:
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE videos SET detection_status='failed', detection_error=%s, "
                    "completed_at=now(), updated_at=now() WHERE id=%s::uuid",
                    (error[:2000], video_id),
                )
        _log("face_video_failed", video_id=video_id, error=error[:200])
    finally:
        conn.close()


def _is_face_trigger(payload: Dict[str, Any]) -> bool:
    source = payload.get("input") if isinstance(payload.get("input"), dict) else payload
    return bool(source.get("video_url") or source.get("target_urls") or source.get("target_url"))


def handle_face_trigger(payload: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    source = payload.get("input") if isinstance(payload.get("input"), dict) else payload

    def field(key: str) -> Any:
        return source.get(key) if source.get(key) is not None else payload.get(key)

    video_url = str(field("video_url") or "").strip()
    target_urls = _string_list(field("target_urls") or field("target_url"))
    _log("face_handle_start", has_video=bool(video_url), target_count=len(target_urls))

    if not video_url:
        raise RuntimeError("video_url is required.")
    if not target_urls:
        raise RuntimeError("target_urls is required (one or more image URLs).")

    meta = {
        "album_id": field("album_id"),
        "album_event_id": field("album_event_id"),
        "customer_id": field("customer_id"),
        "file_name": field("file_name"),
        "original_s3_key": field("original_s3_key"),
        "storage_album_slug": field("storage_album_slug"),
        "storage_event_slug": field("storage_event_slug"),
        "target_person_id": field("target_person_id"),
    }

    # 1. Create the videos row up front so the UI can show a "processing" entry.
    video_id = _face_insert_video(meta, target_urls)

    try:
        # 2. Ensure endpoint + submit the job with a completion webhook.
        tag = _face_resolve_image_tag()
        image_ref = f"{FACE_DOCKER_IMAGE}:{tag}"
        endpoint_id = _face_ensure_endpoint(image_ref)

        webhook = None
        if LAMBDA_WEBHOOK_BASE:
            sep = "&" if "?" in LAMBDA_WEBHOOK_BASE else "?"
            webhook = f"{LAMBDA_WEBHOOK_BASE}{sep}action=detection_result&video_id={video_id}"
            if WEBHOOK_SECRET:
                webhook += f"&secret={WEBHOOK_SECRET}"

        job_id = _face_submit_job(endpoint_id, video_url, target_urls, webhook=webhook)
        _face_set_video_job(video_id, endpoint_id, job_id)
    except Exception as error:  # noqa: BLE001 - mark the row so the UI shows failure
        _face_fail_video(video_id, repr(error))
        raise

    result = {
        "ok": True,
        "mode": "face_trigger",
        "video_id": video_id,
        "endpointId": endpoint_id,
        "jobId": job_id,
        "imageTag": tag,
        "detection_status": "processing",
        "webhook": bool(webhook),
    }
    _log("face_handle_done", duration_ms=round((time.time() - started) * 1000, 2), **result)
    return result


def handle_detection_webhook(video_id: Optional[str], body: Dict[str, Any]) -> Dict[str, Any]:
    """Called by RunPod when a detection job finishes; writes results to the DB."""
    if not video_id:
        raise RuntimeError("detection webhook missing video_id.")
    status = str(body.get("status") or "").upper()
    _log("face_webhook_received", video_id=video_id, status=status)

    if status == "COMPLETED":
        output = body.get("output") or {}
        if isinstance(output, dict) and output.get("error"):
            _face_fail_video(video_id, str(output["error"]))
            return {"ok": False, "video_id": video_id, "error": output["error"]}
        return _face_write_result(video_id, output if isinstance(output, dict) else {})

    error = body.get("error") or f"RunPod job {status or 'UNKNOWN'}"
    _face_fail_video(video_id, str(error))
    return {"ok": False, "video_id": video_id, "status": status}



def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    if _is_face_trigger(payload):
        return handle_face_trigger(payload)

    started = time.time()
    photo_ids = _string_list(payload.get("photoIds") or payload.get("photo_ids"))
    s3_keys = _string_list(payload.get("s3Keys") or payload.get("s3_keys"))
    _log("handle_start", payload=payload, photo_ids=photo_ids, s3_keys=s3_keys)

    rows = _fetch_rows(photo_ids, s3_keys)
    results = []
    ok = skipped = failed = 0

    for row in rows:
        try:
            result = _process_row(row)
        except Exception as error:
            failed += 1
            result = {
                "photo_id": str(row.get("id")),
                "status": "failed",
                "error": repr(error),
            }
            _log("process_row_error", photo_id=str(row.get("id")), error=repr(error))
            print(traceback.format_exc(), flush=True)
        else:
            if result.get("status") == "ok":
                ok += 1
            elif result.get("status") == "skipped":
                skipped += 1
            else:
                failed += 1
        results.append(result)
        _log("process_row_result", result=result)

    summary = {
        "ok": failed == 0,
        "requested_photo_ids": len(photo_ids),
        "requested_s3_keys": len(s3_keys),
        "rows": len(rows),
        "rendered": ok,
        "skipped": skipped,
        "failed": failed,
        "results": results[:100],
    }
    _log("handle_done", **summary, duration_ms=round((time.time() - started) * 1000, 2))
    return summary


def lambda_handler(event, context):
    started = time.time()
    is_http = isinstance(event, dict) and _is_http_event(event)
    _log(
        "lambda_start",
        request_id=getattr(context, "aws_request_id", None),
        function_name=getattr(context, "function_name", None),
        function_version=getattr(context, "function_version", None),
        memory_limit_mb=getattr(context, "memory_limit_in_mb", None),
        is_http=is_http,
        event_summary=_event_summary(event) if isinstance(event, dict) else {"event_type": type(event).__name__},
    )

    try:
        # RunPod completion webhook lands here as an HTTP POST with query params.
        # It cannot send the admin key, so route it before the auth check and
        # guard it with the (unguessable) video_id plus optional shared secret.
        qs = (event.get("queryStringParameters") or {}) if isinstance(event, dict) else {}
        if isinstance(qs, dict) and (qs.get("action") == "detection_result" or qs.get("video_id")):
            if WEBHOOK_SECRET and qs.get("secret") != WEBHOOK_SECRET:
                _log("lambda_webhook_unauthorized", request_id=getattr(context, "aws_request_id", None))
                return _json_response(401, {"ok": False, "error": "Unauthorized"})
            body = _parse_payload(event if isinstance(event, dict) else {})
            result = handle_detection_webhook(qs.get("video_id"), body)
            _log("lambda_webhook_done", request_id=getattr(context, "aws_request_id", None), result=result)
            return _json_response(200, result) if is_http else result

        if is_http and not _authorized(event):
            _log("lambda_unauthorized", request_id=getattr(context, "aws_request_id", None))
            return _json_response(401, {"ok": False, "error": "Unauthorized"})

        payload = _parse_payload(event if isinstance(event, dict) else {})
        result = handle(payload)
        _log("lambda_done", request_id=getattr(context, "aws_request_id", None), result=result, duration_ms=round((time.time() - started) * 1000, 2))
        return _json_response(200 if result.get("ok") else 500, result) if is_http else result
    except Exception as error:
        payload = {"ok": False, "error": str(error)}
        _log("lambda_error", request_id=getattr(context, "aws_request_id", None), error=repr(error), duration_ms=round((time.time() - started) * 1000, 2))
        print(traceback.format_exc(), flush=True)
        return _json_response(500, payload) if is_http else payload
