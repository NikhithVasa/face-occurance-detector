"""
RunPod serverless handler for the face occurrence detector.

The InsightFace model is loaded once at cold start and reused across jobs. Each
job runs the shared detection pipeline and returns the result dict.

Job input schema (``job["input"]``):

    {
      "video_path":   "/runpod-volume/input/video.mp4",   # OR "video_url"
      "video_url":    "https://.../video.mp4",
      "target_paths": ["/runpod-volume/targets/front.jpg"],# OR "target_urls"
      "target_urls":  ["https://.../front.jpg", "https://.../side.jpg"],

      "fps": 1,
      "chunks": 4,
      "parallel_chunks": 2,
      "similarity_threshold": 0.55,
      "merge_gap_sec": 1.5,
            "min_interval_sec": 1.0,

            "video_id": "optional-existing-videos-row-uuid",
            "album_id": "album uuid",
            "album_event_id": "event uuid",
            "customer_id": "optional customer uuid",
            "target_person_id": "optional people.id uuid",
            "file_name": "videoplayback.mp4",
            "original_s3_key": "albums/.../videos/videoplayback.mp4",
            "storage_album_slug": "album-slug",
            "storage_event_slug": "event-slug"
    }

Provide either the ``*_path`` form (files on a mounted network volume) or the
``*_url`` form (downloaded to a temp dir for the duration of the job).
"""

from __future__ import annotations

import os
import tempfile
import urllib.parse
import urllib.request
import uuid

import boto3
import psycopg2
import psycopg2.extras
import runpod

from face_occurrence_detector.config import (
    DEFAULT_CHUNKS,
    DEFAULT_DET_SIZE,
    DEFAULT_FPS,
    DEFAULT_MERGE_GAP_SEC,
    DEFAULT_MIN_INTERVAL_SEC,
    DEFAULT_MODEL_NAME,
    DEFAULT_PARALLEL_CHUNKS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VERIFY_MAX_TOKENS,
    DEFAULT_VERIFY_PROMPT,
)
from face_occurrence_detector.insightface_matcher import InsightFaceMatcher
from face_occurrence_detector.pipeline import run_detection

# ---------------------------------------------------------------------------- #
# Cold start: load the model once. CTX_ID defaults to GPU 0; set CTX_ID=-1 to
# force CPU. This raises if a GPU was requested but CUDA is unavailable, which
# correctly fails the worker rather than silently running on CPU.
# ---------------------------------------------------------------------------- #
_CTX_ID = int(os.getenv("CTX_ID", "0"))
_DET_SIZE = int(os.getenv("DET_SIZE", str(DEFAULT_DET_SIZE)))

_matcher = InsightFaceMatcher(
    model_name=DEFAULT_MODEL_NAME,
    ctx_id=_CTX_ID,
    det_size=(_DET_SIZE, _DET_SIZE),
)
print(f"Worker ready. ONNX Runtime providers: {', '.join(_matcher.providers)}")

RDS_HOST = os.getenv("RDS_HOST", "photo-gallery-postgres-dev.c7o2u4ouqyim.us-east-1.rds.amazonaws.com")
RDS_PORT = int(os.getenv("RDS_PORT", "5432"))
RDS_DB = os.getenv("RDS_DB", "postgres")
RDS_USER = os.getenv("RDS_USER", "photo_worker")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")
RDS_SSLMODE = os.getenv("RDS_SSLMODE", "require")
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
S3_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

_METADATA_KEYS = {
    "video_id",
    "videoId",
    "album_id",
    "albumId",
    "album_event_id",
    "albumEventId",
    "event_id",
    "eventId",
    "customer_id",
    "customerId",
    "target_person_id",
    "targetPersonId",
    "person_id",
    "personId",
    "file_name",
    "fileName",
    "original_s3_key",
    "originalS3Key",
    "storage_album_slug",
    "storageAlbumSlug",
    "storage_event_slug",
    "storageEventSlug",
    "target_s3_keys",
    "targetS3Keys",
    "target_person_ids",
    "targetPersonIds",
    "selected_person_ids",
    "selectedPersonIds",
    "runpod_endpoint_id",
    "runpodEndpointId",
}


def _download(url: str, dest_dir: str) -> str:
    """Download an http(s) URL into dest_dir and return the local path."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    name = os.path.basename(parsed.path) or "download"
    local_path = os.path.join(dest_dir, name)
    with urllib.request.urlopen(url, timeout=120) as response, open(local_path, "wb") as fh:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            fh.write(block)
    return local_path


def _resolve_inputs(inp: dict, tmp_dir: str) -> tuple[str, list[str]]:
    """Resolve video and target inputs to local file paths."""
    video = inp.get("video_path")
    if not video:
        video_url = inp.get("video_url")
        if not video_url:
            raise ValueError("Provide either 'video_path' or 'video_url'.")
        video = _download(video_url, tmp_dir)

    targets = inp.get("target_paths")
    if not targets:
        target_urls = inp.get("target_urls")
        if not target_urls and not _discover_people(inp):
            raise ValueError("Provide either 'target_paths' or 'target_urls'.")
        targets = [_download(url, tmp_dir) for url in target_urls]

    if isinstance(targets, str):
        targets = [targets]

    return video, targets


def _db_connect():
    if not RDS_PASSWORD:
        raise RuntimeError("RDS_PASSWORD is required to store video detection results.")
    return psycopg2.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        dbname=RDS_DB,
        user=RDS_USER,
        password=RDS_PASSWORD,
        sslmode=RDS_SSLMODE,
        cursor_factory=psycopg2.extras.RealDictCursor,
        application_name=os.getenv("DB_APPLICATION_NAME", "face_occurrence_worker"),
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
    )


def _uuid_or_none(value) -> str | None:
    if not value:
        return None
    return str(uuid.UUID(str(value)))


def _text_or_none(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_value(value):
    return psycopg2.extras.Json(value if value is not None else {})


def _bool_input(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _discover_people(inp: dict) -> bool:
    for key in ("discover_people", "discoverPeople", "include_unknown_people", "includeUnknownPeople"):
        if key in inp:
            return _bool_input(inp.get(key), default=True)
    return True


def _target_s3_keys(inp: dict) -> list[str] | None:
    value = inp.get("target_s3_keys") or inp.get("targetS3Keys")
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return None


def _selected_person_ids(inp: dict) -> list[str]:
    value = inp.get("selected_person_ids") or inp.get("selectedPersonIds")
    if not isinstance(value, list):
        return []
    return [_uuid_or_none(item) for item in value if item]


def _target_person_ids(inp: dict) -> list[str | None]:
    value = inp.get("target_person_ids") or inp.get("targetPersonIds")
    if not isinstance(value, list):
        return []
    return [_uuid_or_none(item) if item else None for item in value]


def _detection_params(inp: dict) -> dict:
    return {
        "fps": float(inp.get("fps", DEFAULT_FPS)),
        "chunks": int(inp.get("chunks", DEFAULT_CHUNKS)),
        "parallel_chunks": int(inp.get("parallel_chunks", DEFAULT_PARALLEL_CHUNKS)),
        "similarity_threshold": float(inp.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)),
        "merge_gap_sec": float(inp.get("merge_gap_sec", DEFAULT_MERGE_GAP_SEC)),
        "min_interval_sec": float(inp.get("min_interval_sec", DEFAULT_MIN_INTERVAL_SEC)),
        "verify_model": inp.get("verify_model"),
        "verify_max_tokens": int(inp.get("verify_max_tokens", DEFAULT_VERIFY_MAX_TOKENS)),
        "selected_person_ids": _selected_person_ids(inp),
        "target_person_ids": _target_person_ids(inp),
        "discover_people": _discover_people(inp),
    }


def _should_persist(inp: dict) -> bool:
    if inp.get("persist_results") is True:
        return True
    return any(inp.get(key) for key in _METADATA_KEYS)


def _metadata(inp: dict, job: dict) -> dict:
    return {
        "video_id": _uuid_or_none(inp.get("video_id") or inp.get("videoId")),
        "album_id": _uuid_or_none(inp.get("album_id") or inp.get("albumId")),
        "album_event_id": _uuid_or_none(inp.get("album_event_id") or inp.get("event_id") or inp.get("albumEventId") or inp.get("eventId")),
        "customer_id": _uuid_or_none(inp.get("customer_id") or inp.get("customerId")),
        "target_person_id": _uuid_or_none(inp.get("target_person_id") or inp.get("person_id") or inp.get("targetPersonId") or inp.get("personId")),
        "file_name": _text_or_none(inp.get("file_name") or inp.get("fileName")) or os.path.basename(urllib.parse.urlparse(str(inp.get("video_url") or "")).path),
        "original_s3_key": _text_or_none(inp.get("original_s3_key") or inp.get("originalS3Key")),
        "storage_album_slug": _text_or_none(inp.get("storage_album_slug") or inp.get("album_slug") or inp.get("storageAlbumSlug") or inp.get("albumSlug")),
        "storage_event_slug": _text_or_none(inp.get("storage_event_slug") or inp.get("event_slug") or inp.get("storageEventSlug") or inp.get("eventSlug")),
        "target_s3_keys": _target_s3_keys(inp),
        "runpod_endpoint_id": _text_or_none(inp.get("runpod_endpoint_id") or inp.get("runpodEndpointId")),
        "runpod_job_id": _text_or_none(inp.get("runpod_job_id") or inp.get("runpodJobId") or job.get("id")),
    }


def _album_faces_prefix(meta: dict) -> str | None:
    album_slug = meta.get("storage_album_slug")
    if not album_slug:
        original_key = meta.get("original_s3_key") or ""
        parts = original_key.split("/")
        if len(parts) >= 2 and parts[0] == "albums":
            album_slug = parts[1]
    if not album_slug:
        return None
    return f"albums/{album_slug}/faces"


def _publish_unknown_face_thumbnails(result: dict, inp: dict, job: dict) -> None:
    people = result.get("discovered_people")
    if not isinstance(people, list):
        return

    discovery = result.setdefault("discovery", {})
    if not S3_BUCKET:
        discovery["unknown_thumbnail_uploads"] = 0
        discovery["unknown_thumbnail_upload_error"] = "S3_BUCKET is not configured"
        for person in people:
            if isinstance(person, dict):
                person.pop("thumbnail_path", None)
        return

    meta = _metadata(inp, job)
    video_id = meta.get("video_id") or str(uuid.uuid4())
    faces_prefix = _album_faces_prefix(meta)
    if not faces_prefix:
        discovery["unknown_thumbnail_uploads"] = 0
        discovery["unknown_thumbnail_upload_error"] = "storage_album_slug is not configured"
        for person in people:
            if isinstance(person, dict):
                person.pop("thumbnail_path", None)
        return

    s3 = boto3.client("s3", region_name=S3_REGION)
    uploaded = 0
    for index, person in enumerate(people):
        if not isinstance(person, dict):
            continue
        thumbnail_path = person.pop("thumbnail_path", None)
        if not thumbnail_path or not os.path.exists(thumbnail_path):
            continue
        key = f"{faces_prefix}/video-unknowns/{video_id}/unknown-{index + 1}.jpg"
        try:
            s3.upload_file(
                thumbnail_path,
                S3_BUCKET,
                key,
                ExtraArgs={
                    "ContentType": "image/jpeg",
                    "CacheControl": "public, max-age=31536000, immutable",
                },
            )
        except Exception as exc:
            discovery["unknown_thumbnail_upload_error"] = str(exc)
            continue
        person["thumbnail_s3_key"] = key
        uploaded += 1

    discovery["unknown_thumbnail_uploads"] = uploaded


def _mark_video_processing(inp: dict, job: dict) -> str | None:
    if not _should_persist(inp):
        return None

    meta = _metadata(inp, job)
    params = _detection_params(inp)

    with _db_connect() as conn:
        with conn.cursor() as cur:
            if meta["video_id"]:
                cur.execute(
                    """
                    UPDATE videos
                    SET album_id = COALESCE(%s::uuid, album_id),
                        album_event_id = COALESCE(%s::uuid, album_event_id),
                        customer_id = COALESCE(%s::uuid, customer_id),
                        file_name = COALESCE(%s, file_name),
                        original_s3_key = COALESCE(%s, original_s3_key),
                        storage_album_slug = COALESCE(%s, storage_album_slug),
                        storage_event_slug = COALESCE(%s, storage_event_slug),
                        detection_params = %s::jsonb,
                        target_person_id = COALESCE(%s::uuid, target_person_id),
                        target_s3_keys = COALESCE(%s::jsonb, target_s3_keys),
                        runpod_endpoint_id = COALESCE(%s, runpod_endpoint_id),
                        runpod_job_id = COALESCE(%s, runpod_job_id),
                        detection_status = 'processing',
                        detection_error = NULL,
                        updated_at = now()
                    WHERE id = %s::uuid
                    RETURNING id
                    """,
                    [
                        meta["album_id"],
                        meta["album_event_id"],
                        meta["customer_id"],
                        meta["file_name"],
                        meta["original_s3_key"],
                        meta["storage_album_slug"],
                        meta["storage_event_slug"],
                        _json_value(params),
                        meta["target_person_id"],
                        _json_value(meta["target_s3_keys"]) if meta["target_s3_keys"] is not None else None,
                        meta["runpod_endpoint_id"],
                        meta["runpod_job_id"],
                        meta["video_id"],
                    ],
                )
                row = cur.fetchone()
                if row:
                    return str(row["id"])

            cur.execute(
                """
                INSERT INTO videos(
                  id,
                  album_id,
                  album_event_id,
                  customer_id,
                  file_name,
                  original_s3_key,
                  storage_album_slug,
                  storage_event_slug,
                  detection_params,
                  target_person_id,
                  target_s3_keys,
                  runpod_endpoint_id,
                  runpod_job_id,
                  detection_status,
                  created_at,
                  updated_at
                )
                VALUES(
                  COALESCE(%s::uuid, gen_random_uuid()),
                  %s::uuid,
                  %s::uuid,
                  %s::uuid,
                  %s,
                  %s,
                  %s,
                  %s,
                  %s::jsonb,
                  %s::uuid,
                  %s::jsonb,
                  %s,
                  %s,
                  'processing',
                  now(),
                  now()
                )
                RETURNING id
                """,
                [
                    meta["video_id"],
                    meta["album_id"],
                    meta["album_event_id"],
                    meta["customer_id"],
                    meta["file_name"],
                    meta["original_s3_key"],
                    meta["storage_album_slug"],
                    meta["storage_event_slug"],
                    _json_value(params),
                    meta["target_person_id"],
                    _json_value(meta["target_s3_keys"] or []),
                    meta["runpod_endpoint_id"],
                    meta["runpod_job_id"],
                ],
            )
            row = cur.fetchone()
            return str(row["id"])


def _has_video_match_target_columns(cur) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) = 2 AS has_columns
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'video_face_matches'
          AND column_name IN ('target_index', 'target_s3_key')
        """
    )
    row = cur.fetchone()
    return bool(row and row["has_columns"])


def _store_detection_result(video_id: str, inp: dict, job: dict, result: dict) -> None:
    meta = _metadata(inp, job)
    params = _detection_params(inp)
    matches = result.get("matches") or []
    target_s3_keys = meta["target_s3_keys"] or []
    target_person_ids = _target_person_ids(inp)

    def target_index_for(match: dict) -> int | None:
        value = match.get("target_index")
        return int(value) if isinstance(value, int) else None

    def target_s3_key_for(match: dict) -> str | None:
        target_index = target_index_for(match)
        if target_index is None or target_index < 0 or target_index >= len(target_s3_keys):
            return None
        return target_s3_keys[target_index]

    def person_id_for(match: dict) -> str | None:
        target_index = target_index_for(match)
        if target_index is not None and 0 <= target_index < len(target_person_ids):
            return target_person_ids[target_index]
        if target_index is not None:
            return None
        return meta["target_person_id"]

    with _db_connect() as conn:
        with conn.cursor() as cur:
            has_target_columns = _has_video_match_target_columns(cur)
            cur.execute(
                """
                UPDATE videos
                SET duration_sec = %s,
                    model = %s,
                    detection_params = %s::jsonb,
                    target_s3_keys = COALESCE(%s::jsonb, target_s3_keys),
                    runpod_endpoint_id = COALESCE(%s, runpod_endpoint_id),
                    runpod_job_id = COALESCE(%s, runpod_job_id),
                    detection_status = 'completed',
                    detection_error = NULL,
                    result_json = %s::jsonb,
                    match_count = %s,
                    updated_at = now(),
                    completed_at = now()
                WHERE id = %s::uuid
                """,
                [
                    result.get("duration_sec"),
                    result.get("model"),
                    _json_value(params),
                    _json_value(meta["target_s3_keys"]) if meta["target_s3_keys"] is not None else None,
                    meta["runpod_endpoint_id"],
                    meta["runpod_job_id"],
                    _json_value(result),
                    len(matches),
                    video_id,
                ],
            )
            cur.execute("DELETE FROM video_face_matches WHERE video_id = %s::uuid", [video_id])
            if has_target_columns:
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    INSERT INTO video_face_matches(
                      video_id,
                      album_id,
                      album_event_id,
                      person_id,
                      target_index,
                      target_s3_key,
                      start_sec,
                      end_sec,
                      start_time,
                      end_time,
                      max_similarity,
                      avg_similarity,
                      frames_matched,
                      verified
                    )
                    VALUES(%s::uuid, %s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            video_id,
                            meta["album_id"],
                            meta["album_event_id"],
                            person_id_for(match),
                            target_index_for(match),
                            target_s3_key_for(match),
                            match.get("start_sec"),
                            match.get("end_sec"),
                            match.get("start_time"),
                            match.get("end_time"),
                            match.get("max_similarity"),
                            match.get("avg_similarity"),
                            match.get("frames_matched"),
                            match.get("verified"),
                        )
                        for match in matches
                    ],
                )
            else:
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    INSERT INTO video_face_matches(
                      video_id,
                      album_id,
                      album_event_id,
                      person_id,
                      start_sec,
                      end_sec,
                      start_time,
                      end_time,
                      max_similarity,
                      avg_similarity,
                      frames_matched,
                      verified
                    )
                    VALUES(%s::uuid, %s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            video_id,
                            meta["album_id"],
                            meta["album_event_id"],
                            person_id_for(match),
                            match.get("start_sec"),
                            match.get("end_sec"),
                            match.get("start_time"),
                            match.get("end_time"),
                            match.get("max_similarity"),
                            match.get("avg_similarity"),
                            match.get("frames_matched"),
                            match.get("verified"),
                        )
                        for match in matches
                    ],
                )


def _mark_video_failed(video_id: str | None, error: str) -> None:
    if not video_id:
        return
    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE videos
                SET detection_status = 'failed',
                    detection_error = %s,
                    updated_at = now(),
                    completed_at = now()
                WHERE id = %s::uuid
                """,
                [error, video_id],
            )


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    video_id = None

    try:
        video_id = _mark_video_processing(inp, job)
        with tempfile.TemporaryDirectory() as tmp_dir:
            video, targets = _resolve_inputs(inp, tmp_dir)
            unknown_face_dir = os.path.join(tmp_dir, "unknown-faces")
            result = run_detection(
                video=video,
                targets=targets,
                matcher=_matcher,
                fps=float(inp.get("fps", DEFAULT_FPS)),
                chunks=int(inp.get("chunks", DEFAULT_CHUNKS)),
                parallel_chunks=int(inp.get("parallel_chunks", DEFAULT_PARALLEL_CHUNKS)),
                similarity_threshold=float(
                    inp.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
                ),
                merge_gap_sec=float(inp.get("merge_gap_sec", DEFAULT_MERGE_GAP_SEC)),
                min_interval_sec=float(
                    inp.get("min_interval_sec", DEFAULT_MIN_INTERVAL_SEC)
                ),
                verify_url=inp.get("verify_url"),
                verify_api_key=inp.get("verify_api_key"),
                verify_prompt=inp.get("verify_prompt", DEFAULT_VERIFY_PROMPT),
                verify_model=inp.get("verify_model"),
                verify_max_tokens=int(
                    inp.get("verify_max_tokens", DEFAULT_VERIFY_MAX_TOKENS)
                ),
                discover_people=_discover_people(inp),
                unknown_similarity_threshold=float(
                    inp.get("unknown_similarity_threshold", inp.get("unknownSimilarityThreshold", DEFAULT_SIMILARITY_THRESHOLD))
                ),
                unknown_face_dir=unknown_face_dir,
                progress=False,
            )
            _publish_unknown_face_thumbnails(result, inp, job)
            if video_id:
                _store_detection_result(video_id, inp, job, result)
                result["video_id"] = video_id
            return result
    except Exception as exc:
        try:
            _mark_video_failed(video_id, str(exc))
        except Exception as db_exc:
            return {"error": str(exc), "db_error": str(db_exc), "video_id": video_id}
        return {"error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
