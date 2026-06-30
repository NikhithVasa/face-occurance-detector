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
      "min_interval_sec": 1.0
    }

Provide either the ``*_path`` form (files on a mounted network volume) or the
``*_url`` form (downloaded to a temp dir for the duration of the job).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.parse
import urllib.request

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
        if not target_urls:
            raise ValueError("Provide either 'target_paths' or 'target_urls'.")
        targets = [_download(url, tmp_dir) for url in target_urls]

    if isinstance(targets, str):
        targets = [targets]

    return video, targets


def handler(job: dict) -> dict:
    inp = job.get("input") or {}

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video, targets = _resolve_inputs(inp, tmp_dir)
            return run_detection(
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
                progress=False,
            )
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}


def _job_from_env() -> dict | None:
    """Build a job dict from environment variables, or None if not in one-shot mode.

    One-shot mode is opt-in via RUN_ONESHOT (1/true/yes). This keeps a plain Pod
    able to process a video without a queue (set RUN_ONESHOT=1 + VIDEO_URL +
    TARGET_URLS and start it), while ensuring a Serverless Endpoint is never
    accidentally hijacked just because VIDEO_URL happens to be set.

        RUN_ONESHOT            "1"/"true"/"yes" to enable one-shot mode
        VIDEO_URL              single http(s) URL
        TARGET_URLS            one or more http(s) URLs, separated by commas,
                               whitespace, or newlines
        FPS, CHUNKS, PARALLEL_CHUNKS, SIMILARITY_THRESHOLD,
        MERGE_GAP_SEC, MIN_INTERVAL_SEC   optional numeric overrides
    """
    if os.getenv("RUN_ONESHOT", "").strip().lower() not in ("1", "true", "yes"):
        return None

    video_url = os.getenv("VIDEO_URL")
    if not video_url:
        return None

    raw_targets = os.getenv("TARGET_URLS", "")
    targets = [t for t in raw_targets.replace(",", " ").split() if t]

    inp: dict = {"video_url": video_url, "target_urls": targets}
    for env_key, job_key in (
        ("FPS", "fps"),
        ("CHUNKS", "chunks"),
        ("PARALLEL_CHUNKS", "parallel_chunks"),
        ("SIMILARITY_THRESHOLD", "similarity_threshold"),
        ("MERGE_GAP_SEC", "merge_gap_sec"),
        ("MIN_INTERVAL_SEC", "min_interval_sec"),
    ):
        value = os.getenv(env_key)
        if value is not None and value != "":
            inp[job_key] = value

    return {"input": inp, "id": "env_job"}


if __name__ == "__main__":
    env_job = _job_from_env()
    if env_job is not None:
        # One-shot mode: run the env-configured job, print the result, then idle
        # so the container stays up instead of exiting and being restarted in a
        # loop. To process a new video, change the env vars and restart the pod.
        result = handler(env_job)
        print("RESULT_JSON_BEGIN")
        print(json.dumps(result, indent=2))
        print("RESULT_JSON_END")
        print("Job complete. Idling; change env vars and restart for a new video.")
        while True:
            time.sleep(3600)
    else:
        runpod.serverless.start({"handler": handler})
