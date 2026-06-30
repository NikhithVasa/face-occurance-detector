"""
Optional second-pass identity verification via an external vision-LLM endpoint.

After InsightFace produces candidate intervals, each interval's peak-similarity
frame is cropped to the matched face and sent — together with the reference
target image — to a RunPod serverless vLLM endpoint. The model is asked to
confirm whether the two images show the same person. Only intervals the model
confirms are kept.

The endpoint is called in RunPod-native form:

    POST <verify_url-as-/runsync>
    { "input": { "messages": [ {role: user, content: [text, image, image]} ],
                 "sampling_params": {...} } }

Authentication uses an ``Authorization: Bearer <api_key>`` header.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from typing import TYPE_CHECKING, Callable

from .types import FrameMatch, TimelineInterval

if TYPE_CHECKING:
    import numpy as np


def _to_runsync_url(url: str) -> str:
    """Normalize a RunPod endpoint URL to its blocking /runsync form."""
    url = url.strip().rstrip("/")
    if url.endswith("/runsync"):
        return url
    if url.endswith("/run"):
        return url[: -len("/run")] + "/runsync"
    return url + "/runsync"


def _read_frame_at(video_path: str, timestamp_sec: float):
    """Decode a single frame at the given timestamp, or None if unavailable."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _crop_face(frame: "np.ndarray", bbox: list[float], margin: float = 0.4):
    """Crop the bbox region (with margin) from frame, clamped to bounds."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    cx1 = max(0, int(x1 - box_w * margin))
    cy1 = max(0, int(y1 - box_h * margin))
    cx2 = min(width, int(x2 + box_w * margin))
    cy2 = min(height, int(y2 + box_h * margin))
    if cx2 <= cx1 or cy2 <= cy1:
        return frame
    return frame[cy1:cy2, cx1:cx2]


def _to_data_url(image: "np.ndarray") -> str | None:
    """JPEG-encode an image as a base64 data URL, or None on failure."""
    import cv2

    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _best_match_in_range(
    matches: list[FrameMatch], start_sec: float, end_sec: float
) -> FrameMatch | None:
    """Return the highest-similarity FrameMatch within [start_sec, end_sec]."""
    best: FrameMatch | None = None
    for match in matches:
        if start_sec - 1e-6 <= match.timestamp_sec <= end_sec + 1e-6:
            if best is None or match.similarity > best.similarity:
                best = match
    return best


def _post_json(url: str, api_key: str, payload: dict, timeout: float = 180.0) -> dict:
    """POST a JSON payload with a Bearer header and return the parsed response."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_text(response: object) -> str:
    """Recursively pull assistant text from a variety of vLLM response shapes."""
    texts: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, str):
            texts.append(node)
        elif isinstance(node, dict):
            message = node.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                texts.append(message["content"])
            for key in ("content", "text"):
                if isinstance(node.get(key), str):
                    texts.append(node[key])
            tokens = node.get("tokens")
            if isinstance(tokens, list):
                texts.append("".join(t for t in tokens if isinstance(t, str)))
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    container = response.get("output", response) if isinstance(response, dict) else response
    walk(container)
    return " ".join(t for t in texts if t).strip()


def _decision_is_yes(text: str) -> bool:
    """True if the model's answer starts with an affirmative YES."""
    if not text:
        return False
    match = re.search(r"\b(yes|no)\b", text.strip().lower())
    return bool(match and match.group(1) == "yes")


def verify_intervals(
    *,
    video_path: str,
    intervals: list[TimelineInterval],
    matches: list[FrameMatch],
    target_image_path: str,
    verify_url: str,
    api_key: str,
    prompt: str,
    model: str | None = None,
    max_tokens: int = 64,
    log: Callable[[str], None] = print,
) -> list[TimelineInterval]:
    """
    Return only the intervals an external vision-LLM confirms as the same person.

    For each interval the peak-similarity frame is cropped to the matched face
    and compared against ``target_image_path``. Intervals whose verification call
    fails or is rejected are dropped (the caller asked to keep only confirmed
    matches).
    """
    import cv2

    if not api_key:
        raise ValueError(
            "Verification requested but no API key provided (verify_api_key)."
        )

    url = _to_runsync_url(verify_url)

    reference = cv2.imread(target_image_path)
    if reference is None:
        raise ValueError(f"Could not read reference image for verification: {target_image_path}")
    reference_url = _to_data_url(reference)
    if reference_url is None:
        raise ValueError("Could not encode reference image for verification.")

    confirmed: list[TimelineInterval] = []
    for interval in intervals:
        best = _best_match_in_range(matches, interval.start_sec, interval.end_sec)
        timestamp = best.timestamp_sec if best else (interval.start_sec + interval.end_sec) / 2.0

        frame = _read_frame_at(video_path, timestamp)
        if frame is None:
            log(f"  verify: no frame at {timestamp:.2f}s; dropping interval")
            continue

        candidate = _crop_face(frame, best.bbox) if best else frame
        candidate_url = _to_data_url(candidate)
        if candidate_url is None:
            log(f"  verify: could not encode frame at {timestamp:.2f}s; dropping interval")
            continue

        payload: dict = {
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": reference_url}},
                            {"type": "image_url", "image_url": {"url": candidate_url}},
                        ],
                    }
                ],
                "sampling_params": {"max_tokens": max_tokens, "temperature": 0},
            }
        }
        if model:
            payload["input"]["model"] = model

        try:
            response = _post_json(url, api_key, payload)
        except Exception as exc:  # noqa: BLE001 - any failure means unverified
            log(f"  verify: request failed at {timestamp:.2f}s ({exc}); dropping interval")
            continue

        text = _extract_text(response)
        if _decision_is_yes(text):
            confirmed.append(interval)
            log(f"  verify: CONFIRMED {interval.start_sec:.2f}-{interval.end_sec:.2f}s")
        else:
            log(f"  verify: rejected {interval.start_sec:.2f}-{interval.end_sec:.2f}s ({text[:60]!r})")

    return confirmed
