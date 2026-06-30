from __future__ import annotations

from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def get_video_duration(video_path: str) -> float:
    """Return the duration of a video file in seconds using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS reported by video: {video_path}")

    return frame_count / fps


def sample_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """
    Yield (timestamp_sec, frame) pairs sampled at sample_fps between
    start_sec and end_sec.

    The reported timestamp is the *actual* decoded position of the frame, not
    the requested sample time.  We seek once to the chunk start and then decode
    forward sequentially: frames are cheaply skipped with grab() and only the
    sampled frames are decoded with retrieve().  This avoids the per-frame
    keyframe seeking of CAP_PROP_POS_MSEC (which is slow and snaps the returned
    frame to the nearest keyframe, mislabeling timestamps) and is accurate for
    variable-FPS sources.
    """
    import cv2
    import math

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        interval_sec = 1.0 / sample_fps
        nominal_fps = cap.get(cv2.CAP_PROP_FPS)

        # Seek once to the chunk start. Decoding then proceeds sequentially.
        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

        # Align sampling to a global grid (multiples of interval_sec measured from
        # t=0) rather than the chunk start. Overlapping chunks therefore sample the
        # same absolute timestamps, so duplicate detections deduplicate cleanly.
        next_sample_sec = math.ceil(start_sec * sample_fps - 1e-9) / sample_fps

        while True:
            if not cap.grab():
                break

            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec and pos_msec > 0:
                pos_sec = pos_msec / 1000.0
            elif nominal_fps and nominal_fps > 0:
                # Fallback when the backend does not report POS_MSEC.
                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                pos_sec = max(0.0, (frame_idx - 1) / nominal_fps)
            else:
                break

            # Half-open interval [start_sec, end_sec): the boundary frame is
            # covered by the next chunk, so excluding it here avoids sampling
            # the same frame twice across overlapping chunks.
            if pos_sec >= end_sec:
                break

            if pos_sec + 1e-6 >= next_sample_sec:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                yield pos_sec, frame
                next_sample_sec += interval_sec
                # Skip ahead if real frame spacing exceeds the sample interval
                # (e.g. sparse keyframes or low source FPS).
                while next_sample_sec <= pos_sec:
                    next_sample_sec += interval_sec
    finally:
        cap.release()
