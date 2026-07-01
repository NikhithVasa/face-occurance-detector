"""
End-to-end detection pipeline shared by the CLI and the RunPod handler.

`run_detection` builds (or reuses) an InsightFace matcher, samples and scans the
video in overlapping chunks, merges frame matches into intervals, and returns the
JSON-serializable result dict.  It contains no argument parsing or process exit
logic so it can be embedded in a CLI, a serverless handler, or an API.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .chunking import compute_chunks
from .config import (
    DEFAULT_CHUNKS,
    DEFAULT_CTX_ID,
    DEFAULT_DET_SIZE,
    DEFAULT_FPS,
    DEFAULT_MERGE_GAP_SEC,
    DEFAULT_MIN_INTERVAL_SEC,
    DEFAULT_MODEL_NAME,
    DEFAULT_OVERLAP_SEC,
    DEFAULT_PARALLEL_CHUNKS,
    DEFAULT_PROVIDERS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VERIFY_MAX_TOKENS,
    DEFAULT_VERIFY_PROMPT,
)
from .insightface_matcher import InsightFaceMatcher
from .output import build_output_dict, write_output_json
from .timeline import merge_matches_into_intervals
from .types import FrameFace, FrameMatch, TargetEmbedding
from .verification import verify_intervals
from .video_reader import get_video_duration, sample_frames


def _process_chunk(
    video_path: str,
    chunk_id: int,
    start_sec: float,
    end_sec: float,
    fps: float,
    matcher: InsightFaceMatcher,
    target_embeddings: list[TargetEmbedding],
    threshold: float,
    progress: bool,
) -> list[FrameMatch]:
    """Sample frames from one chunk and return all matching FrameMatch objects."""
    matches: list[FrameMatch] = []

    frames = sample_frames(video_path, start_sec, end_sec, fps)
    if progress:
        from tqdm import tqdm

        estimated = max(1, int((end_sec - start_sec) * fps))
        frames = tqdm(
            frames,
            total=estimated,
            desc=f"Chunk {chunk_id + 1}",
            unit="frame",
            leave=False,
        )

    for timestamp_sec, frame in frames:
        matches.extend(
            matcher.match_frame(
                frame=frame,
                timestamp_sec=timestamp_sec,
                target_embeddings=target_embeddings,
                threshold=threshold,
                chunk_id=chunk_id,
            )
        )

    return matches


def _process_chunk_faces(
    video_path: str,
    chunk_id: int,
    start_sec: float,
    end_sec: float,
    fps: float,
    matcher: InsightFaceMatcher,
    target_embeddings: list[TargetEmbedding],
    progress: bool,
) -> list[FrameFace]:
    """Sample one chunk and return every detected face with target scores."""
    faces: list[FrameFace] = []

    frames = sample_frames(video_path, start_sec, end_sec, fps)
    if progress:
        from tqdm import tqdm

        estimated = max(1, int((end_sec - start_sec) * fps))
        frames = tqdm(
            frames,
            total=estimated,
            desc=f"Chunk {chunk_id + 1}",
            unit="frame",
            leave=False,
        )

    for timestamp_sec, frame in frames:
        faces.extend(
            matcher.detect_frame_faces(
                frame=frame,
                timestamp_sec=timestamp_sec,
                target_embeddings=target_embeddings,
                chunk_id=chunk_id,
            )
        )

    return faces


def _cluster_unknown_faces(
    faces: list[FrameFace],
    start_target_index: int,
    threshold: float,
    unknown_face_dir: str | None = None,
) -> tuple[list[FrameMatch], list[dict]]:
    import numpy as np

    clusters: list[dict] = []
    matches: list[FrameMatch] = []

    for face in sorted(faces, key=lambda item: item.timestamp_sec):
        best_cluster: dict | None = None
        best_score = -1.0
        for cluster in clusters:
            score = float(np.dot(face.embedding, cluster["centroid"]))
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is None or best_score < threshold:
            target_index = start_target_index + len(clusters)
            best_cluster = {
                "target_index": target_index,
                "centroid": face.embedding.copy(),
                "count": 0,
                "thumbnail_jpeg": face.crop_jpeg,
                "thumbnail_area": max(0.0, (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
            }
            clusters.append(best_cluster)
            best_score = 1.0

        face_area = max(0.0, (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        if face.crop_jpeg and face_area > best_cluster.get("thumbnail_area", 0):
            best_cluster["thumbnail_jpeg"] = face.crop_jpeg
            best_cluster["thumbnail_area"] = face_area

        best_cluster["count"] += 1
        best_cluster["centroid"] = best_cluster["centroid"] + (
            face.embedding - best_cluster["centroid"]
        ) / best_cluster["count"]
        norm = np.linalg.norm(best_cluster["centroid"])
        if norm > 0:
            best_cluster["centroid"] = best_cluster["centroid"] / norm

        matches.append(
            FrameMatch(
                timestamp_sec=face.timestamp_sec,
                similarity=best_score,
                target_index=best_cluster["target_index"],
                bbox=face.bbox,
                chunk_id=face.chunk_id,
            )
        )

    discovered = []
    if unknown_face_dir:
        os.makedirs(unknown_face_dir, exist_ok=True)

    for index, cluster in enumerate(clusters):
        thumbnail_path = None
        thumbnail_jpeg = cluster.get("thumbnail_jpeg")
        if unknown_face_dir and thumbnail_jpeg:
            thumbnail_path = os.path.join(unknown_face_dir, f"unknown-{index + 1}.jpg")
            with open(thumbnail_path, "wb") as file:
                file.write(thumbnail_jpeg)

        person = {
            "target_index": cluster["target_index"],
            "label": f"Unknown person {index + 1}",
            "known": False,
            "frames_matched": cluster["count"],
        }
        if thumbnail_path:
            person["thumbnail_path"] = thumbnail_path
        discovered.append(person)

    return matches, discovered


def run_detection(
    *,
    video: str,
    targets: list[str],
    output: str | None = None,
    fps: float = DEFAULT_FPS,
    chunks: int = DEFAULT_CHUNKS,
    parallel_chunks: int = DEFAULT_PARALLEL_CHUNKS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    min_interval_sec: float = DEFAULT_MIN_INTERVAL_SEC,
    det_size: int = DEFAULT_DET_SIZE,
    ctx_id: int = DEFAULT_CTX_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    providers: list[str] | None = None,
    matcher: InsightFaceMatcher | None = None,
    verify_url: str | None = None,
    verify_api_key: str | None = None,
    verify_prompt: str = DEFAULT_VERIFY_PROMPT,
    verify_model: str | None = None,
    verify_max_tokens: int = DEFAULT_VERIFY_MAX_TOKENS,
    discover_people: bool = False,
    unknown_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    unknown_face_dir: str | None = None,
    progress: bool = True,
) -> dict:
    """
    Run the full detection pipeline and return the result dict.

    Pass a prebuilt `matcher` to reuse a loaded model across calls (e.g. a
    serverless worker that loads once at cold start).  Set `progress=False` to
    suppress stdout logging and progress bars.  If `output` is given, the result
    is also written to that path as JSON.

    Raises ValueError for unreadable inputs or missing faces, and RuntimeError if
    a GPU was requested but CUDA is unavailable.
    """
    if providers is None:
        providers = DEFAULT_PROVIDERS

    def log(message: str) -> None:
        if progress:
            print(message)

    # 1. Load model (unless one was supplied).
    if matcher is None:
        log(f"Loading InsightFace model '{model_name}'...")
        matcher = InsightFaceMatcher(
            model_name=model_name,
            ctx_id=ctx_id,
            det_size=(det_size, det_size),
            providers=providers,
        )
    log(f"  ONNX Runtime providers: {', '.join(matcher.providers)}")

    # 2. Build target embeddings.
    log(f"Building target embeddings from {len(targets)} image(s)...")
    target_embeddings = matcher.build_target_embeddings(targets)
    log(f"  Built {len(target_embeddings)} embedding(s).")

    # 3. Inspect video.
    log(f"Inspecting video: {video}")
    duration_sec = get_video_duration(video)
    log(f"  Duration: {duration_sec:.1f}s")

    # 4. Compute chunks.
    chunk_windows = compute_chunks(duration_sec, chunks, overlap_sec=DEFAULT_OVERLAP_SEC)
    log(
        f"Processing {chunks} chunk(s) with {parallel_chunks} parallel "
        f"worker(s) at {fps} FPS..."
    )

    # 5. Process chunks (sequential or parallel).
    all_matches: list[FrameMatch] = []
    discovered_people: list[dict] = []
    discovery = {
        "enabled": discover_people,
        "detected_faces": None,
        "known_frame_matches": None,
        "unknown_face_detections": None,
        "unknown_frame_matches": None,
        "unknown_clusters": None,
        "unknown_similarity_threshold": unknown_similarity_threshold,
    }

    if discover_people:
        all_faces: list[FrameFace] = []
        if parallel_chunks <= 1:
            for chunk_id, (start_sec, end_sec) in enumerate(chunk_windows):
                all_faces.extend(
                    _process_chunk_faces(
                        video_path=video,
                        chunk_id=chunk_id,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        fps=fps,
                        matcher=matcher,
                        target_embeddings=target_embeddings,
                        progress=progress,
                    )
                )
        else:
            max_workers = min(parallel_chunks, len(chunk_windows))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _process_chunk_faces,
                        video,
                        chunk_id,
                        start_sec,
                        end_sec,
                        fps,
                        matcher,
                        target_embeddings,
                        progress,
                    )
                    for chunk_id, (start_sec, end_sec) in enumerate(chunk_windows)
                ]
                for future in as_completed(futures):
                    all_faces.extend(future.result())

        known_matches = [
            FrameMatch(
                timestamp_sec=face.timestamp_sec,
                similarity=face.best_similarity or 0,
                target_index=face.best_target_index or 0,
                bbox=face.bbox,
                chunk_id=face.chunk_id,
            )
            for face in all_faces
            if face.best_target_index is not None
            and face.best_similarity is not None
            and face.best_similarity >= similarity_threshold
        ]
        unknown_faces = [
            face
            for face in all_faces
            if face.best_similarity is None or face.best_similarity < similarity_threshold
        ]
        unknown_matches, discovered_people = _cluster_unknown_faces(
            faces=unknown_faces,
            start_target_index=len(target_embeddings),
            threshold=unknown_similarity_threshold,
            unknown_face_dir=unknown_face_dir,
        )
        all_matches = known_matches + unknown_matches
        discovery.update(
            {
                "detected_faces": len(all_faces),
                "known_frame_matches": len(known_matches),
                "unknown_face_detections": len(unknown_faces),
                "unknown_frame_matches": len(unknown_matches),
                "unknown_clusters": len(discovered_people),
            }
        )
        log(
            f"  Raw detected faces: {len(all_faces)} "
            f"({len(known_matches)} known, {len(unknown_matches)} unknown-clustered)."
        )
    elif parallel_chunks <= 1:
        for chunk_id, (start_sec, end_sec) in enumerate(chunk_windows):
            all_matches.extend(
                _process_chunk(
                    video_path=video,
                    chunk_id=chunk_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    fps=fps,
                    matcher=matcher,
                    target_embeddings=target_embeddings,
                    threshold=similarity_threshold,
                    progress=progress,
                )
            )
    else:
        # Each chunk uses its own VideoCapture (separate decode state). The
        # shared matcher dispatches to ONNX Runtime, whose session.run releases
        # the GIL, so chunk workers can overlap GPU work.
        max_workers = min(parallel_chunks, len(chunk_windows))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    video,
                    chunk_id,
                    start_sec,
                    end_sec,
                    fps,
                    matcher,
                    target_embeddings,
                    similarity_threshold,
                    progress,
                )
                for chunk_id, (start_sec, end_sec) in enumerate(chunk_windows)
            ]
            # .result() re-raises the first worker exception to the caller.
            for future in as_completed(futures):
                all_matches.extend(future.result())

    log(f"  Raw frame matches: {len(all_matches)}")

    # 6. Merge into intervals.
    intervals = merge_matches_into_intervals(
        matches=all_matches,
        fps=fps,
        merge_gap_sec=merge_gap_sec,
        min_interval_sec=min_interval_sec,
    )

    # 6b. Optional second-pass vision-LLM verification. Only intervals the
    # external model confirms as the same person are kept.
    if verify_url and intervals:
        if not targets:
            raise ValueError("Verification requires at least one target image.")
        log(f"Verifying {len(intervals)} interval(s) via vision-LLM...")
        intervals = verify_intervals(
            video_path=video,
            intervals=intervals,
            matches=all_matches,
            target_image_path=targets[0],
            verify_url=verify_url,
            api_key=verify_api_key or "",
            prompt=verify_prompt,
            model=verify_model,
            max_tokens=verify_max_tokens,
            log=log,
        )
        log(f"  Verified interval(s): {len(intervals)}")

    # 7. Build result and optionally write JSON.
    data = build_output_dict(
        video_path=video,
        fps=fps,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        target_count=len(target_embeddings),
        duration_sec=duration_sec,
        intervals=intervals,
        discovered_people=discovered_people,
        discovery=discovery,
    )

    if output:
        write_output_json(output, data)

    return data
