from __future__ import annotations

from typing import TYPE_CHECKING

from .config import DEFAULT_MODEL_NAME, DEFAULT_CTX_ID, DEFAULT_DET_SIZE, DEFAULT_PROVIDERS
from .types import TargetEmbedding, FrameFace, FrameMatch

if TYPE_CHECKING:
    import numpy as np


def _resolve_providers(ctx_id: int, requested_providers: list[str]) -> list[str]:
    """Select ONNX Runtime providers and fail fast on unintended CPU fallback."""
    if ctx_id < 0:
        return ["CPUExecutionProvider"]

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available. Install a CUDA-compatible "
            "onnxruntime-gpu build, or run with --ctx-id -1 for CPU fallback."
        )

    providers = [provider for provider in requested_providers if provider in available]
    if "CUDAExecutionProvider" not in providers:
        providers.insert(0, "CUDAExecutionProvider")
    if "CPUExecutionProvider" not in providers and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


class InsightFaceMatcher:
    """Wraps InsightFace buffalo_l for face detection and embedding comparison."""

    @staticmethod
    def _crop_face_jpeg(frame: np.ndarray, bbox: list[float], margin: float = 0.35) -> bytes | None:
        import cv2

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        pad_x = box_width * margin
        pad_y = box_height * margin

        left = max(0, int(x1 - pad_x))
        top = max(0, int(y1 - pad_y))
        right = min(width, int(x2 + pad_x))
        bottom = min(height, int(y2 + pad_y))
        if right <= left or bottom <= top:
            return None

        crop = frame[top:bottom, left:right]
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return encoded.tobytes() if ok else None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        ctx_id: int = DEFAULT_CTX_ID,
        det_size: tuple[int, int] = (DEFAULT_DET_SIZE, DEFAULT_DET_SIZE),
        providers: list[str] | None = None,
    ) -> None:
        from insightface.app import FaceAnalysis

        if providers is None:
            providers = DEFAULT_PROVIDERS

        providers = _resolve_providers(ctx_id, providers)

        self.model_name = model_name
        self.providers = providers
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def build_target_embeddings(self, target_paths: list[str]) -> list[TargetEmbedding]:
        """
        Extract normalized face embeddings from one or more reference images.

        Raises ValueError if no face is detected in all reference images.
        When multiple faces appear in a single image, the largest face by bounding
        box area is used.
        """
        import cv2

        embeddings: list[TargetEmbedding] = []

        for target_index, path in enumerate(target_paths):
            image = cv2.imread(path)
            if image is None:
                print(f"Skipping unreadable reference image: {path}")
                continue

            faces = self.app.get(image)
            if not faces:
                print(f"Skipping reference image with no detected face: {path}")
                continue

            # Choose the largest face by bounding box area
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )

            embeddings.append(
                TargetEmbedding(
                    source_path=path,
                    target_index=target_index,
                    embedding=face.normed_embedding,
                    face_bbox=face.bbox.tolist(),
                )
            )

        if not embeddings:
            raise ValueError("No face detected in any reference image")

        return embeddings

    def match_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        target_embeddings: list[TargetEmbedding],
        threshold: float,
        chunk_id: int,
    ) -> list[FrameMatch]:
        """
        Detect faces in a single frame and return any that match the target embeddings
        above the given cosine similarity threshold.

        Uses max-over-targets scoring so that multiple reference images (front, side,
        angled) all contribute to the match decision.
        """
        matches: list[FrameMatch] = []

        for detected_face in self.detect_frame_faces(
            frame=frame,
            timestamp_sec=timestamp_sec,
            target_embeddings=target_embeddings,
            chunk_id=chunk_id,
        ):
            if detected_face.best_target_index is None or detected_face.best_similarity is None:
                continue
            score = detected_face.best_similarity

            if score >= threshold:
                matches.append(
                    FrameMatch(
                        timestamp_sec=timestamp_sec,
                        similarity=score,
                        target_index=detected_face.best_target_index,
                        bbox=detected_face.bbox,
                        chunk_id=chunk_id,
                    )
                )

        return matches

    def detect_frame_faces(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        target_embeddings: list[TargetEmbedding],
        chunk_id: int,
    ) -> list[FrameFace]:
        """Detect all faces in a frame and optionally score each against targets."""
        import numpy as np

        faces = self.app.get(frame)
        if not faces:
            return []

        detected: list[FrameFace] = []
        for face in faces:
            embedding: np.ndarray = face.normed_embedding
            best_target_index: int | None = None
            best_similarity: float | None = None

            if target_embeddings:
                best_target = max(
                    target_embeddings,
                    key=lambda target: float(np.dot(embedding, target.embedding)),
                )
                best_target_index = best_target.target_index
                best_similarity = float(np.dot(embedding, best_target.embedding))

            detected.append(
                FrameFace(
                    timestamp_sec=timestamp_sec,
                    embedding=embedding,
                    bbox=face.bbox.tolist(),
                    crop_jpeg=self._crop_face_jpeg(frame, face.bbox.tolist()),
                    chunk_id=chunk_id,
                    best_target_index=best_target_index,
                    best_similarity=best_similarity,
                )
            )

        return detected
