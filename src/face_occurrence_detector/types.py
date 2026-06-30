from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class TargetEmbedding:
    source_path: str
    embedding: np.ndarray
    face_bbox: list[float]


@dataclass
class FrameMatch:
    timestamp_sec: float
    similarity: float
    bbox: list[float]
    chunk_id: int


@dataclass
class TimelineInterval:
    start_sec: float
    end_sec: float
    max_similarity: float
    avg_similarity: float
    frames_matched: int
