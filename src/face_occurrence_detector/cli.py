"""
CLI entry point.

Usage:
    python -m face_occurrence_detector.cli \\
        --video ./input/video.mp4 \\
        --targets ./targets/front.jpg ./targets/side.jpg \\
        --output ./output/result.json
"""

from __future__ import annotations

import argparse
import sys

from .config import (
    DEFAULT_CHUNKS,
    DEFAULT_CTX_ID,
    DEFAULT_DEBUG_DIR,
    DEFAULT_DET_SIZE,
    DEFAULT_FPS,
    DEFAULT_MERGE_GAP_SEC,
    DEFAULT_MIN_INTERVAL_SEC,
    DEFAULT_PARALLEL_CHUNKS,
    DEFAULT_SAVE_DEBUG,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from .output import print_summary
from .pipeline import run_detection


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face_occurrence_detector",
        description="Detect target person face occurrences in a video using InsightFace buffalo_l.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--video", required=True, help="Path to input video file.")
    p.add_argument(
        "--targets",
        required=True,
        nargs="+",
        metavar="IMAGE",
        help="One or more reference images of the target person.",
    )
    p.add_argument("--output", required=True, help="Path to write the output JSON.")

    # Processing
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame sampling rate.")
    p.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS, help="Number of video chunks.")
    p.add_argument(
        "--parallel-chunks",
        type=int,
        choices=(1, 2, 4),
        default=DEFAULT_PARALLEL_CHUNKS,
        dest="parallel_chunks",
        help="Number of chunks processed in parallel.",
    )

    # Matching
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        dest="similarity_threshold",
        help="Cosine similarity threshold for a positive match.",
    )
    p.add_argument(
        "--merge-gap-sec",
        type=float,
        default=DEFAULT_MERGE_GAP_SEC,
        dest="merge_gap_sec",
        help="Merge nearby detections if the gap between them is smaller than this value.",
    )
    p.add_argument(
        "--min-interval-sec",
        type=float,
        default=DEFAULT_MIN_INTERVAL_SEC,
        dest="min_interval_sec",
        help="Drop intervals shorter than this duration.",
    )

    # Model / device
    p.add_argument(
        "--det-size",
        type=int,
        default=DEFAULT_DET_SIZE,
        dest="det_size",
        help="InsightFace detector input size (square).",
    )
    p.add_argument(
        "--ctx-id",
        type=int,
        default=DEFAULT_CTX_ID,
        dest="ctx_id",
        help="GPU device id. Use -1 for CPU-only fallback.",
    )

    # Debug
    p.add_argument(
        "--save-debug",
        action="store_true",
        default=DEFAULT_SAVE_DEBUG,
        dest="save_debug",
        help="Save sampled frame crops to the debug directory.",
    )
    p.add_argument(
        "--debug-dir",
        default=DEFAULT_DEBUG_DIR,
        dest="debug_dir",
        help="Directory for debug outputs.",
    )

    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if args.chunks <= 0:
        parser.error("--chunks must be greater than 0")
    if args.similarity_threshold < -1.0 or args.similarity_threshold > 1.0:
        parser.error("--similarity-threshold must be between -1.0 and 1.0")
    if args.merge_gap_sec < 0:
        parser.error("--merge-gap-sec must be greater than or equal to 0")
    if args.min_interval_sec < 0:
        parser.error("--min-interval-sec must be greater than or equal to 0")

    try:
        data = run_detection(
            video=args.video,
            targets=args.targets,
            output=args.output,
            fps=args.fps,
            chunks=args.chunks,
            parallel_chunks=args.parallel_chunks,
            similarity_threshold=args.similarity_threshold,
            merge_gap_sec=args.merge_gap_sec,
            min_interval_sec=args.min_interval_sec,
            det_size=args.det_size,
            ctx_id=args.ctx_id,
            progress=True,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Results written to: {args.output}")
    print_summary(data["matches"])


if __name__ == "__main__":
    main()
