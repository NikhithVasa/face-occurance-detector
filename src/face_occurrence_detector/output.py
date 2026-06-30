import json
import os

from .types import TimelineInterval


def format_seconds(sec: float) -> str:
    """Return sec formatted as HH:MM:SS (truncates sub-seconds)."""
    total = int(sec)
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_output_dict(
    video_path: str,
    fps: float,
    model_name: str,
    similarity_threshold: float,
    target_count: int,
    duration_sec: float,
    intervals: list[TimelineInterval],
) -> dict:
    return {
        "video": os.path.basename(video_path),
        "fps": fps,
        "model": f"insightface/{model_name}",
        "similarity_threshold": similarity_threshold,
        "target_count": target_count,
        "duration_sec": round(duration_sec, 2),
        "matches": [
            {
                "start_sec": round(interval.start_sec, 2),
                "end_sec": round(interval.end_sec, 2),
                "start_time": format_seconds(interval.start_sec),
                "end_time": format_seconds(interval.end_sec),
                "max_similarity": round(interval.max_similarity, 4),
                "avg_similarity": round(interval.avg_similarity, 4),
                "frames_matched": interval.frames_matched,
            }
            for interval in intervals
        ],
    }


def write_output_json(output_path: str, data: dict) -> None:
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def print_summary(matches: list[dict]) -> None:
    """Print a readable CLI summary from the result dict's `matches` list."""
    if not matches:
        print("\nTarget person not found in video.")
        return

    count = len(matches)
    print(f"\nFound target person in {count} interval{'s' if count != 1 else ''}:\n")
    for i, match in enumerate(matches, 1):
        start = match["start_time"]
        end = match["end_time"]
        print(f"  {i}. {start} \u2192 {end}, confidence {match['max_similarity']:.2f}")
    print()
