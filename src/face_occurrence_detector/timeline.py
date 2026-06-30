from .types import FrameMatch, TimelineInterval


def deduplicate_matches(matches: list[FrameMatch]) -> list[FrameMatch]:
    """
    Remove duplicate timestamps that arise from overlapping chunk windows.
    When the same timestamp appears in multiple chunks, keep the detection
    with the highest similarity score.
    """
    best: dict[float, FrameMatch] = {}
    for match in matches:
        key = round(match.timestamp_sec, 3)
        if key not in best or match.similarity > best[key].similarity:
            best[key] = match
    return list(best.values())


def merge_matches_into_intervals(
    matches: list[FrameMatch],
    fps: float,
    merge_gap_sec: float,
    min_interval_sec: float,
) -> list[TimelineInterval]:
    """
    Convert a flat list of frame-level matches into merged time intervals.

    Steps:
    1. Deduplicate timestamps from overlapping chunks.
    2. Sort by timestamp.
    3. Greedily merge consecutive matches whose gap is <= merge_gap_sec.
    4. Drop intervals shorter than min_interval_sec.
    """
    if not matches:
        return []

    matches = deduplicate_matches(matches)
    matches = sorted(matches, key=lambda m: m.timestamp_sec)

    frame_duration = 1.0 / fps

    intervals: list[TimelineInterval] = []
    current_start = matches[0].timestamp_sec
    current_end = matches[0].timestamp_sec + frame_duration
    current_sims = [matches[0].similarity]

    for match in matches[1:]:
        frame_end = match.timestamp_sec + frame_duration

        if match.timestamp_sec - current_end <= merge_gap_sec:
            # Extend current interval
            current_end = max(current_end, frame_end)
            current_sims.append(match.similarity)
        else:
            # Finalise current interval if long enough
            if current_end - current_start >= min_interval_sec:
                intervals.append(
                    TimelineInterval(
                        start_sec=current_start,
                        end_sec=current_end,
                        max_similarity=max(current_sims),
                        avg_similarity=sum(current_sims) / len(current_sims),
                        frames_matched=len(current_sims),
                    )
                )
            # Begin a new interval
            current_start = match.timestamp_sec
            current_end = frame_end
            current_sims = [match.similarity]

    # Finalise the last interval
    if current_end - current_start >= min_interval_sec:
        intervals.append(
            TimelineInterval(
                start_sec=current_start,
                end_sec=current_end,
                max_similarity=max(current_sims),
                avg_similarity=sum(current_sims) / len(current_sims),
                frames_matched=len(current_sims),
            )
        )

    return intervals
