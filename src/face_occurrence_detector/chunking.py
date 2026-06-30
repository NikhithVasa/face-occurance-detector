def compute_chunks(
    duration_sec: float,
    chunk_count: int,
    overlap_sec: float = 1.0,
) -> list[tuple[float, float]]:
    """
    Split a video of duration_sec into chunk_count chunks, each extended by
    overlap_sec on the boundary shared with the next/previous chunk.

    Example for duration=600, chunk_count=4, overlap_sec=1:
        chunk 0: (  0, 151)
        chunk 1: (149, 301)
        chunk 2: (299, 451)
        chunk 3: (449, 600)

    Returns a list of (start_sec, end_sec) tuples using absolute timestamps.
    """
    if chunk_count <= 0:
        raise ValueError("chunk_count must be a positive integer")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    chunk_duration = duration_sec / chunk_count
    chunks: list[tuple[float, float]] = []

    for i in range(chunk_count):
        start = i * chunk_duration
        end = (i + 1) * chunk_duration

        # Extend into the previous chunk to avoid boundary misses
        if i > 0:
            start = max(0.0, start - overlap_sec)

        # Extend into the next chunk to avoid boundary misses
        if i < chunk_count - 1:
            end = min(duration_sec, end + overlap_sec)

        chunks.append((start, end))

    return chunks
