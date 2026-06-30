# face-occurance-detector

Detect when a target person's face appears in a video, returning precise timestamp intervals.

Uses [InsightFace `buffalo_l`](https://github.com/deepinsight/insightface) for face detection, alignment, and recognition embeddings.
The same model is used for both reference images and video frames — embeddings are never mixed across models.

---

## Features

- Multiple reference images per target (front, side, angled) for robust matching
- Configurable FPS sampling — default 1 FPS
- Video split into overlapping chunks processed in parallel
- Outputs JSON with timestamp intervals and confidence scores
- GPU-accelerated via CUDA (ONNX Runtime)

---

## Hardware

Primary target: **RTX 4090 24 GB**  
Secondary: RTX 5090 32 GB  
Fallback: RTX A5000 24 GB

The CLI fails fast if `--ctx-id` is `0` or higher and ONNX Runtime cannot see `CUDAExecutionProvider`. Use `--ctx-id -1` only when you intentionally want CPU fallback.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Or install the package in editable mode (enables `python -m face_occurrence_detector.cli`):

```bash
pip install -e .
```

---

## Usage

```bash
python -m face_occurrence_detector.cli \
  --video ./examples/videos/sample.mp4 \
  --targets ./examples/targets/person_front.jpg ./examples/targets/person_side.jpg \
  --output ./output/result.json \
  --fps 1 \
  --chunks 4 \
  --parallel-chunks 2
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to input video file |
| `--targets` | *(required)* | One or more reference images of the target person |
| `--output` | *(required)* | Path to write output JSON |
| `--fps` | `1` | Frame sampling rate |
| `--chunks` | `4` | Number of video chunks |
| `--parallel-chunks` | `2` | Parallel chunk workers |
| `--similarity-threshold` | `0.55` | Cosine similarity threshold |
| `--merge-gap-sec` | `1.5` | Merge detections within this gap (seconds) |
| `--min-interval-sec` | `1.0` | Drop intervals shorter than this (seconds) |
| `--det-size` | `640` | InsightFace detector input size |
| `--ctx-id` | `0` | GPU device id (`-1` for CPU fallback) |
| `--save-debug` | `false` | Save debug images |
| `--debug-dir` | `./debug` | Directory for debug output |

---

## Output

### JSON (`--output`)

```json
{
  "video": "video.mp4",
  "fps": 1,
  "model": "insightface/buffalo_l",
  "similarity_threshold": 0.55,
  "target_count": 2,
  "duration_sec": 600.0,
  "matches": [
    {
      "start_sec": 14.0,
      "end_sec": 18.0,
      "start_time": "00:00:14",
      "end_time": "00:00:18",
      "max_similarity": 0.82,
      "avg_similarity": 0.76,
      "frames_matched": 5
    }
  ]
}
```

### CLI summary

```
Found target person in 2 intervals:

  1. 00:00:14 → 00:00:18, confidence 0.82
  2. 00:03:41 → 00:03:49, confidence 0.79
```

---

## Notes

- **Use multiple reference images** — providing front, side, and angled shots significantly improves detection accuracy.
- **Same model for all embeddings** — do not mix InsightFace embeddings with embeddings from other models (CLIP, SigLIP, Qwen, etc.).
- **1 FPS may miss brief appearances** — if the person appears for less than 1 second it may not be captured. Use a higher `--fps` value for finer granularity.
- **Parallel chunks and GPU** — running 4 parallel chunks on a single GPU shares CUDA resources. Start with `--parallel-chunks 2`, benchmark, then try `4`.
- **Maximizing GPU usage** — on an RTX 4090, 1 FPS may underuse the GPU because video decode and per-frame overhead dominate. For better utilization, benchmark `--fps 5` or `--fps 10` with `--parallel-chunks 4` and monitor memory/utilization with `nvidia-smi`.
- **Future two-pass scanning** — the design supports a future second pass at higher FPS (e.g. 5–10 FPS) over candidate intervals for improved timestamp accuracy.

### GPU monitoring

```bash
# Monitor GPU utilisation during a run
watch -n 1 nvidia-smi
```

### Licensing

InsightFace `buffalo_l` is free for non-commercial use. For commercial deployments, review the [InsightFace license](https://github.com/deepinsight/insightface/blob/master/LICENSE) before production use.

---

## Docker / RunPod

The image is based on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, which bundles a CUDA 12.4 + cuDNN 9 stack matching the `onnxruntime-gpu` CUDA 12 build. The `buffalo_l` model is **baked into the image at build time**, so workers need no network access at runtime.

### Build and push

```bash
# Build (replace with your registry/namespace)
docker build -t <registry>/face-occurrence-detector:latest .

# Push
docker login
docker push <registry>/face-occurrence-detector:latest
```

The build fails fast if `CUDAExecutionProvider` is not present in the `onnxruntime` build, surfacing CUDA/cuDNN mismatches early.

### RunPod Serverless (default entrypoint)

The default `CMD` runs [handler.py](handler.py), a RunPod serverless worker. The model loads once at cold start and is reused across jobs. Set `CTX_ID=-1` to force CPU; the default is GPU `0`.

Job input (provide either the `*_path` form for files on a mounted network volume, or the `*_url` form to download):

```json
{
  "input": {
    "video_url": "https://example.com/video.mp4",
    "target_urls": [
      "https://example.com/front.jpg",
      "https://example.com/side.jpg"
    ],
    "fps": 1,
    "chunks": 4,
    "parallel_chunks": 2,
    "similarity_threshold": 0.55
  }
}
```

The handler returns the same result object documented under [Output](#output) (or `{"error": "..."}` on bad input).

### RunPod Pod (interactive CLI)

In a GPU Pod, override the command to run the CLI against files on the pod/volume:

```bash
docker run --gpus all --rm \
  -v /local/data:/workspace \
  <registry>/face-occurrence-detector:latest \
  python -m face_occurrence_detector.cli \
    --video /workspace/video.mp4 \
    --targets /workspace/front.jpg /workspace/side.jpg \
    --output /workspace/result.json \
    --fps 1 --chunks 4 --parallel-chunks 2
```

`--gpus all` requires the NVIDIA Container Toolkit on the host (RunPod provides this).

---

## Project structure

```
face-occurance-detector/
  README.md
  requirements.txt
  pyproject.toml
  Dockerfile
  .dockerignore
  handler.py                  # RunPod serverless entrypoint

  src/
    face_occurrence_detector/
      __init__.py
      cli.py                  # argument parsing + CLI summary
      pipeline.py             # shared run_detection() orchestration
      config.py               # default constants
      types.py                # TargetEmbedding, FrameMatch, TimelineInterval
      insightface_matcher.py  # model loading, embedding, matching
      video_reader.py         # duration query, frame sampling
      chunking.py             # chunk splitting with overlap
      timeline.py             # dedup, merge, filter intervals
      output.py               # JSON writing, CLI summary

  examples/
    targets/   # place reference images here
    videos/    # place input videos here

  output/      # JSON results written here
  debug/       # debug images when --save-debug is used
```
