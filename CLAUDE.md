# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Official implementation of **VSI: Visual–Subtitle Integration for Keyframe Selection** (CVPR 2026 Findings). A training-free, plug-and-play multimodal keyframe retrieval framework for long video understanding.

## Environment Setup

Run `bash scripts/setup_env.sh` — this handles the full stack (torch,
opencv-python, mmcv, mmdet, mmengine, mmyolo, YOLO-World, and VSI's own
`requirements.txt`) in an order that avoids a real version-conflict chain
(mmdet==3.0.0 requires mmcv<2.1.0, which only has prebuilt wheels for
torch2.0.1/cu118; YOLO-World's own pyproject.toml separately requires
torchvision>=0.16.2, which conflicts with that). See the "Verified
Environment / Known Issues" section of README.md for the full explanation
if you need to adapt this to a different CUDA/torch version.

Do **not** naively `pip install mmengine mmdet supervision` — see above.

## Entry Points

| Script | Purpose |
|---|---|
| `VSI_keyframe_search.py` | **Main pipeline** — batch keyframe retrieval with VSI dual-branch fusion |
| `Keyframe_Matching.py` | Keyframe evaluation / matching utilities |
| `examples/run_keyframe_search.py` | Single-video quick start example |
| `KFSBench/scripts/evaluation/eval_json.py` | Evaluate retrieval results from a JSON output file |
| `KFSBench/scripts/inference/gpt4_inference.py` | Run GPT-4o VideoQA on retrieved keyframes |

### Running the main VSI pipeline

```bash
python VSI_keyframe_search.py \
  --input_json  /path/to/data.json \        # pre-extracted grounding_objects required
  --output_json /path/to/results.json \
  --config_path /path/to/yolo_world.py \
  --checkpoint_path /path/to/yolo_world.pth \
  --dataset LongVideoBench \                # or VideoMME
  --subtitle_root /path/to/subtitles/ \     # omit to run visual-only
  --text_weight 0.3 \
  --search_nframes 8 \
  --device cuda:0
```

## Architecture

### Dual-branch search (VSI)

```
Query ──┬── Video Search Branch  (YOLO-World object detection on frame grids)
        │                               │
        └── Subtitle Match Branch       │  Score Fusion (Z-score norm + adaptive weight)
            (sentence-transformers      │
             cosine similarity)         │
                                        ▼
                           Updated frame probability distribution
                           → iterate until search_budget exhausted
                           → return top-k keyframes
```

### Key classes in `VSI_keyframe_search.py`

| Class | Role |
|---|---|
| `VideoSearchConfig` | All hyperparameters as a dataclass (text_weight, grid shape, budget…) |
| `SubtitleProcessor` | Loads `.json` (LVBench) or `.srt` (VideoMME) subtitle files |
| `TextSimilarityCalculator` | Encodes query+subtitle with `all-mpnet-base-v2`, soft-threshold boost, Gaussian temporal spread |
| `ScoreFusionManager` | Z-score normalises visual and text scores, then blends with `text_weight` |
| `MultimodalTStarFramework` | Orchestrates one video: grounding → subtitle scoring → fused distribution → search |
| `VideoProcessor` | Batch wrapper; iterates dataset JSON and calls `MultimodalTStarFramework` |

### TStar base layer (`TStar/`)

| File | Role |
|---|---|
| `interface_searcher.py` | `TStarSearcher` — iterative frame sampling, spline/Gaussian distribution updates |
| `interface_yolo.py` | `YoloWorldInterface` — YOLO-World detection wrapper (MMDetection) |
| `interface_llm.py` | `TStarUniversalGrounder` — GPT-4o grounding: extracts target/cue objects from query |
| `utilites.py` | Video frame loading, base64 encoding, GIF saving |

`TStarFramework` in `TStar/TStarFramework.py` is the **visual-only** baseline (no subtitle branch). `MultimodalTStarFramework` in `VSI_keyframe_search.py` extends it with subtitle fusion by monkey-patching `update_frame_distribution`.

### KFSBench (`KFSBench/`)

Evaluation suite for the keyframe search benchmark:
- `src/evaluation/metrics.py` — valid-search-ratio and temporal-coverage metrics
- `scripts/evaluation/eval_json.py` — scores a JSON result file
- `scripts/evaluation/search_frames.py` — runs retrieval over the full benchmark
- `scripts/data_processing/` — dataset preparation (annotation conversion, frame extraction)

## Critical hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `text_weight` | 0.3 | Weight of subtitle branch in score fusion (0 = visual-only) |
| `search_nframes` | 8 | Number of keyframes returned |
| `grid_rows/cols` | 4×4 | Frames packed into one YOLO inference grid; higher = more context, more VRAM |
| `confidence_threshold` | 0.7 | YOLO detection threshold; lower = more sensitive |
| `search_budget` | 1.0 | Fraction of total frames the searcher may visit |

## Input JSON format

`VSI_keyframe_search.py` expects a JSON list where each item has:
```json
{
  "video_id": "xxx",
  "video_path": "/path/to/video.mp4",
  "question": "...",
  "options": "A) ...\nB) ...",
  "grounding_objects": {
    "target_objects": ["person", "cup"],
    "cue_objects": ["table"],
    "relations": []
  }
}
```
`grounding_objects` must be pre-extracted (e.g., via `TStarUniversalGrounder`); the main pipeline skips the LLM grounding step.
