# VSI: Visual–Subtitle Integration for Keyframe Selection
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-orange.svg)](https://cvpr.thecvf.com/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://jacksonha7.github.io/VSI-page/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.06869-b31b1b.svg)](https://arxiv.org/abs/2508.06869)

Official implementation of the paper **VSI: Visual–Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding** (CVPR 2026 findings). A multimodal keyframe retrieval framework that fuses visual and subtitle information for long video understanding, achieving the best keyframe-search accuracy on LongVideoBench and improving downstream VideoQA on LongVideoBench and Video-MME.

🌐 **Project page:** https://jacksonha7.github.io/VSI-page/ · 📄 **Paper:** https://arxiv.org/abs/2508.06869

## 1. Project Introduction
### 1.1 Core Motivation
Existing keyframe selection algorithms rely solely on visual modality, leading to poor performance on text-related long video tasks and deviation from core semantic content. VSI addresses this issue via a **dual-branch collaborative retrieval mechanism** (Video Search + Subtitle Match) to fuse complementary visual and textual information.

### 1.2 Key Advantages
- **Multimodal Fusion**: Integrate visual object detection and subtitle semantic similarity for precise keyframe localization.
- **Plug-and-Play**: No additional training required, lightweight and flexible, can be integrated into existing video-LM pipelines.
- **Keyframe Search Accuracy**: 73.89% keyframe search accuracy on LongVideoBench (64 frames), ahead of T\* (67.58%) and VSLS (70.23%); on text-related tasks, search accuracy rises from 29.48% to 45.00% (8-frame setting).
- **Strong Generalization**: Improves downstream VideoQA; on the text-related subsets of LongVideoBench, GPT-4o accuracy on long videos rises from 53.76% (uniform sampling) to 69.57% with VSI-selected frames.

### 1.3 Main Tasks
- Keyframe retrieval for long video (3-60 minutes)
- Downstream VideoQA (visual/text-related sub-tasks)
- Multimodal long video understanding

## 2. Environment Configuration
### 2.1 Basic Dependencies (recommended: one-shot script)
Getting the YOLO-World detector adapter working involves a chain of real
version conflicts between YOLO-World's own `pyproject.toml`, mmdet's runtime
version check, and openmmlab's prebuilt wheel index (details in
[2.4](#24-️-verified-environment--known-issues)). Rather than rediscovering
these, run:
```bash
bash scripts/setup_env.sh
```
This creates a `vsi` conda env (Python 3.10), installs a verified-working
pinned combo (`constraints.txt`), clones/patches YOLO-World into
`third_party/YOLO-World`, and downloads the small LVIS metadata file that
`init_detector()` needs even for pure custom-vocabulary inference. It is
idempotent — safe to re-run if it fails partway through.

Override `CONDA_ENV_NAME`, `YOLO_WORLD_DIR`, or `PYTHON_VERSION` as
environment variables if you need different values.

### 2.1.1 Manual install (if you don't need the YOLO-World branch)
If you only need the subtitle-similarity branch (`--subtitle_root`, no
`--detector yoloworld`), a plain install is enough and none of the version
conflicts in 2.4 apply:
```bash
conda create -n vsi python=3.10
conda activate vsi
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2.2 Pre-trained Model Dependencies
VSI relies on the following pre-trained models (automatically downloaded or manual download links provided):
- **Video Search Branch**: YOLO-World-110M ([Official Repo](https://github.com/AILab-CVC/YOLO-World))
- **Subtitle Match Branch**: all-mpnet-base-v2 ([Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2))
- **Downstream VideoQA**: GPT-4o/LLaVA-Video-7B-Qwen2/Qwen2.5VL-7B-Instruct (follow official instructions for API/weight deployment)

### 2.3 Hardware Requirements
- GPU: NVIDIA A100/V100/3090/4090 (≥16G VRAM, recommended ≥24G for long video processing)
- CPU: ≥8 cores, ≥16G RAM
- Disk: ≥100G free space (for dataset and pre-trained weights)

### 2.4 ⚠️ Verified Environment / Known Issues
The exact combo below was verified end-to-end (env setup → checkpoint load →
keyframe search on a real video) on Ubuntu with CUDA 11.8. `scripts/setup_env.sh`
installs precisely this. If you hit an error not listed here, please open an
issue with the full traceback.

| Package | Version | Why it's pinned |
|---|---|---|
| `torch` / `torchvision` | `2.0.1` / `0.15.2` (cu118) | See mmcv row below |
| `mmcv` | `2.0.1` | `mmdet==3.0.0` asserts `mmcv<2.1.0` at import time, but openmmlab only publishes a prebuilt `mmcv<2.1.0` wheel for the `torch2.0.x`/`cu118` combo — not `torch2.1`/`cu121`. Using a newer torch here silently gets you an incompatible mmcv and a cryptic `AssertionError` on the very first import. |
| `mmdet` | `3.0.0` | Pinned by YOLO-World; see above. |
| `transformers` | `4.36.2` | Newer versions raise torch's minimum required version and re-trigger the conflict above. |
| `sentence-transformers` | `2.7.0` | Same reason. |
| `opencv-python` | `4.9.0.80` | See "opencv-python vs. opencv-python-headless" below. |
| `numpy` | `<2` (e.g. `1.26.4`) | torch 2.0.1 wheels are built against numpy 1.x; numpy 2.x imports but raises "Failed to initialize NumPy: _ARRAY_API not found" warnings and can misbehave. |
| `setuptools` | `<81` | See "pkg_resources" below. |
| `supervision` | `0.19.0` | Pinned by YOLO-World; older than what `TStar/interface_yolo.py` was originally written against (see "LabelAnnotator" below). |

Known issues, in the order you'll likely hit them from a naive install:

1. **`pkg_resources` missing while building `mmyolo`.** setuptools ≥81 no
   longer bundles `pkg_resources`, but torch's `cpp_extension.py` (imported
   while building mmyolo from source) still needs it, and pip's build
   isolation means the outer environment's torch install isn't visible to
   the isolated build. Fix: pin `setuptools<81`, and install YOLO-World/mmyolo
   with `pip install --no-build-isolation ...`.
2. **`mmdet==3.0.0` requires `mmcv<2.1.0`, but that wheel doesn't exist for
   torch2.1/cu121.** This is the root conflict described in the table above —
   it forces the whole stack down to torch2.0.1/cu118.
3. **YOLO-World's own `pyproject.toml` requires `torchvision>=0.16.2`**,
   which directly conflicts with the torch2.0.1-compatible combo above (this
   is an inconsistency in the upstream repo, not something we can pin
   around). Fix: install YOLO-World and its `third_party/mmyolo` submodule
   with `pip install --no-build-isolation --no-deps -e .`, then install the
   small remaining deps (`mmdet`, `supervision`, `prettytable`) explicitly.
   `--no-deps` also means `mmdet` isn't pulled in automatically — install it
   as its own step.
4. **`opencv-python` vs. `opencv-python-headless`.** `openmim`/`supervision`
   pull in `opencv-python-headless` as a transitive dependency. It and
   `opencv-python` both install into the same `cv2/` package directory and
   silently corrupt each other — `import cv2; cv2.__version__` ends up
   reporting whichever was installed last, and you get sporadic, hard-to-explain
   native crashes. Fix: never let both be installed; if you see both in
   `pip list`, `pip uninstall -y opencv-python-headless opencv-python` and
   reinstall only `opencv-python==4.9.0.80`.
5. **Upstream syntax bug in AILab-CVC/YOLO-World.**
   `yolo_world/models/detectors/yolo_world.py`'s `reparameterize()` has
   `self.text_feats, None = self.backbone.forward_text(texts)` — assigning to
   the literal `None` is a `SyntaxError` in Python and breaks importing
   `yolo_world` entirely. `scripts/setup_env.sh` patches this automatically
   (`None` → `_`); if you install YOLO-World some other way, apply the same
   one-line fix.
6. **Import order: `torch` must be imported before `cv2`/`decord`.** If
   `cv2`/`decord` get imported first and `torch`+CUDA is only touched later
   in the same process, you get a native crash on GPU nodes:
   `terminate called after throwing an instance of 'std::system_error' what():
   random_device could not be read`. This is an OpenMP/MKL runtime clash
   between the `opencv-python`/`decord` wheels and torch's bundled CUDA
   runtime, and only reproduces where a GPU is actually visible (a CPU-only
   sanity check will not catch it). `TStar/__init__.py` now imports `torch`
   first for this reason — if you add new modules to this package, keep that
   import first.
7. **`LabelAnnotator(..., smart_position=True)` fails on `supervision==0.19.0`.**
   `smart_position` was added in a later `supervision` release; the pinned
   0.19.0 (required by YOLO-World) doesn't have it. Already removed from
   `TStar/interface_yolo.py`; the visualization still works, just without
   smart label placement.
8. **`init_detector()` needs real LVIS files, even for custom-vocabulary
   inference.** The bundled YOLO-World `configs/pretrain/*lvis_minival*.py`
   configs point their `test_dataloader` at a real LVIS-minival annotation
   file and a small class-name JSON, and `mmdet`'s `init_detector()` builds
   that dataset just to read off `.metainfo` — even though VSI overrides the
   text prompts at runtime via `reparameterize_object_list()`. Both paths are
   resolved **relative to the working directory you run the script from**,
   not the config file's location. `scripts/setup_env.sh` downloads the
   ~35MB annotation JSON and symlinks YOLO-World's bundled class-text
   directory into `./data/` for you; if you hit
   `FileNotFoundError: data/coco/lvis/...` or
   `data/texts/lvis_v1_class_texts.json`, re-run it or recreate those two
   paths by hand.
9. **`ModuleNotFoundError: No module named 'TStar'` when running the example
   scripts directly.** `python examples/run_keyframe_search.py` puts
   `examples/` (not the repo root) on `sys.path`. Run from the repo root with
   the repo root on `PYTHONPATH`:
   ```bash
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   ```
10. **The YOLO-World README's own model-card table links the wrong
    checkpoint file for the S-size model** (it points "YOLO-World-S" at an
    `x_stage1-*.pth` file). Get the actual filename from the
    [`wondervictor/YOLO-World-V2.1`](https://huggingface.co/wondervictor/YOLO-World-V2.1)
    HuggingFace repo's file listing instead of trusting the table — the real
    S-size stage-1 checkpoint is `s_stage1-d1c1d7d8.pth`.
11. **`TStarSearcher.__init__() got an unexpected keyword argument
    'relations'`** when running `VSI_keyframe_search.py` (the subtitle-fusion
    path). `TStarSearcher` had its `relations` parameter removed, but
    `MultimodalTStarFramework._initialize_searcher` still passed it through.
    Fixed — already removed from the call site.
12. **`logging.info(...)` calls in the example/main scripts don't print
    anything.** `mmengine`/`mmdet` configure the root logger's handlers as a
    side effect of import, so the `logging.basicConfig(...)` call in
    `main()` (which is a no-op if the root logger already has handlers) never
    takes effect. The actual search results are still written correctly to
    the output JSON/`.npy` files — only the progress/summary log lines are
    silently dropped. If you need those messages, configure your own logger
    explicitly rather than relying on `logging.basicConfig`.

## 3. Data Preparation
### 3.1 Supported Datasets
We evaluate VSI on two mainstream long video understanding benchmarks:
1. **LongVideoBench** ([Download Link](https://huggingface.co/datasets/longvideobench/LongVideoBench)): Large-scale long video-language understanding benchmark (3-60 minutes videos)
2. **VideoMME** ([Download Link](https://github.com/OpenGVLab/VideoMME)): Multimodal video QA benchmark (4-60 minutes videos)

### 3.2 Data Preprocessing
1. Download the original dataset and unzip to `./data/`
2. LongVideoBench provides subtitle files (`.json`) alongside annotations — place them under `subtitles/`.
3. VideoMME provides subtitle files in `.srt` format — VSI loads them directly without additional conversion.

### 3.3 Dataset Directory Structure
```
./data/processed/
├── LongVideoBench/
│   ├── videos/          # Original video files (mp4/mkv)
│   ├── subtitles/       # Timestamp-aligned subtitles (.json per video)
│   └── annotations/     # Official QA/keyframe annotations
└── VideoMME/
    ├── videos/          # Original video files
    └── subtitles/       # Subtitle files (.srt per video)
```

## 4. Quick Start
Run every command below from the repo root, with the repo root on
`PYTHONPATH` (needed because `TStar` is imported by absolute package name —
see [2.4, item 9](#24-️-verified-environment--known-issues)):
```bash
conda activate vsi
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 4.1 Keyframe Retrieval (Single Video — Visual-Only)
For a quick test on a single video without subtitles, use the example script.
`--yoloworld_config`/`--yoloworld_ckpt` should point at one of the configs
under `third_party/YOLO-World/configs/pretrain/` (created by
`scripts/setup_env.sh`) and a matching checkpoint downloaded from
[`wondervictor/YOLO-World-V2.1`](https://huggingface.co/wondervictor/YOLO-World-V2.1)
(double-check the filename against that repo's own file listing — see
[2.4, item 10](#24-️-verified-environment--known-issues)):
```bash
python examples/run_keyframe_search.py \
  --video_path ./data/videos/xxx.mp4 \
  --target_objects "person,cup" \
  --cue_objects "table" \
  --detector yoloworld \
  --yoloworld_config third_party/YOLO-World/configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
  --yoloworld_ckpt /path/to/s_stage1-d1c1d7d8.pth \
  --search_nframes 8 \
  --device cuda:0 \
  --output_dir ./output/
```

### 4.2 Keyframe Retrieval (Batch — Full VSI with Subtitles)
The main VSI pipeline (`VSI_keyframe_search.py`) runs in batch mode over a JSON dataset file.
Each item in the JSON must contain `video_id`, `video_path`, `question`, `options`, and `grounding_objects`
(target/cue objects pre-extracted via `TStarUniversalGrounder`).

```bash
python VSI_keyframe_search.py \
  --input_json  ./data/processed/LongVideoBench/annotations.json \
  --output_json ./output/results_lvbench.json \
  --config_path /path/to/yolo_world.py \
  --checkpoint_path /path/to/yolo_world.pth \
  --dataset LongVideoBench \
  --subtitle_root ./data/processed/LongVideoBench/subtitles/ \
  --text_weight 0.3 \
  --search_nframes 8 \
  --device cuda:0
```

To run on VideoMME, set `--dataset VideoMME` (loads `.srt` subtitles automatically).
Omit `--subtitle_root` to run in visual-only mode (`text_weight` has no effect).

### 4.3 Downstream VideoQA (GPT-4o)
Run GPT-4o on the retrieved keyframes using the KFSBench inference script:
```bash
export OPENAI_API_KEY=<your_openai_api_key>
python KFSBench/scripts/inference/gpt4_inference.py \
  --videos_path ./data/processed/LongVideoBench \
  --json_path   ./data/processed/LongVideoBench/annotations \
  --json_file   results_lvbench.json \
  --output_file ./output/gpt4_qa_results.jsonl \
  --model gpt-4o
```

## 5. Experiment Reproduction
### 5.1 Keyframe Search Accuracy
After running `VSI_keyframe_search.py` to produce a result JSON, evaluate keyframe retrieval accuracy (Precision / Recall / F1, threshold ≤200 frames from ground truth):
```bash
python KFSBench/scripts/evaluation/eval_json.py \
  --json_path ./output/results_lvbench.json
```

For uniform-sampling baseline frames extraction:
```bash
python KFSBench/scripts/evaluation/search_frames.py \
  --json_path  ./data/processed/LongVideoBench/annotations.json \
  --output_dir ./output/uniform_frames/ \
  --num_frames 8 \
  --video_dir  ./data/processed/LongVideoBench/videos/
```

### 5.2 Downstream VideoQA Performance
Run GPT-4o on retrieved frames (see Section 4.3), then evaluate answer accuracy:
```bash
python KFSBench/scripts/evaluation/eval_metrics_lvbench.py \
  --result_file ./output/gpt4_qa_results.jsonl
```

### 5.3 Baseline Comparison
Baseline evaluation scripts are not yet released. Uniform sampling can be reproduced with `search_frames.py` (see 5.1 above). TSTAR and VSLS baselines are available in the TStar module (`TStar/TStarFramework.py`).

## 6. Main Experimental Results
All numbers below are from the [paper](https://arxiv.org/abs/2508.06869) (arXiv v4). The [project page](https://jacksonha7.github.io/VSI-page/) has the full tables.

### 6.1 Keyframe Search Accuracy (LongVideoBench, 64 frames)
| Method | TFLOPs | Latency (s) | Full | Image-only | Text-only |
|--------|--------|-------------|------|------------|-----------|
| T\*    | 31.7   | 28.96       | 67.58% | 70.34% | 66.36% |
| VSLS   | 33.3   | 33.26       | 70.23% | 68.62% | 66.36% |
| **VSI (Ours)** | 36.8 | 31.71 | **73.89%** | **71.56%** | **77.17%** |

All three methods are training-free and use YOLO-World-110M for video search; VSI also encodes subtitles with all-mpnet-base-v2. Latency is on the full set.

### 6.2 Text-related Tasks (LongVideoBench text-related perception subsets, 8 frames)
| Method | Keyframe Search Acc | GPT-4o Acc (Medium) | GPT-4o Acc (Long) |
|--------|---------------------|---------------------|-------------------|
| GPT-4o (uniform sampling) | – | 48.28% | 53.76% |
| GPT-4o + subtitles in prompt | – | 46.55% | 58.06% |
| GPT-4o + T\* | 29.48% | 48.28% | 58.06% |
| GPT-4o + VSLS | 27.17% | 48.28% | 55.91% |
| **GPT-4o + VSI (Ours)** | **45.00%** | **62.07%** | **69.57%** |

### 6.3 Downstream VideoQA
Evaluated with GPT-4o, LLaVA-Video-7B-Qwen2 and Qwen2.5-VL-7B-Instruct at 8 and 32 frames, on medium and long videos. Compared with uniform sampling, VSI improves or ties in all 12 LongVideoBench settings and improves in 11 of the 12 Video-MME settings. For example, with GPT-4o and 32 frames on LongVideoBench, accuracy goes from 48.2% to 51.2% on long videos and from 51.9% to 53.9% on medium videos.

## 7. Project Structure
```
VSI/
├── VSI_keyframe_search.py   # Main pipeline — batch VSI dual-branch keyframe retrieval
├── Keyframe_Matching.py     # Keyframe evaluation / matching utilities
├── TStar/                   # Visual-only baseline and core search utilities
│   ├── TStarFramework.py    # Visual-only TStarFramework baseline
│   ├── interface_searcher.py# TStarSearcher — iterative frame sampling & distribution update
│   ├── interface_yolo.py    # YOLO-World detection wrapper (MMDetection)
│   ├── interface_llm.py     # TStarUniversalGrounder — GPT-4o grounding (extracts objects)
│   └── utilites.py          # Frame loading, base64 encoding, GIF saving
├── KFSBench/                # Keyframe search benchmark evaluation suite
│   ├── scripts/
│   │   ├── evaluation/      # eval_json.py, search_frames.py, eval_metrics_lvbench.py
│   │   └── inference/       # gpt4_inference.py — downstream VideoQA with GPT-4o
│   └── src/evaluation/      # metrics.py — valid-search-ratio and temporal-coverage
├── examples/
│   └── run_keyframe_search.py  # Single-video quick start (visual-only)
├── scripts/
│   └── setup_env.sh         # One-shot conda env + YOLO-World setup (see 2.4)
├── third_party/YOLO-World/  # Cloned by scripts/setup_env.sh, not committed
├── requirements.txt         # Dependencies list (VSI's own; see 2.4 for YOLO-World's)
├── constraints.txt          # pip constraints pinning the verified-working combo
├── LICENSE                  # License file
└── README.md                # Project documentation
```

## 8. Model Weight
All pre-trained model weights used in VSI are publicly available:
- YOLO-World-110M: [https://github.com/AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- all-mpnet-base-v2: [https://huggingface.co/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- LLaVA-Video-7B-Qwen2: [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- Qwen2.5VL-7B-Instruct: [https://huggingface.co/Qwen/Qwen2.5VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5VL-7B-Instruct)

Download each weight manually from the links above and place them under `./checkpoints/`.

## 9. Ablation Study
Ablation study scripts (single-branch evaluation, text weight tuning) are not yet released. To reproduce single-branch results manually, run `VSI_keyframe_search.py` with `--text_weight 0` (visual-only) or pass `--subtitle_root` without `--text_weight` to control branch contribution.

## 10. Citation
If you find our work useful in your research, please cite our paper:
```bibtex
@misc{he2025vsivisualsubtitleintegration,
      title={VSI: Visual Subtitle Integration for Keyframe Selection to enhance Long Video Understanding}, 
      author={Jianxiang He and Meisheng Hong and Jungang Li and Ziyang Chen and Weiyu Guo and Xuming Hu and Hui Xiong},
      year={2025},
      eprint={2508.06869},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.06869}, 
}
```

## 11. License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 12. Acknowledgements
- We thank the authors of YOLO-World, Sentence-Transformers, LLaVA-Video for their open-source code and pre-trained models.
- We acknowledge the support of HKUST(GZ) and MBZUAI.
- This work is based on the LongVideoBench and VideoMME datasets, thanks to their authors for the public release.

## 13. Contact
If you have any questions, issues or suggestions, please contact:
- Author Email: jhe307@connect.hkust-gz.edu.cn

- GitHub Issues: [https://github.com/Jacksonha7/Visual-Subtitle-Integration/issues](https://github.com/Jacksonha7/Visual-Subtitle-Integration/issues)

