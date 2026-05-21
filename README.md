# VSI: Visual–Subtitle Integration for Keyframe Selection
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-orange.svg)](https://cvpr.thecvf.com/)

Official implementation of the paper **VSI: Visual–Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding** (CVPR 2026 findings). A multimodal keyframe retrieval framework that fuses visual and subtitle information for long video understanding, achieving SOTA performance on LongVideoBench and VideoMME datasets.

## 1. Project Introduction
### 1.1 Core Motivation
Existing keyframe selection algorithms rely solely on visual modality, leading to poor performance on text-related long video tasks and deviation from core semantic content. VSI addresses this issue via a **dual-branch collaborative retrieval mechanism** (Video Search + Subtitle Match) to fuse complementary visual and textual information.

### 1.2 Key Advantages
- **Multimodal Fusion**: Integrate visual object detection and subtitle semantic similarity for precise keyframe localization.
- **Plug-and-Play**: No additional training required, lightweight and flexible, can be integrated into existing video-LM pipelines.
- **SOTA Performance**: Achieves 73.89% average keyframe search accuracy (sampling only 3.2% video frames) on LongVideoBench; 40.00% keyframe search accuracy on text-related tasks (4-frame setting, up from 19.65% baseline).
- **Strong Generalization**: Significantly improves downstream VideoQA performance (GPT-4o + VSI achieves 22.24% accuracy gain on Long-VideoQA tasks).

### 1.3 Main Tasks
- Keyframe retrieval for long video (3-60 minutes)
- Downstream VideoQA (visual/text-related sub-tasks)
- Multimodal long video understanding

## 2. Environment Configuration
### 2.1 Basic Dependencies
Our code is tested on Ubuntu 20.04/22.04 with the following core dependencies:
```bash
# Create conda environment
conda create -n vsi python=3.10
conda activate vsi

# Install PyTorch (match your CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other core dependencies
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
### 4.1 Keyframe Retrieval (Single Video — Visual-Only)
For a quick test on a single video without subtitles, use the example script:
```bash
python examples/run_keyframe_search.py \
  --video_path ./data/videos/xxx.mp4 \
  --target_objects "person,cup" \
  --cue_objects "table" \
  --detector yoloworld \
  --yoloworld_config /path/to/yolo_world.py \
  --yoloworld_ckpt /path/to/yolo_world.pth \
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
### 6.1 Keyframe Search Accuracy (LongVideoBench)
| Method       | Top-k | Image-only | Text-only | Full Dataset |
|--------------|-------|------------|-----------|--------------|
| TSTAR        | 64    | <xxx>%     | <xxx>%    | <xxx>%       |
| VSLS         | 64    | <xxx>%     | <xxx>%    | <xxx>%       |
| **VSI (Ours)**| 64    | <xxx>%     | **77.17%** | **73.89%**  |

### 6.2 Text-related Task Performance (LongVideoBench, 4-frame)
| Method       | Keyframe Acc | Medium VideoQA | Long VideoQA |
|--------------|--------------|----------------|--------------|
| GPT4o+TSTAR  | 19.65%       | 53.45%         | 53.76%       |
| GPT4o+VSLS   | 18.50%       | 50.00%         | 52.69%       |
| **GPT4o+VSI** | **40.00%**  | **63.79%**     | **68.48%**   |

### 6.3 Downstream VideoQA Gain (GPT-4o, 32-frame)
- LongVideoBench: **22.24%** accuracy gain compared to uniform sampling baseline
- VideoMME: **<xxx>%** accuracy gain compared to uniform sampling baseline

More detailed results can be found in our [paper](<paper.pdf>).

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
├── requirements.txt         # Dependencies list
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

