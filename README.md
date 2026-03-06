# VSI: Visual–Subtitle Integration for Keyframe Selection
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-orange.svg)](https://cvpr.thecvf.com/)

Official implementation of the paper **VSI: Visual–Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding** (CVPR 2026 Submission #9004). A multimodal keyframe retrieval framework that fuses visual and subtitle information for long video understanding, achieving SOTA performance on LongVideoBench and VideoMME datasets.

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
1. **LongVideoBench** ([Download Link](<https://xxx>)): Large-scale long video-language understanding benchmark (3-60 minutes videos)
2. **VideoMME** ([Download Link](<https://github.com/OpenGVLab/VideoMME>)): Multimodal video QA benchmark (4-60 minutes videos)

### 3.2 Data Preprocessing
1. Download the original dataset and unzip to `./data/`
2. Run subtitle-video alignment preprocessing (generate timestamp-aligned subtitle files):
   ```bash
   python scripts/preprocess_subtitle.py --data_root ./data/LongVideoBench --save_root ./data/processed/LongVideoBench
   ```
3. Extract video frames (optional, for faster processing):
   ```bash
   python scripts/extract_frames.py --video_root ./data/LongVideoBench/videos --save_root ./data/processed/LongVideoBench/frames --fps 1
   ```

### 3.3 Dataset Directory Structure
```
./data/processed/
├── LongVideoBench/
│   ├── videos/          # Original video files (mp4/mkv)
│   ├── frames/          # Extracted video frames (per video folder)
│   ├── subtitles/       # Timestamp-aligned subtitles (json/txt)
│   └── annotations/     # Official QA/keyframe annotations
└── VideoMME/
    ├── [same structure as LongVideoBench]
```

## 4. Quick Start
### 4.1 Keyframe Retrieval
Run VSI to retrieve keyframes for a single video with a textual query:
```bash
python src/inference_keyframe.py \
  --video_path ./data/processed/LongVideoBench/videos/xxx.mp4 \
  --subtitle_path ./data/processed/LongVideoBench/subtitles/xxx.json \
  --query "Your textual query for keyframe retrieval" \
  --top_k 4 \ # Number of keyframes to retrieve (4/8/32 as in paper)
  --save_path ./output/keyframes/xxx/
```
**Output**: Top-k keyframe images + keyframe timestamp file + retrieval score file.

### 4.2 Downstream VideoQA
Use VSI-retrieved keyframes for VideoQA with GPT-4o/LLaVA-Video:
```bash
# GPT-4o (need to set OPENAI_API_KEY in env)
python src/inference_videoqa.py \
  --keyframe_root ./output/keyframes/xxx/ \
  --query "Your VideoQA question" \
  --model gpt-4o \
  --api_key <your_openai_api_key> \
  --save_answer ./output/qa/xxx_gpt4o.json

# LLaVA-Video (local weight)
python src/inference_videoqa.py \
  --keyframe_root ./output/keyframes/xxx/ \
  --query "Your VideoQA question" \
  --model llava-video-7b \
  --model_path <path_to_llava_video_weight> \
  --save_answer ./output/qa/xxx_llava.json
```

### 4.3 Batch Processing
Run keyframe retrieval + VideoQA for the whole dataset:
```bash
python scripts/run_batch.py \
  --data_root ./data/processed/LongVideoBench \
  --top_k 4 \
  --model gpt-4o \
  --api_key <your_openai_api_key> \
  --output_root ./output/batch/LongVideoBench/
```

## 5. Experiment Reproduction
### 5.1 Keyframe Search Accuracy
Reproduce the keyframe retrieval accuracy on LongVideoBench/VideoMME:
```bash
python experiments/eval_keyframe_accuracy.py \
  --data_root ./data/processed/LongVideoBench \
  --top_k 4/8/64 \
  --save_result ./output/eval/keyframe_accuracy.json
```
The evaluation metric follows the paper: **valid search ratio** (frame index deviation ≤200 from ground truth).

### 5.2 Downstream VideoQA Performance
Reproduce VideoQA accuracy on medium/long video settings:
```bash
python experiments/eval_videoqa.py \
  --data_root ./data/processed/LongVideoBench \
  --top_k 8/32 \
  --model gpt-4o/llava-video/qwen2.5vl \
  --save_result ./output/eval/videoqa_accuracy.json
```

### 5.3 Baseline Comparison
We provide implementation of baseline methods (TSTAR/VSLS/uniform sampling) for comparison:
```bash
# Run uniform sampling baseline
python experiments/run_baseline.py --method uniform --top_k 4 --data_root ./data/processed/LongVideoBench
# Run TSTAR baseline
python experiments/run_baseline.py --method tstar --top_k 4 --data_root ./data/processed/LongVideoBench
# Run VSLS baseline
python experiments/run_baseline.py --method vsls --top_k 4 --data_root ./data/processed/LongVideoBench
```

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
├── src/                # Core source code
│   ├── video_search/   # Video Search branch (YOLO-World, object scoring)
│   ├── subtitle_match/ # Subtitle Match branch (text encoding, similarity)
│   ├── fusion/         # Score fusion module (adaptive weighted fusion)
│   ├── qa/             # Downstream VideoQA inference
│   └── utils/          # Tool functions (frame processing, subtitle alignment, etc.)
├── data/               # Dataset (raw/processed)
├── scripts/            # Preprocessing/batch running scripts
├── experiments/        # Experiment evaluation scripts (accuracy/QA)
├── output/             # Inference/evaluation results (keyframes, QA answers, logs)
├── checkpoints/        # Pre-trained model weights (YOLO-World, etc.)
├── requirements.txt    # Dependencies list
├── LICENSE             # License file
└── README.md           # Project documentation
```

## 8. Model Weight
All pre-trained model weights used in VSI are publicly available:
- YOLO-World-110M: [https://github.com/AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- all-mpnet-base-v2: [https://huggingface.co/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- LLaVA-Video-7B-Qwen2: [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- Qwen2.5VL-7B-Instruct: [https://huggingface.co/Qwen/Qwen2.5VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5VL-7B-Instruct)

We provide a weight download script for quick setup:
```bash
python scripts/download_pretrain.py --save_path ./checkpoints/
```

## 9. Ablation Study
Code for ablation studies in the paper (branch contribution/text weight tuning):
```bash
# Evaluate single branch performance (Video Search/Subtitle Match)
python experiments/ablation_branch.py --data_root ./data/processed/LongVideoBench --top_k 4
# Evaluate text weight tuning (0.3/0.7/1.0)
python experiments/ablation_text_weight.py --data_root ./data/processed/LongVideoBench --text_weight 1.0 --top_k 4
```

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
- We acknowledge the support of <Your Lab/Institution/Funding Agency>.
- This work is based on the LongVideoBench and VideoMME datasets, thanks to their authors for the public release.

## 13. Contact
If you have any questions, issues or suggestions, please contact:
- Author Email: jhe307@connect.hkust-gz.edu.cn

- GitHub Issues: [https://github.com/<your_repo>/VSI/issues](https://github.com/<your_repo>/VSI/issues)

