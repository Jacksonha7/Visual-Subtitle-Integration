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
