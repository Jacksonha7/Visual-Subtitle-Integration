"""
Multimodal TStar keyframe search pipeline.

This module provides a modular reference implementation for **multimodal keyframe search**:
- Visual scoring via object detection on sampled frame grids
- Text scoring via question–subtitle similarity (optional)
- Score fusion to update a frame probability distribution during search

It is designed as a *pipeline script* (CLI) and a *reference implementation* that can be
reused as a library. For open-source release, avoid hard-coded paths and provide external
resources (datasets/checkpoints) via CLI arguments.
"""

import os
import json
import time
import copy
import logging
import argparse
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
from decord import VideoReader, cpu

# 导入TStar核心组件
from TStar.interface_llm import TStarUniversalGrounder
from TStar.interface_yolo import YoloWorldInterface
from TStar.interface_searcher import TStarSearcher
from TStar.utilites import load_video_frames, encode_image_to_base64

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VideoSearchConfig:
    """Configuration for multimodal keyframe search on a single video."""
    # 基础参数
    video_path: str
    subtitle_path: str
    question: str
    options: str
    
    # 模型参数
    text_weight: float = 0.4
    search_nframes: int = 8
    grid_rows: int = 4
    grid_cols: int = 4
    confidence_threshold: float = 0.7
    search_budget: float = 1.0
    relation_alpha: float = 0.0
    
    # 设备配置
    device: str = "cuda:7"
    output_dir: str = "./output"
    prefix: str = "stitched_image"
    
    # 文本编码器
    text_encoder_name: str = "sentence-transformers/all-mpnet-base-v2"


@dataclass
class SearchResult:
    """Structured outputs produced by the keyframe search pipeline."""
    video_path: str
    grounding_objects: Dict[str, Any]
    keyframe_timestamps: List[float]
    frame_distribution: List[float]
    score_distribution: List[float]
    num_iterations: int
    search_time: float
    total_time: float
    position_match_analysis: Optional[Dict] = None


class SubtitleProcessor:
    """Subtitle loader supporting JSON (LVBench) and SRT (VideoMME)."""
    
    @staticmethod
    def load_subtitles(subtitle_path: str) -> List[Dict]:
        """
        根据文件类型加载字幕文件
        
        Args:
            subtitle_path: 字幕文件路径
            
        Returns:
            字幕列表，每项包含start, end, text字段
        """
        if not subtitle_path or subtitle_path == "" or not os.path.exists(subtitle_path):
            return []
            
        if subtitle_path.endswith('.json'):
            return SubtitleProcessor._load_json_subtitles(subtitle_path)
        elif subtitle_path.endswith('.srt'):
            return SubtitleProcessor._load_srt_subtitles(subtitle_path)
        else:
            raise ValueError(f"不支持的字幕文件格式: {subtitle_path}")
    
    @staticmethod
    def _load_json_subtitles(subtitle_path: str) -> List[Dict]:
        """加载JSON格式字幕（LongVideoBench）"""
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载JSON字幕文件失败: {subtitle_path}, 错误: {e}")
            return []
    
    @staticmethod
    def _load_srt_subtitles(subtitle_path: str) -> List[Dict]:
        """加载SRT格式字幕（VideoMME）"""
        import re
        subtitles = []
        
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            blocks = re.split(r'\n\s*\n', content.strip())
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                    if time_match:
                        start, end = time_match.group(1), time_match.group(2)
                        start_sec = SubtitleProcessor._timestamp_to_seconds(start)
                        end_sec = SubtitleProcessor._timestamp_to_seconds(end)
                        text = ' '.join([re.sub(r'<.*?>', '', l) for l in lines[2:]]).strip()
                        subtitles.append({'start': start_sec, 'end': end_sec, 'text': text})
        except Exception as e:
            logger.warning(f"加载SRT字幕文件失败: {subtitle_path}, 错误: {e}")
            return []
        
        return subtitles
    
    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """将时间戳字符串转换为秒"""
        try:
            if "," in timestamp:  # SRT格式 HH:MM:SS,mmm
                h, m, s = timestamp.split(":")
                s, ms = s.split(",")
                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            else:  # 普通格式 HH:MM:SS.sss
                h, m, s = timestamp.split(":")
                if "." in s:
                    s_val, ms_val = s.split(".")
                    return int(h) * 3600 + int(m) * 60 + int(s_val) + int(ms_val) / 1000
                else:
                    return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception as e:
            logger.warning(f"时间戳解析错误: {timestamp}, 错误: {e}")
            return 0.0


class TextSimilarityCalculator:
    """Compute question–subtitle similarity and map it to frame-level scores."""
    
    def __init__(self, text_encoder_name: str, device: str):
        """
        初始化文本编码器
        
        Args:
            text_encoder_name: 文本编码器模型名称
            device: 运行设备
        """
        # 延迟导入sentence_transformers
        from sentence_transformers import SentenceTransformer
        
        self.text_encoder = SentenceTransformer(text_encoder_name)
        if device.startswith('cuda'):
            self.text_encoder = self.text_encoder.to(device)
        
        # 停用词列表
        self.stopwords = {"a", "an", "the", "is", "are", "in", "on", "at", "of", "and", "or", "with"}
    
    def compute_similarity_scores(self, question: str, subtitles: List[Dict], 
                                total_frames: int, fps: float) -> np.ndarray:
        """
        计算问题与字幕的相似度分数
        
        Args:
            question: 问题文本
            subtitles: 字幕列表
            total_frames: 视频总帧数
            fps: 视频帧率
            
        Returns:
            每个时间戳的相似度分数数组
        """
        if not subtitles:
            logger.info("没有字幕可用，返回零分数数组")
            return np.zeros(total_frames)
        
        # 提取字幕文本和时间戳
        subtitle_texts, timestamps = self._extract_subtitle_data(subtitles)
        
        # 计算文本相似度
        similarities = self._compute_text_similarities(question, subtitle_texts)
        
        # 应用软阈值增强
        boosted_similarities = self._apply_soft_threshold(similarities)
        
        # 转换为帧级分数
        frame_scores = self._convert_to_frame_scores(
            timestamps, boosted_similarities, total_frames, fps
        )
        
        return frame_scores
    
    def _extract_subtitle_data(self, subtitles: List[Dict]) -> Tuple[List[str], List[Tuple[float, float]]]:
        """提取字幕文本和时间戳"""
        subtitle_texts = []
        timestamps = []
        
        for sub in subtitles:
            # 兼容不同格式的字幕数据
            subtitle_text = sub.get("line", sub.get("text", ""))
            subtitle_texts.append(subtitle_text)
            
            # 兼容不同格式的时间戳，处理None值
            start_val = sub.get("start")
            end_val = sub.get("end")
            
            if start_val is None or end_val is None:
                continue  # 跳过没有时间戳的字幕
            
            if isinstance(start_val, (int, float)):
                start_time = float(start_val)
            else:
                start_time = SubtitleProcessor._timestamp_to_seconds(str(start_val))
            
            if isinstance(end_val, (int, float)):
                end_time = float(end_val)
            else:
                end_time = SubtitleProcessor._timestamp_to_seconds(str(end_val))
            
            timestamps.append((start_time, end_time))
        
        return subtitle_texts, timestamps
    
    def _compute_text_similarities(self, question: str, subtitle_texts: List[str]) -> np.ndarray:
        """计算文本相似度"""
        # 过滤问题文本
        question_filtered = self._filter_text(question)
        
        with torch.no_grad():
            # 编码问题文本
            q_embedding = self.text_encoder.encode(question_filtered, convert_to_tensor=True)
            
            # 批量处理字幕文本
            batch_size = 32
            similarities = []
            
            for i in range(0, len(subtitle_texts), batch_size):
                batch_texts = subtitle_texts[i:i+batch_size]
                filtered_batch_texts = [self._filter_text(text) for text in batch_texts]
                
                batch_embeddings = self.text_encoder.encode(filtered_batch_texts, convert_to_tensor=True)
                batch_similarities = torch.nn.functional.cosine_similarity(
                    q_embedding.unsqueeze(0), batch_embeddings
                ).cpu().numpy()
                
                similarities.extend(batch_similarities)
        
        return np.array(similarities)
    
    def _filter_text(self, text: str) -> str:
        """过滤文本，移除停用词"""
        words = text.lower().split()
        filtered_words = [w for w in words if w not in self.stopwords]
        return " ".join(filtered_words)
    
    def _apply_soft_threshold(self, similarities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """应用软阈值增强高相似度分数"""
        boosted_similarities = np.where(
            similarities > threshold,
            threshold + (similarities - threshold) * 2,  # 放大高相似度值
            similarities
        )
        return np.minimum(boosted_similarities, 1.0)  # 确保最大值不超过1.0
    
    def _convert_to_frame_scores(self, timestamps: List[Tuple[float, float]], 
                               similarities: np.ndarray, total_frames: int, 
                               fps: float, time_window: float = 2.0) -> np.ndarray:
        """将字幕相似度转换为帧级分数"""
        frame_scores = np.zeros(total_frames)
        
        for (start, end), score in zip(timestamps, similarities):
            if score > 0.2:  # 过滤低相似度字幕
                start_frame = max(0, int((start - time_window) * fps))
                end_frame = min(total_frames, int((end + time_window) * fps))
                
                # 使用高斯加权
                frame_count = end_frame - start_frame
                if frame_count > 0:
                    center = (start_frame + end_frame) // 2
                    sigma = frame_count / 4
                    
                    for f in range(start_frame, end_frame):
                        weight = np.exp(-((f - center) ** 2) / (2 * sigma ** 2))
                        frame_scores[f] = max(frame_scores[f], score * weight)
        
        return frame_scores


class ScoreFusionManager:
    """Fuse visual scores and text scores into a single distribution."""
    
    def __init__(self, text_weight: float):
        """
        初始化融合管理器
        
        Args:
            text_weight: 文本相似度权重
        """
        self.text_weight = text_weight
    
    def fuse_scores(self, visual_scores: np.ndarray, text_scores: np.ndarray) -> np.ndarray:
        """
        融合视觉和文本分数
        
        Args:
            visual_scores: 视觉检测分数
            text_scores: 文本相似度分数
            
        Returns:
            融合后的分数
        """
        # 重采样文本分数以匹配视觉分数长度
        resampled_text_scores = self._resample_text_scores(text_scores, len(visual_scores))
        
        # 归一化处理
        normalized_visual = self._normalize_scores(visual_scores)
        normalized_text = self._normalize_scores(resampled_text_scores)
        
        # 静态权重融合
        fused_scores = (1 - self.text_weight) * normalized_visual + self.text_weight * normalized_text
        
        return fused_scores
    
    def _resample_text_scores(self, text_scores: np.ndarray, target_length: int) -> np.ndarray:
        """重采样文本分数以匹配目标长度"""
        if len(text_scores) == target_length:
            return text_scores

        original_indices = np.linspace(0, len(text_scores) - 1, len(text_scores))
        target_indices = np.linspace(0, len(text_scores) - 1, target_length)

        # NumPy-only linear interpolation (avoids SciPy binary dependency).
        return np.interp(target_indices, original_indices, text_scores, left=0.0, right=0.0)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """归一化分数"""
        score_max = np.max(scores)
        score_min = np.min(scores)
        
        if score_max > score_min:
            return (scores - score_min) / (score_max - score_min)
        else:
            return np.ones_like(scores) * 0.5


class MultimodalTStarFramework:
    """多模态TStar框架主类 - 整合所有组件"""
    
    def __init__(
        self,
        config: VideoSearchConfig,
        grounder: Optional[TStarUniversalGrounder],
        yolo_scorer: YoloWorldInterface,
    ):
        """
        初始化多模态TStar框架
        
        Args:
            config: 视频搜索配置
            grounder: 通用目标定位器
            yolo_scorer: YOLO目标检测器
        """
        self.config = config
        self.grounder = grounder
        self.yolo_scorer = yolo_scorer
        
        # 初始化组件
        self.subtitle_processor = SubtitleProcessor()
        self.text_calculator = TextSimilarityCalculator(
            config.text_encoder_name, config.device
        )
        self.score_fusion_manager = ScoreFusionManager(config.text_weight)
        
        # 加载字幕
        self.subtitles = self.subtitle_processor.load_subtitles(config.subtitle_path)
        
        # 获取视频信息
        self._initialize_video_info()
        
        # 初始化结果存储
        self.results = {}
    
    def _initialize_video_info(self):
        """初始化视频信息"""
        video_reader = VideoReader(self.config.video_path, ctx=cpu(0))
        self.total_frame_num = len(video_reader)
        self.fps = video_reader.get_avg_fps()
        video_reader = None  # 释放资源
    
    def process_video(self, grounding_objects: Dict[str, Any]) -> SearchResult:
        """
        处理单个视频的搜索和问答任务
        
        Args:
            grounding_objects: 预提取的目标对象信息
            
        Returns:
            搜索结果
        """
        start_time = time.time()
        
        # 提取目标对象
        target_objects = grounding_objects['target_objects']
        cue_objects = grounding_objects['cue_objects']
        relations = grounding_objects.get('relations', [])
        
        # 初始化视频搜索器
        video_searcher = self._initialize_searcher(target_objects, cue_objects, relations)
        
        # 计算文本相似度分数
        text_scores = self._compute_text_similarity_scores()
        
        # 设置分数融合
        self._setup_score_fusion(video_searcher, text_scores)
        
        # 执行搜索
        search_start = time.time()
        all_frames, timestamps = self._perform_search(video_searcher)
        search_time = time.time() - search_start
        
        # 收集结果
        result = self._collect_results(
            video_searcher, all_frames, timestamps, search_time, time.time() - start_time
        )
        
        return result
    
    def _initialize_searcher(self, target_objects: List[str], cue_objects: List[str], 
                           relations: List[Tuple]) -> TStarSearcher:
        """初始化视频搜索器"""
        video_searcher = TStarSearcher(
            video_path=self.config.video_path,
            target_objects=target_objects,
            cue_objects=cue_objects,
            relations=relations,
            search_nframes=self.config.search_nframes,
            image_grid_shape=(self.config.grid_rows, self.config.grid_cols),
            output_dir=self.config.output_dir,
            confidence_threshold=self.config.confidence_threshold,
            search_budget=self.config.search_budget,
            yolo_scorer=self.yolo_scorer,
            update_method="spline"
        )
        
        video_searcher.relation_alpha = self.config.relation_alpha
        return video_searcher
    
    def _compute_text_similarity_scores(self) -> np.ndarray:
        """计算文本相似度分数"""
        return self.text_calculator.compute_similarity_scores(
            self.config.question, self.subtitles, self.total_frame_num, self.fps
        )
    
    def _setup_score_fusion(self, video_searcher: TStarSearcher, text_scores: np.ndarray):
        """设置分数融合"""
        original_update_method = video_searcher.update_frame_distribution
        
        def fused_update_method(sampled_frame_indices, confidence_maps, detected_objects_maps, bbox_maps):
            # 获取视觉分数
            frame_confidences, frame_detected_objects = original_update_method(
                sampled_frame_indices, confidence_maps, detected_objects_maps, bbox_maps
            )
            
            # 获取当前视觉分数分布
            visual_scores = video_searcher.score_distribution.copy()
            
            # 融合分数
            fused_scores = self.score_fusion_manager.fuse_scores(visual_scores, text_scores)
            
            # 更新分数分布
            video_searcher.score_distribution = fused_scores
            
            # 重新计算概率分布
            self._update_probability_distribution(video_searcher)
            
            return frame_confidences, frame_detected_objects
        
        # 替换更新方法
        video_searcher.update_frame_distribution = fused_update_method
    
    def _update_probability_distribution(self, video_searcher: TStarSearcher):
        """更新概率分布"""
        if video_searcher.update_method == "spline":
            video_searcher.P = video_searcher.spline_keyframe_distribution(
                video_searcher.non_visiting_frames,
                video_searcher.score_distribution,
                len(video_searcher.score_distribution)
            )
        elif video_searcher.update_method == "gaussian":
            video_searcher.P = video_searcher.gaussioan_score_distribution(
                video_searcher.non_visiting_frames,
                video_searcher.score_distribution,
                len(video_searcher.score_distribution)
            )
        
        # 存储更新后的分布
        video_searcher.store_score_distribution()
    
    def _perform_search(self, video_searcher: TStarSearcher) -> Tuple[List[np.ndarray], List[float]]:
        """执行视频搜索"""
        try:
            all_frames, timestamps, num_iterations = video_searcher.search_with_visualization()
            self.results['num_iterations'] = num_iterations
            return all_frames, timestamps
        finally:
            # 清理资源
            torch.cuda.empty_cache()
    
    def _collect_results(self, video_searcher: TStarSearcher, all_frames: List[np.ndarray], 
                        timestamps: List[float], search_time: float, total_time: float) -> SearchResult:
        """收集搜索结果"""
        return SearchResult(
            video_path=self.config.video_path,
            grounding_objects=self.results.get('Searching_Objects', {}),
            keyframe_timestamps=timestamps,
            frame_distribution=video_searcher.P_history[-1] if video_searcher.P_history else [],
            score_distribution=video_searcher.score_distribution.tolist(),
            num_iterations=self.results.get('num_iterations', 0),
            search_time=search_time,
            total_time=total_time
        )


class VideoProcessor:
    """视频处理器 - 负责批量处理视频"""
    
    def __init__(self, grounder: Optional[TStarUniversalGrounder], yolo_scorer: YoloWorldInterface):
        """
        初始化视频处理器
        
        Args:
            grounder: 通用目标定位器
            yolo_scorer: YOLO目标检测器
        """
        self.grounder = grounder
        self.yolo_scorer = yolo_scorer
    
    def process_single_video(self, data_item: Dict[str, Any], config: VideoSearchConfig, dataset: str = "LongVideoBench") -> SearchResult:
        """
        处理单个视频
        
        Args:
            data_item: 视频数据项
            config: 搜索配置
            
        Returns:
            搜索结果
        """
        # 更新配置
        config.video_path = data_item['video_path']
        config.question = data_item['question']
        config.options = data_item['options']
        
        # 设置字幕路径
        subtitle_path = self._get_subtitle_path(data_item, config, dataset)
        
        # 更新配置中的字幕路径
        config.subtitle_path = subtitle_path
        
        # 创建框架实例
        framework = MultimodalTStarFramework(config, self.grounder, self.yolo_scorer)
        
        # 处理视频
        result = framework.process_video(data_item['grounding_objects'])
        
        return result
    
    def _get_subtitle_path(self, data_item: Dict[str, Any], config: VideoSearchConfig, dataset: str = "LongVideoBench") -> str:
        """Get subtitle file path for a given video (optional)."""
        video_id = data_item['video_id']

        # If subtitle root is not provided, run in visual-only mode.
        if not config.subtitle_path:
            return ""

        # Choose subtitle extension by dataset convention.
        if dataset == 'VideoMME':
            return os.path.join(config.subtitle_path, f"{video_id}.srt")
        return os.path.join(config.subtitle_path, f"{video_id}_en.json")


def create_config_from_args(args) -> VideoSearchConfig:
    """从命令行参数创建配置"""
    return VideoSearchConfig(
        video_path="",  # 将在处理时设置
        subtitle_path=args.subtitle_root,
        question="",  # 将在处理时设置
        options="",  # 将在处理时设置
        text_weight=args.text_weight,
        search_nframes=args.search_nframes,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        relation_alpha=args.relation_alpha,
        device=args.device,
        output_dir=args.output_dir,
        prefix=args.prefix
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="重构版多模态TStar框架")
    
    # 基础参数
    parser.add_argument('--text_weight', type=float, default=0.3, help='文本相似度权重')
    parser.add_argument('--input_json', type=str, required=True, help='Input JSON containing videos + grounding objects.')
    parser.add_argument('--output_json', type=str, default=None, help='Where to write results JSON. Defaults to output_dir/<auto_name>.json')
    parser.add_argument('--dataset', type=str, default="LongVideoBench", help='数据集名称')
    parser.add_argument('--subtitle_root', type=str, default=None, help='字幕文件根目录')
    
    # 模型参数
    parser.add_argument('--config_path', type=str, required=True, help='YOLO-World config path (MMDetection).')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='YOLO-World checkpoint path.')
    parser.add_argument('--device', type=str, default="cuda:0", help='运行设备')
    
    # 搜索参数
    parser.add_argument('--search_nframes', type=int, default=8, help='返回的top帧数')
    parser.add_argument('--grid_rows', type=int, default=4, help='图像网格行数')
    parser.add_argument('--grid_cols', type=int, default=4, help='图像网格列数')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='检测置信度阈值')
    parser.add_argument('--search_budget', type=float, default=1.0, help='搜索预算')
    parser.add_argument('--relation_alpha', type=float, default=0.0, help='关系权重')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--prefix', type=str, default='stitched_image', help='输出文件前缀')
    parser.add_argument('--num', type=int, default=0, help='处理的视频数量')
    parser.add_argument('--save_batch', type=int, default=10, help='批量保存间隔')
    
    # 调试参数
    parser.add_argument('--debug_mode', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型
    # This pipeline expects pre-extracted `grounding_objects` in the input JSON, so LLM is optional.
    grounder = None
    
    yolo_scorer = YoloWorldInterface(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 初始化处理器
    processor = VideoProcessor(grounder, yolo_scorer)
    
    # 加载数据
    with open(args.input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if args.num > 0:
        dataset = dataset[:args.num]
    
    # 处理视频
    results = []
    for idx, data_item in enumerate(dataset):
        logger.info(f"处理 {idx+1}/{len(dataset)}: {data_item['video_id']}")
        
        try:
            result = processor.process_single_video(data_item, config, args.dataset)
            
            # 转换为字典格式
            result_dict = {
                "video_path": result.video_path,
                "grounding_objects": result.grounding_objects,
                "keyframe_timestamps": result.keyframe_timestamps,
                "frame_distribution": result.frame_distribution,
                "score_distribution": result.score_distribution,
                "num_iterations": result.num_iterations,
                "search_time": result.search_time,
                "total_time": result.total_time
            }
            
            # 更新原始数据项
            data_item.update(result_dict)
            results.append(data_item)
            
            logger.info(f"完成: {data_item['video_id']}")
            
        except Exception as e:
            logger.error(f"处理 {data_item['video_id']} 时出错: {e}")
            data_item.update({
                "video_path": data_item.get('video_path', ''),
                "grounding_objects": [],
                "keyframe_timestamps": [],
                "error": str(e)
            })
            results.append(data_item)
        
        # 批量保存
        if (idx + 1) % args.save_batch == 0 or (idx + 1) == len(dataset):
            if args.output_json:
                output_json = args.output_json
            else:
                base = os.path.splitext(os.path.basename(args.input_json))[0]
                output_json = os.path.join(
                    args.output_dir,
                    f"{base}_MVSLS_F{args.search_nframes}_w{args.text_weight}_rel{args.relation_alpha}.json",
                )

            with open(output_json, 'w', encoding='utf-8') as f_out:
                json.dump(results, f_out, indent=4, ensure_ascii=False)
    
    logger.info("批量处理完成，结果已保存")


if __name__ == "__main__":
    main()
