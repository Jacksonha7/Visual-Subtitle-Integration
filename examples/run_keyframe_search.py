import argparse
import logging
import os
from typing import List

from TStar.interface_searcher import TStarSearcher
from TStar.interface_yolo import YoloV5Interface, YoloWorldInterface


def _parse_list(csv: str) -> List[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run TStar keyframe search (visual-only).")

    parser.add_argument("--video_path", required=True, help="Path to an input video.")
    parser.add_argument("--target_objects", required=True, help="Comma-separated target objects (e.g. 'person,microphone').")
    parser.add_argument("--cue_objects", default="", help="Comma-separated cue objects.")

    parser.add_argument("--detector", choices=["yolov5", "yoloworld"], default="yolov5")
    parser.add_argument("--yoloworld_config", default=None, help="YOLO-World config path (MMDetection).")
    parser.add_argument("--yoloworld_ckpt", default=None, help="YOLO-World checkpoint path.")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--search_nframes", type=int, default=8)
    parser.add_argument("--grid_rows", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--search_budget", type=float, default=0.5, help="Ratio of frames (at 1fps) to process.")
    parser.add_argument("--output_dir", default="./output", help="Where to save debug artifacts (optional).")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    target_objects = _parse_list(args.target_objects)
    cue_objects = _parse_list(args.cue_objects)

    if args.detector == "yoloworld":
        if not args.yoloworld_config or not args.yoloworld_ckpt:
            raise ValueError("--yoloworld_config and --yoloworld_ckpt are required for detector=yoloworld")
        yolo = YoloWorldInterface(
            config_path=args.yoloworld_config,
            checkpoint_path=args.yoloworld_ckpt,
            device=args.device,
        )
    else:
        yolo = YoloV5Interface(device=args.device)

    searcher = TStarSearcher(
        video_path=args.video_path,
        target_objects=target_objects,
        cue_objects=cue_objects,
        search_nframes=args.search_nframes,
        image_grid_shape=(args.grid_rows, args.grid_cols),
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        output_dir=args.output_dir,
        yolo_scorer=yolo,
        update_method="spline",
    )

    frames, timestamps, num_iters = searcher.search_with_visualization()
    logging.info("Done. iters=%d, timestamps=%s", num_iters, timestamps)


if __name__ == "__main__":
    main()

