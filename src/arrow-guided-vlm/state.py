import argparse
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class OCRDetectionState(BaseModel):
    detection_result: Dict[str, List[Dict]]
    detection_ocr_result: Optional[Dict] = None
    detection_ocr_match_threshold: float = 0.5

    image_path: str
    image_num: str
    out_dir: str
    text_and_bboxes: List[Tuple[str, List[Tuple[float, float]]]]
    object_categories: Dict[int, str]
    arrow_category: int
    arrow_start: int
    arrow_end: int
    directed_graph_text: Optional[str] = None

    prompt: Optional[str] = None
    llm_result: Optional[Any] = None


def parser():
    parser = argparse.ArgumentParser(description="lllm_args")
    parser.add_argument(
        "--process_name", "-pn", type=str, default="load_pdf", help="process name"
    )
    parser.add_argument(
        "--img_path",
        "-igp",
        type=str,
        default="../images/flowchart-example166.png",
        help="image path",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        default="../output/",
        help="path to output directory",
    )
    parser.add_argument(
        "--ocr_precision",
        "-orp",
        type=str,
        default="low",
        help="low: vision/v3.2/ocr, high: vision/v3.2/read/analyze",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    usage for debug)
    python state.py
    """
    args = parser()
    print("Pydantic model defined. Run graph.py for usage.")
