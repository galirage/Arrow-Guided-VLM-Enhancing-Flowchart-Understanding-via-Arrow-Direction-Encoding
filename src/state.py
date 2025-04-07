from typing import TypedDict, Optional
import argparse
import os
import datetime
from azure.ai.formrecognizer import DocumentAnalysisClient


class OCRState(TypedDict): # define state
    image_path: str
    extracted_text: Optional[str]
    document_analysis_client: DocumentAnalysisClient
    out_dir: str
    result: None
    text_and_bboxes: list


def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_path', '-igp', type=str, default="../images/flowchart-example166.png", help='image path')
    parser.add_argument('--output_dir', '-od', type=str, default='../output/', help='path to output directory')
    parser.add_argument('--ocr_precision', '-orp', type=str, default='low', help='low: vision/v3.2/ocr, high: vision/v3.2/read/analyze')

    return parser.parse_args()


if __name__ == '__main__':
    """
    usage for debug)
    python state.py
    """
    args = parser()
    state = OCRState()
    state['image_path'] = args.img_path
    print("state['image_path'], ", state['image_path'])