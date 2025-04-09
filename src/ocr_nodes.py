from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Tuple, List, Dict
from dotenv import load_dotenv
import argparse
import os
import datetime
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from state import OCRDetectionState
# from state import DetectionState
import json


# class OCRState(TypedDict): # define state
#     image_path: str
#     extracted_text: Optional[str]
#     document_analysis_client: DocumentAnalysisClient
#     out_dir: str
#     result: None

def convert_polygon_to_bbox(polygon: List[Tuple[float, float]]) -> List[float]:
    """4点のpolygonを[x, y, w, h]に変換"""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """2つの矩形のIoUを計算"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def create_filename_with_timestamp() -> str:
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")  # 14桁のタイムスタンプを作成
    time_str = f"{timestamp}"
    return time_str

def run_azure_ocr(state: OCRDetectionState) -> OCRDetectionState: # run ocr
    image_path = state["image_path"]
    with open(image_path, "rb") as f:
        poller = state['document_analysis_client'].begin_analyze_document("prebuilt-read", document=f)
        result = poller.result()
        print("type(result) : ", type(result))
    extracted_text = "\n".join([line.content for page in result.pages for line in page.lines])
    state["image_path"] = image_path
    state["extracted_text"] = extracted_text
    state["result"] = result
    return state


def print_result(state: OCRDetectionState) -> OCRDetectionState: # output ocr
    print("\n 抽出されたテキスト:\n")
    print("state['extracted_text']", state['extracted_text'])
    result = state['result']

    extracted_text = []
    image = Image.open(state['image_path'])
    # print("image.size, ", image.size)
    draw = ImageDraw.Draw(image)

    for page in result.pages: # result.pages.lines.polygon.x, .y
        for line in page.lines:
            if line.polygon:
                points = [(point.x, point.y) for point in line.polygon]
                points.append(points[0])  # 最初の点に戻って閉じる
                draw.line(points, fill="red", width=2)

    # OCR結果を保存
    timestamp = create_filename_with_timestamp()
    image.save(os.path.join(state['out_dir'], 'out_img_{}_{}'.format(timestamp, args.img_path.rsplit('/', 1)[1])))

    return state

def match_textBbox_to_detectionResult(state: OCRDetectionState) -> OCRDetectionState:
    """
    match text bounding box to correspond detection result
    1) 
    """
    """OCR結果とCOCOフォーマットを統合"""
    image_id_to_annotations = {img['id']: [] for img in state['detection_result']['images']}
    
    # OCRを bbox 形式に変換
    ocr_boxes = [
        (text, convert_polygon_to_bbox(coords))
        for text, coords in state['text_and_bboxes']
    ]
    print("ocr_boxes, ", ocr_boxes)

    for annotation in  state['detection_result']['annotations']:
        image_id = annotation['image_id']
        obj_bbox = annotation['bbox']
        
        matched_texts = []
        for text, text_bbox in ocr_boxes:
            iou = calculate_iou(obj_bbox, text_bbox)
            if iou >= state['detection_ocr_iou_threshold']:
                matched_texts.append(text)
        
        # アノテーションにtextを追加（複数の場合は空白区切り）
        if matched_texts:
            annotation['text'] = " ".join(matched_texts)
        else:
            annotation['text'] = ""
        
        image_id_to_annotations[image_id].append(annotation)
    
    # 元の構造に戻す
    merged_data = {
        "images": state["detection_result"]["images"],
        "annotations": sum(image_id_to_annotations.values(), [])
    }
    state['detection_ocr_result'] = merged_data
    return state


def get_result(state: OCRDetectionState) -> OCRDetectionState:
    result = state['result']
    text_and_bboxes = []

    for page in result.pages:
        for line in page.lines:
            if line.polygon:
                points = [(point.x, point.y) for point in line.polygon]
                text_and_bboxes.append((line.content, points))

    # stateに追加して戻す（必要であれば）
    state['text_and_bboxes'] = text_and_bboxes
    print("text_and_bboxes, ", text_and_bboxes)

    return state

def main(args):
    load_dotenv()

    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    print("env finish")
    documentAnalysisClient1 = DocumentAnalysisClient(
        endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
    )

    builder = StateGraph(OCRDetectionState)
    builder.add_node("RunOCR", run_azure_ocr)
    if args.process_name == 'debug_out_image': # debug. printout bbox of text to image
        builder.add_node("GetResult", print_result)
    else:
        builder.add_node("GetResult", get_result)
    builder.add_node("MergeOCRDetection", match_textBbox_to_detectionResult)
    builder.set_entry_point("RunOCR")
    builder.add_edge("RunOCR", "GetResult")
    builder.add_edge("GetResult", "MergeOCRDetection")
    builder.add_edge("MergeOCRDetection", END)

    graph = builder.compile()

    # run langgraph
    with open('../json/instances_custom.json', 'r') as f:
        data_dict = json.load(f)
    input_state = {"image_path": args.img_path, "extracted_text": None, "result": None, "document_analysis_client": documentAnalysisClient1,
    "out_dir": args.output_dir, "text_and_bboxes": None, "detection_result": data_dict, "detection_ocr_result": None,
     "image_num":int(args.img_path.split('flowchart-example', 1)[1].split('.', 1)[0]), "prompt": None, "llm_result": None,
    "detection_ocr_iou_threshold": 0.5}
    final_state = graph.invoke(input_state)

    
def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_path', '-igp', type=str, default="../images/flowchart-example166.png", help='image path')
    parser.add_argument('--output_dir', '-od', type=str, default='../output/', help='path to output directory')
    parser.add_argument('--ocr_precision', '-orp', type=str, default='low', help='low: vision/v3.2/ocr, high: vision/v3.2/read/analyze')

    return parser.parse_args()


if __name__ == '__main__':
    """
    usage)
    python ocr_nodes.py --img_path ../images/flowchart-example163.png
    """
    args = parser()
    main(args)