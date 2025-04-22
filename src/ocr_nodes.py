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
from math import sqrt


def _calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    if boxAArea + boxBArea - interArea == 0:
        return 0.0

    return interArea / (boxAArea + boxBArea - interArea)

def _bbox_from_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    x_min, y_min = min(x1, x2), min(y1, y2)
    x_max, y_max = max(x1, x2), max(y1, y2)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def _get_center(bbox):
    """COCO形式 [x, y, w, h] の中心座標を返す"""
    return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

def _euclidean_dist(p1, p2):
    """2点間のユークリッド距離"""
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def _is_near_bbox_edge(point, bbox, margin=20):
    """
    点が bbox の上下左右の辺に margin ピクセル以内で近いかどうかを判定
    """
    x, y, w, h = bbox
    px, py = point

    near_left   = abs(px - x) <= margin and y - margin <= py <= y + h + margin
    near_right  = abs(px - (x + w)) <= margin and y - margin <= py <= y + h + margin
    near_top    = abs(py - y) <= margin and x - margin <= px <= x + w + margin
    near_bottom = abs(py - (y + h)) <= margin and x - margin <= px <= x + w + margin

    return near_left or near_right or near_top or near_bottom


def build_flowchart_graph(state: OCRDetectionState) -> OCRDetectionState:
    # OCRテキスト（位置付き）を取得
    ocr_texts = [
    (text, _convert_polygon_to_bbox(coords))
    for text, coords in state['text_and_bboxes']
]

    objects = {}  # object_num: {type, text, center, bbox, before_object, after_object}
    arrows = []   # [{start_point, end_point}]
    arrow_start_candidates = []  # [(x, y)]
    arrow_end_candidates = []    # [(x, y)]
    arrow_bboxes = []            # arrow自体（category_id=2）

    object_num = 1
    id_to_objnum = {}

    # 1. オブジェクト・矢印情報を分類
    for ann in state['detection_ocr_result']['annotations']:
        cat_id = ann['category_id']
        bbox = ann['bbox']
        center = _get_center(bbox)

        if cat_id in state['object_categories']:
            obj = {
                "type": state['object_categories'][cat_id],
                "text": ann.get('text', ''),
                "center": center,
                "bbox": bbox,
                "object_num": object_num,
                "before_object": [],
                "after_object": [],
            }
            objects[object_num] = obj
            id_to_objnum[ann['id']] = object_num
            object_num += 1
        elif cat_id == state['arrow_start']:
            arrow_start_candidates.append(center)
        elif cat_id == state['arrow_end']:
            arrow_end_candidates.append(center)
        elif cat_id == state['arrow_category']:
            arrow_bboxes.append(bbox)
    print("arrow_start_candidates, ", arrow_start_candidates)
    
    arrow_start_end_margin = 30
    iou_threshold = 0.3
    ARROW_SIZE_THRESHOLD = 2000

    # for bbox in arrow_bboxes:
    #     bbox_area = bbox[2] * bbox[3]

    #     if bbox_area >= ARROW_SIZE_THRESHOLD:
    #         # ----------------------
    #         # 大きな矢印 → IoUベースで最適なstart/endペアを探す
    #         best_iou = 0
    #         best_pair = None

    #         for start_pt in arrow_start_candidates:
    #             for end_pt in arrow_end_candidates:
    #                 start_end_bbox = _bbox_from_two_points(start_pt, end_pt)
    #                 iou = _calculate_iou(bbox, start_end_bbox)

    #                 if iou > best_iou:
    #                     best_iou = iou
    #                     best_pair = (start_pt, end_pt)

    #         if best_pair and best_iou >= iou_threshold:
    #             arrows.append({
    #                 "start": best_pair[0],
    #                 "end": best_pair[1]
    #             })

    #     else:
    #         # ----------------------
    #         # 小さな矢印 → bboxの縁に近いstart/endを1個ずつ探す（従来方式）
    #         matched_start = None
    #         matched_end = None

    #         for pt in arrow_start_candidates:
    #             if _is_near_bbox_edge(pt, bbox, margin=arrow_start_end_margin):
    #                 matched_start = pt
    #                 break

    #         for pt in arrow_end_candidates:
    #             if _is_near_bbox_edge(pt, bbox, margin=arrow_start_end_margin):
    #                 matched_end = pt
    #                 break

    #         if matched_start and matched_end:
    #             arrows.append({
    #                 "start": matched_start,
    #                 "end": matched_end
    #             })
    for bbox in arrow_bboxes:
        bbox_area = bbox[2] * bbox[3]

        matched_start = None
        matched_end = None

        if bbox_area >= ARROW_SIZE_THRESHOLD:
            # 大きな矢印 → IoUベースで最適なstart/endペアを探す
            best_iou = 0
            best_pair = None
            for start_pt in arrow_start_candidates:
                for end_pt in arrow_end_candidates:
                    start_end_bbox = _bbox_from_two_points(start_pt, end_pt)
                    iou = _calculate_iou(bbox, start_end_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_pair = (start_pt, end_pt)

            if best_pair and best_iou >= iou_threshold:
                matched_start, matched_end = best_pair

        else:
            # 小さな矢印 → bboxの縁に近いstart/endを1個ずつ探す
            for pt in arrow_start_candidates:
                if _is_near_bbox_edge(pt, bbox, margin=arrow_start_end_margin):
                    matched_start = pt
                    break
            for pt in arrow_end_candidates:
                if _is_near_bbox_edge(pt, bbox, margin=arrow_start_end_margin):
                    matched_end = pt
                    break

        if matched_start and matched_end:
            # 矢印に関連するテキスト（yes/noなど）をマッチング（IoUで判定）
            # label = None
            # best_label_iou = 0.0
            # for text, text_bbox in ocr_texts:
            #     iou = _calculate_iou(bbox, text_bbox)
            #     if iou > 0.3 and iou > best_label_iou:
            #         label = text
            #         best_label_iou = iou
            # 矢印に関連するテキスト（yes/noなど）をマッチング：bboxの縁の近くにあるもの
            label = None
            for text, text_bbox in ocr_texts:
                if _is_near_bbox_edge(_get_center(text_bbox), bbox, margin=20):
                    label = text
                    break  # 最初に見つかったものを採用（必要に応じて変更可）

            arrows.append({
                "start": matched_start,
                "end": matched_end,
                "label": label
            })

    # 3. 矢印の始点・終点がどの object の bbox edge に近いかを調べる
    for arrow in arrows:
        start_pt = arrow["start"]
        end_pt = arrow["end"]
        label = arrow.get("label")
        start_obj = None
        end_obj = None

        for obj in objects.values():
            if start_obj is None and _is_near_bbox_edge(start_pt, obj['bbox']):
                start_obj = obj

        # end_obj 探索時に start_obj と同じ object_num を除外する
        for obj in objects.values():
            if end_obj is None and _is_near_bbox_edge(end_pt, obj['bbox']):
                if start_obj is not None and obj['object_num'] == start_obj['object_num']:
                    continue  # 自己ループを避ける
                end_obj = obj

        if start_obj and end_obj:
            # start_obj['after_object'].append((end_obj['object_num'], label))
            # end_obj['before_object'].append((start_obj['object_num'], label))
            if start_obj['object_num'] != end_obj['object_num']:
                # start_obj が decision のときのみ label を記録
                if start_obj['type'].lower() == "decision":
                    start_obj['after_object'].append((end_obj['object_num'], label))
                    end_obj['before_object'].append((start_obj['object_num'], label))
                else:
                    # 他のタイプの矢印はラベルなしで繋ぐ
                    start_obj['after_object'].append((end_obj['object_num'], None))
                    end_obj['before_object'].append((start_obj['object_num'], None))

    # 4. LLMに渡す形式のテキスト出力を生成
    directed_graph = _format_for_llm(objects)
    print("-------------- directed graph -----------------")
    print(directed_graph)

    state["directed_graph_text"] = directed_graph
    return state


# def _format_for_llm(objects: Dict[int, dict]) -> str:
#     lines = []
#     for obj in objects.values():
#         line = f"type: {obj['type']}, text: {obj['text']}, object_num: {obj['object_num']}"
#         if obj['before_object']:
#             befores = ', '.join(f"object_num: {objects[num]['object_num']}, type: {objects[num]['type']}, text: {objects[num]['text']}" for num in obj['before_object'])
#             line += f"\n    before_object: {befores}"
#         else:
#             line += f"\n    before_object: none"
#         if obj['after_object']:
#             afters = ', '.join(f"object_num: {objects[num]['object_num']}, type: {objects[num]['type']}, text: {objects[num]['text']}" for num in obj['after_object'])
#             line += f"\n    after_object: {afters}"
#         else:
#             line += f"\n    after_object: none"
#         lines.append(line)
#     return '\n\n'.join(lines)
def _format_for_llm(objects: Dict[int, dict]) -> str:
    lines = []
    for obj in objects.values():
        line = f"type: {obj['type']}, text: {obj['text']}, object_num: {obj['object_num']}"

        if obj['before_object']:
            befores = ', '.join(
                f"object_num: {objects[num]['object_num']}, type: {objects[num]['type']}, text: {objects[num]['text']}" +
                (f", label: {label}" if label else "")
                for num, label in obj['before_object']
            )
            line += f"\n    before_object: {befores}"
        else:
            line += f"\n    before_object: none"

        if obj['after_object']:
            afters = ', '.join(
                f"object_num: {objects[num]['object_num']}, type: {objects[num]['type']}, text: {objects[num]['text']}" +
                (f", label: {label}" if label else "")
                for num, label in obj['after_object']
            )
            line += f"\n    after_object: {afters}"
        else:
            line += f"\n    after_object: none"

        lines.append(line)
    return '\n\n'.join(lines)


def _convert_polygon_to_bbox(polygon: List[Tuple[float, float]]) -> List[float]:
    """4点のpolygonを[x, y, w, h]に変換"""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _calculate_containment_ratio(boxA: List[float], boxB: List[float]) -> float:
    """
    boxB（小さい方、textのbbox）がどれだけboxA（大きい方、object）に含まれているかを返す。
    0〜1のスコア。1に近いほど、boxBはboxAにしっかり含まれている。
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxB_area = boxB[2] * boxB[3]

    if boxB_area == 0:
        return 0.0

    return inter_area / boxB_area


def _create_filename_with_timestamp() -> str:
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
    timestamp = _create_filename_with_timestamp()
    image.save(os.path.join(state['out_dir'], 'out_img_{}_{}'.format(timestamp, args.img_path.rsplit('/', 1)[1])))

    return state


def match_textBbox_to_detectionResult(state: OCRDetectionState) -> OCRDetectionState:
    """
    Match text bounding boxes (from OCR) to detection results (in COCO format).
    Only valid categories (3–7) receive text assignments, but all annotations (including arrows) are kept.
    """
    valid_category_ids = {3, 4, 5, 6, 7}  # 3:terminator, ..., 7:connection

    image_id_to_annotations = {img['id']: [] for img in state['detection_result']['images']}

    # OCRのpolygonを bbox に変換
    ocr_boxes = [
        (text, _convert_polygon_to_bbox(coords))
        for text, coords in state['text_and_bboxes']
    ]

    for annotation in state['detection_result']['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        obj_bbox = annotation['bbox']

        # 対象カテゴリ（3〜7）のみに対してテキストマッチング
        if category_id in valid_category_ids:
            matched_texts = []
            for text, text_bbox in ocr_boxes:
                score = _calculate_containment_ratio(obj_bbox, text_bbox)
                if score >= state['detection_ocr_match_threshold']:
                    matched_texts.append(text)
                    print(f"matched!!! obj_bbox: {obj_bbox}, text_bbox: {text_bbox}, score: {score:.3f}")

            # テキストを空白で結合して追加
            annotation['text'] = " ".join(matched_texts) if matched_texts else ""
        
        # 他のカテゴリ（1, 2, 8, 9）はそのまま text を付けずに残す
        image_id_to_annotations[image_id].append(annotation)

    # 全データを構築
    merged_data = {
        "images": state["detection_result"]["images"],
        "annotations": sum(image_id_to_annotations.values(), [])
    }
    state['detection_ocr_result'] = merged_data
    print("merged_data: ", merged_data)
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
    builder.add_node("BuildDirectedGraph", build_flowchart_graph)

    builder.set_entry_point("RunOCR")
    builder.add_edge("RunOCR", "GetResult")
    builder.add_edge("GetResult", "MergeOCRDetection")
    builder.add_edge("MergeOCRDetection", "BuildDirectedGraph")
    builder.add_edge("BuildDirectedGraph", END)

    graph = builder.compile()

    # run langgraph
    json_path = args.img_path.replace('images', 'json').rsplit('.', 1)[0] + '.json'
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    input_state = {"image_path": args.img_path, "extracted_text": None, "result": None, "document_analysis_client": documentAnalysisClient1,
        "out_dir": args.output_dir, "text_and_bboxes": None, "detection_result": data_dict, "detection_ocr_result": None,
        "image_num":int(args.img_path.split('flowchart-example', 1)[1].split('.', 1)[0]), "prompt": None, "llm_result": None,
        "detection_ocr_match_threshold": 0.5, 'object_categories':{3: "terminator", 4: "data", 5: "process", 6: "decision", 7: "connection"},
        "arrow_category": 2, "arrow_start": 8, "arrow_end": 9}
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