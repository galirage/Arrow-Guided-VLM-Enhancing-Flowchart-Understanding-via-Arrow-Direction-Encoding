from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from dotenv import load_dotenv
import argparse
import os
import datetime
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class OCRState(TypedDict): # define state
    image_path: str
    extracted_text: Optional[str]
    document_analysis_client: DocumentAnalysisClient
    out_dir: str
    result: None

def create_filename_with_timestamp() -> str:
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")  # 14桁のタイムスタンプを作成
    time_str = f"{timestamp}"
    return time_str

def run_azure_ocr(state: OCRState) -> OCRState: # run ocr
    image_path = state["image_path"]
    with open(image_path, "rb") as f:
        poller = state['document_analysis_client'].begin_analyze_document("prebuilt-read", document=f)
        result = poller.result()
        print("type(result) : ", type(result))
    extracted_text = "\n".join([line.content for page in result.pages for line in page.lines])
    return {"image_path": image_path, "extracted_text": extracted_text, "result": result}


def print_result(state: OCRState) -> OCRState: # output ocr
    print("\n 抽出されたテキスト:\n")
    print("state['extracted_text']", state['extracted_text'])
    result = state['result']

    extracted_text = []
    image = Image.open(state['image_path'])
    # print("image.size, ", image.size)
    draw = ImageDraw.Draw(image)

    for page in result.pages:
        for line in page.lines:
            if line.polygon:
                points = [(point.x, point.y) for point in line.polygon]
                points.append(points[0])  # 最初の点に戻って閉じる
                draw.line(points, fill="red", width=2)

    # OCR結果を保存
    timestamp = create_filename_with_timestamp()
    image.save(os.path.join(state['out_dir'], 'out_img_{}_{}'.format(timestamp, args.img_path.rsplit('/', 1)[1])))

    return state

def main(args):
    load_dotenv()

    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    documentAnalysisClient1 = DocumentAnalysisClient(
        endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
    )

    builder = StateGraph(OCRState)
    builder.add_node("RunOCR", run_azure_ocr)
    builder.add_node("ShowResult", print_result)

    builder.set_entry_point("RunOCR")
    builder.add_edge("RunOCR", "ShowResult")
    builder.add_edge("ShowResult", END)

    graph = builder.compile()

    # run langgraph
    input_state = {"image_path": args.img_path, "extracted_text": None, "result": None, "document_analysis_client": documentAnalysisClient1,
    "out_dir": args.output_dir}
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
    python ocr_agent.py --img_path ../images/flowchart-example163.png
    """
    args = parser()
    main(args)