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
from state import OCRState

def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_num', '-ign', type=int, default=163, help='image number')
    parser.add_argument('--output_dir', '-od', type=str, default='output/', help='path to output directory')
    return parser.parse_args()

def encode_image_to_base64(image_path:str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8") 

def ask_to_llm(state: OCRState) -> OCRState:
    pass


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

if __name__ == '__main__':
    """
    usage)
    python llm_with_langchain.py --img_num 163
    python llm_with_langchain.py --process_name not_use_langchain --img_num 163
    """
    args = parser()
    if args.process_name == 'not_use_langchain':
        not_use_langchain(args)
    else:
        main(args)