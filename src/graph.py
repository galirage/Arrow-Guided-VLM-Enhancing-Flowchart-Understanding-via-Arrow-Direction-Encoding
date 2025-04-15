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
import openai as openai_vanila
import base64
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import pandas as pd
import json
from math import sqrt
from typing import List, Dict, Tuple
from llm_nodes import ask_to_llm_with_detection_result, ask_to_llm, print_result
from ocr_nodes import build_flowchart_graph, run_azure_ocr, match_textBbox_to_detectionResult, get_result
# from graphviz import Digraph


def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_num', '-ign', type=int, default=163, help='image number')
    parser.add_argument('--img_path', '-igp', type=str, default="../images/flowchart-example163.png", help='image path')
    parser.add_argument('--output_dir', '-od', type=str, default='output/', help='path to output directory')
    return parser.parse_args()


def visualize_langgraph(builder, filename: str = "langgraph", format: str = "png"):
    dot = Digraph(comment="LangGraph State Graph", format=format)

    # ノード追加
    for node in builder.nodes:
        dot.node(node)

    # エッジ追加
    for from_node, to_node in builder.graph_structure:
        if to_node == END:
            dot.node("END", shape="doublecircle")
            dot.edge(from_node, "END")
        else:
            dot.edge(from_node, to_node)

    # エントリーポイント強調
    if builder.entry_point:
        dot.node(builder.entry_point, shape="box", style="filled", color="lightblue")

    # グラフ出力
    output_path = dot.render(filename, view=True)
    print(f"LangGraph visualized and saved to {output_path}")

def main(args):
    # set env
    load_dotenv()

    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    print("env finish")
    documentAnalysisClient1 = DocumentAnalysisClient(
        endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
    )

    builder = StateGraph(OCRDetectionState)
    # OCR nodes
    builder.add_node("RunOCR", run_azure_ocr)
    builder.add_node("GetResult", get_result)
    builder.add_node("MergeOCRDetection", match_textBbox_to_detectionResult)
    builder.add_node("BuildDirectedGraph", build_flowchart_graph)
    # LLM nodes
    builder.add_node("ASKLLM", ask_to_llm)
    builder.add_node("ShowResult", print_result)
    # graph
    builder.set_entry_point("RunOCR")
    builder.add_edge("RunOCR", "GetResult")
    builder.add_edge("GetResult", "MergeOCRDetection")
    builder.add_edge("MergeOCRDetection", "BuildDirectedGraph")
    builder.add_edge("BuildDirectedGraph", "ASKLLM")
    builder.add_edge("ASKLLM", "ShowResult")
    builder.add_edge("ShowResult", END)

    graph = builder.compile()
    # visualize_langgraph_from_graph(graph)
    # visualize_langgraph(builder)

    json_path = args.img_path.replace('images', 'json').rsplit('.', 1)[0] + '.json'
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    input_state = {"image_path": args.img_path, "extracted_text": None, "result": None, "document_analysis_client": documentAnalysisClient1,
        "out_dir": args.output_dir, "text_and_bboxes": None, "detection_result": data_dict, "detection_ocr_result": None,
        "image_num":int(args.img_path.split('flowchart-example', 1)[1].split('.', 1)[0]), "prompt": None, "llm_result": None,
        "detection_ocr_match_threshold": 0.5, 'object_categories':{3: "terminator", 4: "data", 5: "process", 6: "decision", 7: "connection"},
        "arrow_category": 2, "arrow_start": 8, "arrow_end": 9, "directed_graph_text":None}

    # run langgraph
    final_state = graph.invoke(input_state)

if __name__ == '__main__':
    """
    usage)
    python graph.py --process_name main --img_path ../images/flowchart-example163.png

    """
    args = parser()
    if args.process_name == 'main':
        main(args)
