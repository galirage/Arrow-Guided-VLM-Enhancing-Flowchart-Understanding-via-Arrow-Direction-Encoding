import argparse
import glob
import json
import os

from langgraph.graph import END, StateGraph

from .llm_nodes import ask_to_llm
from .ocr_nodes import (
    build_flowchart_graph,
    match_textBbox_to_detectionResult,
    run_azure_ocr,
)
from .state import OCRDetectionState


def parser():
    parser = argparse.ArgumentParser(description="llm_args")
    parser.add_argument(
        "--process_name", "-pn", type=str, default="load_pdf", help="process name"
    )
    parser.add_argument("--img_num", "-ign", type=int, default=163, help="image number")
    parser.add_argument(
        "--img_path",
        "-igp",
        type=str,
        default="../images/flowchart-example163.png",
        help="image path",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        default="output/",
        help="path to output directory",
    )
    parser.add_argument(
        "--img_dir",
        "-igd",
        type=str,
        default="../../images/",
        help="path to directory incuding images",
    )
    return parser.parse_args()


def main(args):
    builder = StateGraph(OCRDetectionState)
    # OCR nodes
    builder.add_node("RunOCR", run_azure_ocr)
    builder.add_node("MergeOCRDetection", match_textBbox_to_detectionResult)
    builder.add_node("BuildDirectedGraph", build_flowchart_graph)
    # LLM nodes
    builder.add_node("ASKLLM", ask_to_llm)
    # graph
    builder.set_entry_point("RunOCR")
    builder.add_edge("RunOCR", "MergeOCRDetection")
    builder.add_edge("MergeOCRDetection", "BuildDirectedGraph")
    builder.add_edge("BuildDirectedGraph", "ASKLLM")
    builder.add_edge("ASKLLM", END)

    graph = builder.compile()

    json_path = args.img_path.replace("images", "json").rsplit(".", 1)[0] + ".json"
    with open(json_path, "r") as f:
        data_dict = json.load(f)
    input_state = {
        "image_path": args.img_path,
        "out_dir": args.output_dir,
        "text_and_bboxes": [],
        "detection_result": data_dict,
        "detection_ocr_result": None,
        "image_num": args.img_path.split("flowchart-example", 1)[1].split(".", 1)[0],
        "prompt": None,
        "llm_result": None,
        "detection_ocr_match_threshold": 0.5,
        "object_categories": {
            3: "terminator",
            4: "data",
            5: "process",
            6: "decision",
            7: "connection",
        },
        "arrow_category": 2,
        "arrow_start": 8,
        "arrow_end": 9,
        "directed_graph_text": None,
    }

    # run langgraph
    final_state = graph.invoke(input_state)
    for key, value in final_state.__dict__.items():
        print("------------------------------------------")
        print(f"{key}: {value}")
    return final_state["llm_result"]  # list[dict]


def all_image(args):
    # img_files = glob.glob('../images_predict/*')
    img_files = glob.glob(os.path.join(args.img_dir, "*"))
    print("img_files")
    print("len(img_files) ", len(img_files))
    all_results = []
    for num, file1 in enumerate(img_files):
        args.img_path = file1
        print("args.img_path: ", args.img_path)
        results1 = main(args)
        for result1 in results1:
            all_results.append(result1)

    with open(
        os.path.join(args.output_dir, "q_a_result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    args = parser()
    if args.process_name == "main":
        results = main(args)
        with open(
            os.path.join(args.output_dir, "q_a_result_017.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f)
    elif args.process_name == "all_image":
        all_image(args)
