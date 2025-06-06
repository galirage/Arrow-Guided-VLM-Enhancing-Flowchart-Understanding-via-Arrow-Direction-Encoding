import argparse
import base64
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from .configuration import ArrowGuidedVLMConfiguration
from .prompts import (
    common_system_prompt,
    dec_ocr_human_prompt,
    original_human_prompt,
)
from .state import OCRDetectionState
from .utils import init_model


def parser():
    parser = argparse.ArgumentParser(description="lllm_args")
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
    return parser.parse_args()


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _get_question(image_num: int) -> list[dict[str, str]]:
    """
    Get the question and answer for the given image number.
    """
    df = pd.read_csv("csv/source_of_FC_questions.csv")
    df.info()
    print("df.head(10)")
    print(df.head(10))

    df_num1 = df[df["img_file_name"] == image_num]
    print("df['img_file_name'].unique(), ", df["img_file_name"].unique())
    assert len(df_num1) == 3, "len(df_num1) != 3. len(df_num1) == {}".format(
        len(df_num1)
    )

    question_answers = []
    for _, row in df_num1.iterrows():
        if row["question_type"] == 1:
            question = (
                "In this flowchart diagram, what is the next step after '{}'?".format(
                    row["q_ref_step1"]
                )
            )
            answer = "The next step after '{}' is '{}'.".format(
                row["q_ref_step1"], row["answer_ref_step"]
            )
            question_answers.append(
                {
                    "question_type": row["question_type"],
                    "question": question,
                    "answer_collect": answer,
                }
            )
        elif row["question_type"] == 2:
            question = "In this flowchart diagram, if '{}' is '{}', what is the next step?".format(
                row["q_ref_step1"], row["q_ref_yes_no"]
            )
            answer = "If '{}' is '{}', the next step is '{}'.".format(
                row["q_ref_step1"], row["q_ref_yes_no"], row["answer_ref_step"]
            )
            question_answers.append(
                {
                    "question_type": row["question_type"],
                    "question": question,
                    "answer_collect": answer,
                }
            )
        else:  # 3
            question = "In the flowchart diagram, which of the steps before an object '{}' except '{}'?".format(
                row["q_ref_step1"], row["q_ref_step2"]
            )
            answer = "The step before '{}' except '{}' is '{}'.".format(
                row["q_ref_step1"], row["q_ref_step2"], row["answer_ref_step"]
            )
            question_answers.append(
                {
                    "question_type": row["question_type"],
                    "question": question,
                    "answer_collect": answer,
                }
            )
    return question_answers


def ask_to_llm(state: OCRDetectionState, config: RunnableConfig) -> dict[str, Any]:
    # get configuration
    configuration = ArrowGuidedVLMConfiguration.from_runnable_config(config)

    model = init_model(configuration)

    image_base64 = _encode_image_to_base64(state.image_path)
    image_num = int(state.image_path.split("flowchart-example", 1)[1].split(".", 1)[0])

    question_answer_list = _get_question(image_num)
    results = []
    for q_a1 in question_answer_list:
        # 1) with no detection, ocr
        prompt_template = ChatPromptTemplate(
            messages=[
                ("system", common_system_prompt),
                (
                    "user",
                    [
                        {"type": "text", "text": original_human_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                ),
            ],
            input_variables=["question_prompt"],
        )
        chain = prompt_template | model

        output_ori = chain.invoke({"question_prompt": q_a1["question"]})

        # 2) with detection, ocr
        non_ref_text = "(No directed graph text available)"

        prompt_template = ChatPromptTemplate(
            messages=[
                ("system", common_system_prompt),
                (
                    "user",
                    [
                        {"type": "text", "text": dec_ocr_human_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                ),
            ],
            input_variables=["question_prompt", "flow_chart_text"],
        )
        chain = prompt_template | model

        output_dec_ocr = chain.invoke(
            {
                "question_prompt": q_a1["question"],
                "flow_chart_text": state.directed_graph_text
                if state.directed_graph_text
                else non_ref_text,
            }
        )

        results.append(
            {
                "question": q_a1["question"],
                "answer_collect": q_a1["answer_collect"],
                "answer_from_llm_with_no_dec_ocr": output_ori.content,
                "answer_from_llm_with_dec_ocr": output_dec_ocr.content,
            }
        )

    return {
        "llm_result": results,
    }


def main(args):
    builder = StateGraph(OCRDetectionState)
    builder.add_node("ASKLLM", ask_to_llm)

    builder.set_entry_point("ASKLLM")
    builder.add_edge("ASKLLM", END)

    graph = builder.compile()

    # run langgraph
    input_state = {
        "image_path": args.img_path,
        "extracted_text": None,
        "result": None,
        "document_analysis_client": None,
        "out_dir": args.output_dir,
        "text_and_bboxes": None,
        "detection_result": None,
        "detection_ocr_result": None,
        "image_num": int(
            args.img_path.split("flowchart-example", 1)[1].split(".", 1)[0]
        ),
        "prompt": None,
        "llm_result": None,
        "detection_ocr_iou_threshold": 0.5,
    }
    final_state = graph.invoke(input_state)
    print(final_state)


if __name__ == "__main__":
    """
    usage)
    python llm_with_langchain.py --img_num 163
    python llm_nodes.py --process_name ocr --img_num 163
    python llm_nodes.py --process_name main --img_path ../images/flowchart-example179.png

    """
    args = parser()
    if args.process_name == "main":
        main(args)
    elif args.process_name == "question":
        question_str = _get_question(179)
        print("question_str, ")
        print(question_str)
