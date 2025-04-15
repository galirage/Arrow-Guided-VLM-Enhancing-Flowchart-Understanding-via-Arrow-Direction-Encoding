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
from state import OCRDetectionState
import openai as openai_vanila
import base64
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import pandas as pd
from typing import List, Dict, Tuple

def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_num', '-ign', type=int, default=163, help='image number')
    parser.add_argument('--img_path', '-igp', type=str, default="../images/flowchart-example163.png", help='image path')
    parser.add_argument('--output_dir', '-od', type=str, default='output/', help='path to output directory')
    return parser.parse_args()

def _encode_image_to_base64(image_path:str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8") 

def print_result(state: OCRDetectionState) -> OCRDetectionState:
    return state

def _get_question(image_num:int) -> List[Dict[str, str]]:
    df = pd.read_csv('../csv/source_of_FC_questions.csv')
    df.info()
    print("df.head(10)")
    print(df.head(10))

    df_num1 = df[df['img_file_name']==image_num]
    print("df['img_file_name'].unique(), ", df['img_file_name'].unique())
    assert len(df_num1) == 3, 'len(df_num1) != 3. len(df_num1) == {}'.format(len(df_num1))

    question_answers = []
    for idx, row in df_num1.iterrows():
        if row['question_type'] == 1:
            question = "In this flowchart diagram, what is the next step after '{}'?".format(row['q_ref_step1'])
            answer = "The next step after '{}' is '{}'.".format(row['q_ref_step1'], row['answer_ref_step'])
            question_answers.append({'question':question, 'answer_collect':answer})
        elif row['question_type'] == 2:
            question = ""
            answer = ""
            question_answers.append({'question':question, 'answer_collect':answer})
        else: # 3
            question = "In the flowchart diagram, which of the steps before an object '{}' except '{}'?".format(row['q_ref_step1'], row['q_ref_step2'])
            answer = "The next step after '{}' is '{}'.".format(row['q_ref_step1'], row['answer_ref_step'])
            question_answers.append({'question':question, 'answer_collect':answer})
    return question_answers


def ask_to_llm_with_detection_result(state: OCRDetectionState) -> OCRDetectionState:
    load_dotenv()

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

    image_base64 = _encode_image_to_base64(state['image_path'])
    prompt_with_arrow_163 = """以下のflow-chart図の読み取り結果を参考にして、下記の質問に答えてください。

    ## flow-chart図の読み取り結果
    1)type: terminator, text: Start, object_num: 1, 
            before_object: none,
            after_object: 'type: process, text: Sensor Starts Up' object_num: 2, 
    2)type: process, text: Sensor Starts Up,  object_num: 2, 
            before_object: 'type: terminator, text: Start', object_num: 1, 
            after_object: 'type:process, text: Sensor Ready' object_num: 3, 
    3)type: process, text: Sensor Ready, object_num: 3, 
            before_object: 'type: process, text: Sensor Starts Up', object_num: 2, 
            after_object: 'type:process, text: Sensor Detects for Gas' object_num: 4, 
    4)type: process, text: Sensor Detects for Gas, object_num: 4, 
            before_object: 'type:process, text: Sensor Ready, object_num: 3', 'type: process, text:System Updates, object_num: 15', 
            after_object: 'type:data, text: Data Sent to Arduino object_num: 5'
    5)type: data, text: Data Sent to Arduino, object_num: 5, 
            before_object: 'type: process, text: Sensor Detects for Gas, object_num: 4', 
            after_object: 'dicision, text: Gas?, object_num: 6, 
    6)type: dicision, text: Gas?, object_num: 6, 
            before_object: 'type:data, text: Data Sent to Arduino, object_num: 5',
            after_object: Yes:'type:connection object_num: 7', no: 'type: data, text: Gas < 200, object_num: 8',
    ...
    15)type: process, text:System Updates, object_num: 15, 
            before_object: 'type:connection, object_num: 14',
            after_object: 'type: process, text: Sensor Detects for Gas, object_num: 4'

    ## 質問
    この画像でSystem Updateがなされた場合、次のprocessは何ですか？次の4つから1つ選んで答えてください。1)Sensor Starts Up, 2)Sensor Detects for Gas, 3)Display: 'Gas: SAFE' No Buzzer Green LED, 4) Display: 'Gas:WARNING!' Buzzer Yellow LED.
    """

    imgNum_prompt_dict = {
                          '151':  "この画像でPick up items from current inventoryの次のprocessは何ですか？",
                          '163': "この画像でSystem Updateがなされた場合、次のprocessは何ですか？",
                          '165': "この画像でOne player will be X and the Other will be Oの次のprocessは何ですか？",
                          '166': "この画像でPrototypeでない場合、次のprocessは何ですか？",
                          '187': "この画像で The patient goes back home to check in later の次のprocessは何ですか？",
                          '203': "この画像でHave all particle been processedでNoとなるとどのprocessとなりますか？",
                          }

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=[
            {"type": "text", "text": imgNum_prompt_dict['{}'.format(state['image_num'])]},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ])
    ]
    model = AzureChatOpenAI(
        model="gpt-4o",
        # model="gpt-4o-chat",
        api_version="2025-01-01-preview",
        # api_version="2023-12-01-preview",
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        temperature=0
    )

    output = model.invoke(messages)
    state['llm_result'] = output.content
    print("output, ", output)
    print("type(output), ", type(output))
    return state

def ask_to_llm_only_prompt(state: OCRDetectionState) -> OCRDetectionState:
    load_dotenv()

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

    image_base64 = _encode_image_to_base64(state['image_path'])
    image_num = int(state['image_path'].split('flowchart-example', 1)[1].split('.', 1)[0])
    
    model = AzureChatOpenAI(
        model="gpt-4o",
        # model="gpt-4o-chat",
        api_version="2025-01-01-preview",
        # api_version="2023-12-01-preview",
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        temperature=0
    )
    question_answer_list = _get_question(image_num)
    results = []
    for q_a1 in question_answer_list:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=[
                {"type": "text", "text": q_a1['question']},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ])
        ]

        output = model.invoke(messages)
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print("question: ", q_a1['question'])
        print("-----------------------------------------------------------")
        print("answer_collect: ", q_a1['answer_collect'])
        print("-----------------------------------------------------------")
        print("answer_llm, ", output.content)
        # print("type(output), ", type(output))
        results.append({'question': q_a1['question'],
                        'answer_collect': q_a1['answer_collect'],
                        'answer_from_llm': output.content})
    state['llm_result'] = results

    return state


def ask_to_llm(state: OCRDetectionState) -> OCRDetectionState:
    load_dotenv()

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

    image_base64 = _encode_image_to_base64(state['image_path'])
    image_num = int(state['image_path'].split('flowchart-example', 1)[1].split('.', 1)[0])
    
    model = AzureChatOpenAI(
        model="gpt-4o",
        # model="gpt-4o-chat",
        api_version="2025-01-01-preview",
        # api_version="2023-12-01-preview",
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        temperature=0
    )
    question_answer_list = _get_question(image_num)
    results = []
    for q_a1 in question_answer_list:

        question_prompt = "Please answer the following questions using the following flow-chart diagram. \n\n"
        question_prompt += q_a1['question'] + "\n"
        if state['directed_graph_text'] is not None:
            question_prompt += "Refer to the following flow-chart diagram readings by the detection model. \n"
            question_prompt += state['directed_graph_text']
            print("directed graph is used!!")

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=[
                {"type": "text", "text": question_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ])
        ]

        output = model.invoke(messages)
        print("-----------------------------------------------------------")
        print("-----------------------------------------------------------")
        print("question: ", q_a1['question'])
        print("-----------------------------------------------------------")
        print("answer_collect: ", q_a1['answer_collect'])
        print("-----------------------------------------------------------")
        print("answer_llm, ", output.content)
        # print("type(output), ", type(output))
        results.append({'question': q_a1['question'],
                        'answer_collect': q_a1['answer_collect'],
                        'answer_from_llm': output.content})
    state['llm_result'] = results

    return state

def main(args):
    builder = StateGraph(OCRDetectionState)
    builder.add_node("ASKLLM", ask_to_llm_only_prompt)
    builder.add_node("ShowResult", print_result)

    builder.set_entry_point("ASKLLM")
    builder.add_edge("ASKLLM", "ShowResult")
    builder.add_edge("ShowResult", END)

    graph = builder.compile()

    # run langgraph
    input_state = {"image_path": args.img_path, "extracted_text": None, "result": None, "document_analysis_client": None,
    "out_dir": args.output_dir, "text_and_bboxes": None, "detection_result": None, "detection_ocr_result": None,
     "image_num":int(args.img_path.split('flowchart-example', 1)[1].split('.', 1)[0]), "prompt": None, "llm_result": None,
    "detection_ocr_iou_threshold": 0.5}
    final_state = graph.invoke(input_state)

if __name__ == '__main__':
    """
    usage)
    python llm_with_langchain.py --img_num 163
    python llm_nodes.py --process_name ocr --img_num 163
    python llm_nodes.py --process_name main --img_path ../images/flowchart-example179.png

    """
    args = parser()
    if args.process_name == 'main':
        main(args)
    elif args.process_name == 'question':
        question_str = _get_question(179)
        print("question_str, ")
        print(question_str)