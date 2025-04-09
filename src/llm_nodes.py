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

def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_num', '-ign', type=int, default=163, help='image number')
    parser.add_argument('--img_path', '-igp', type=str, default="../images/flowchart-example163.png", help='image path')
    parser.add_argument('--output_dir', '-od', type=str, default='output/', help='path to output directory')
    return parser.parse_args()

def encode_image_to_base64(image_path:str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8") 

def print_result(state: OCRDetectionState) -> OCRDetectionState:
    return state

def ask_to_llm(state: OCRDetectionState) -> OCRDetectionState:
    load_dotenv()

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

    image_base64 = encode_image_to_base64(state['image_path'])
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
                        #   '134':  "この画像でLogic & Data ErrorでYESの場合、次のprocessは何ですか？",
                        #   '134':  "この画像でSystem Readyの次のprocessは何ですか？", # -> 「System Ready」の次のプロセスは「Enter Program」です。
                        #   '134':  "この画像でProgram Codeの次のprocessは何ですか？", # -> Program Codeの次のプロセスは「Enter Program」です。
                        #   '134':  "この画像でEnter Programの次のprocessは何ですか？", # -> この画像の「Enter Program」の次のプロセスは「Edit Source」です。
                        #   '134':  "この画像でEdit Sourceの次のprocessは何ですか？", # -> この画像で「Edit Source」の次のプロセスは「Compile Source」です。
                        #   '134':  "この画像でCompile Sourceの次のprocessは何ですか？", # -> 「Compile Source」の次のプロセスは「Syntax Error」です。
                        #   '134':  "この画像でSyntax ErrorでYESの場合、次のprocessは何ですか？", # -> 画像では、"Syntax Error" が "YES" の場合、次のプロセスは "Edit Source" です。
                        #   '134':  "この画像でSyntax ErrorでNOの場合、次のprocessは何ですか？", # -> このフローチャートで、Syntax ErrorがNOの場合、次のプロセスは「Link System Library」です。
                        #   '134':  "この画像でSystem Libraryの次のprocessは何ですか？", # -> System Libraryの次のプロセスは「Link System Library」です。
                        #   '134':  "この画像でLink System Libraryの次のprocessは何ですか？", # -> Link System Libraryの次のプロセスは「Execute System Code」です。
                        #   '134':  "この画像でExecute System Codeyの次のprocessは何ですか？", # -> 画像のフローチャートによれば、「Execute System Code」の次のプロセスは「Logic & Data Error」の判定です。ここで「YES」の場合は「Edit Source」に戻り、「NO」の場合は「Correct Code」に進みます。
                          '134':  "この画像でLogic & Data ErrorでNOの場合、次のprocessは何ですか？", # -> このフローチャートでは、「Logic & Data Error」が「NO」の場合、次のプロセスは「Correct Code」で、その後「STOP」となります

                          '151':  "この画像でPick up items from current inventoryの次のprocessは何ですか？",
                        #   '163': "この画像でGasだと判明した場合、次のprocessはどうなりますか？",
                          '163': "この画像でSystem Updateがなされた場合、次のprocessは何ですか？",
                        #   '163':  "この画像でSystem Updateがなされた場合、次のprocessは何ですか？次の4つから1つ選んで答えてください。1)Sensor Starts Up, 2)Sensor Detects for Gas, 3)Display: 'Gas: SAFE' No Buzzer Green LED, 4) Display: 'Gas:WARNING!' Buzzer Yellow LED.",
                        #   '163':  prompt_with_arrow_163,

                        #   '165': "この画像でPlayer's 2 Turnの次の処理は何ですか？",
                        #   '165': "この画像でDid any player connect three in a rowでYesの場合、どうなりますか？",
                          '165': "この画像でOne player will be X and the Other will be Oの次のprocessは何ですか？",
                        #   '166': "この画像でDesignされたものがTestに通らない場合、どのようなprocessとなりますか？",
                          '166': "この画像でPrototypeでない場合、次のprocessは何ですか？",
                          '187': "この画像で The patient goes back home to check in later の次のprocessは何ですか？",

                        #   '203': "この画像でRepair the coded charging schedule の次はどのようなprocessとなりますか？",
                          '203': "この画像でHave all particle been processedでNoとなるとどのprocessとなりますか？",
                        #   '203': "この画像でRepair the coded charging schedule の次はどのようなprocessとなりますか？次の4つの選択肢から選んでください。1)Calculate velocity vector, 2)Decode the charging schedule, 3)Go to next particle, 4) Move the particle",
                          }

    # message1=[
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": imgNum_prompt_dict['{}'.format(state['image_num'])]},
    #             {
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url": f"data:image/jpeg;base64,{image_base64}"
    #                 },
    #             },
    #         ],
    #     },
    # ]
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

    # response = openai_vanila.chat.completions.create(
    #     model="gpt-4o",
    #     messages=message1,
    #     max_tokens=300,
    # )
    """
    langchain==0.1.14
    langchain-community==0.0.34
    openai==1.65.3
    pydantic==2.6.4
    """

    output = model.invoke(messages)
    state['llm_result'] = output.content
    print("output, ", output)
    print("type(output), ", type(output))
    return state

def main(args):


    builder = StateGraph(OCRDetectionState)
    builder.add_node("ASKLLM", ask_to_llm)
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
    python llm_nodes.py --process_name main --img_path ../images/flowchart-example163.png
    """
    args = parser()
    if args.process_name == 'not_use_langchain':
        not_use_langchain(args)
    else:
        main(args)