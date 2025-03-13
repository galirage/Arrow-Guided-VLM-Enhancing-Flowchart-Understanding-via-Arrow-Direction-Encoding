# from openai import OpenAI
import os
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import base64
import argparse

def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_num', '-ign', type=int, default=163, help='image number')
    parser.add_argument('--output_dir', '-od', type=str, default='output/', help='path to output directory')
    return parser.parse_args()

def encode_image_to_base64(image_path:str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")    

def main(args:argparse.Namespace) -> None:
    # .envファイルの読み込み
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    langchain_project = os.getenv("LANGCHAIN_PROJECT")

    # 環境変数に設定
    print("openai_api_key ", openai_api_key)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

    # model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    # model = ChatOpenAI(model="gpt-4-vision-preview", temperature=0)
    model = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    image_path = "../images/flowchart-example{}.png".format(args.img_num)
    image_base64 = encode_image_to_base64(image_path) # encode

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
                        #   '163': "この画像でGasだと判明した場合、次のprocessはどうなりますか？",
                        #   '163': "この画像でSystem Updateがなされた場合、次のprocessは何ですか？",
                        #   '163':  "この画像でSystem Updateがなされた場合、次のprocessは何ですか？次の4つから1つ選んで答えてください。1)Sensor Starts Up, 2)Sensor Detects for Gas, 3)Display: 'Gas: SAFE' No Buzzer Green LED, 4) Display: 'Gas:WARNING!' Buzzer Yellow LED.",
                          '163':  prompt_with_arrow_163,

                        #   '165': "この画像でPlayer's 2 Turnの次の処理は何ですか？",
                        #   '165': "この画像でDid any player connect three in a rowでYesの場合、どうなりますか？",
                          '165': "この画像でOne player will be X and the Other will be Oの次のprocessは何ですか？",
                        #   '166': "この画像でDesignされたものがTestに通らない場合、どのようなprocessとなりますか？",
                          '166': "この画像でPrototypeでない場合、次のprocessは何ですか？",
                        #   '203': "この画像でRepair the coded charging schedule の次はどのようなprocessとなりますか？",
                        #   '203': "この画像でHave all particle been processedでNoとなるとどのprocessとなりますか？",
                          '203': "この画像でRepair the coded charging schedule の次はどのようなprocessとなりますか？次の4つの選択肢から選んでください。1)Calculate velocity vector, 2)Decode the charging schedule, 3)Go to next particle, 4) Move the particle",
                          }

    messages = [
        SystemMessage(content="You are a helpful assistant that can analyze images."),
        HumanMessage(
            content=[
                {"type": "text", "text": imgNum_prompt_dict['{}'.format(args.img_num)]}, # 163
                # {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        )
    ]

    # output = model.invoke("自己紹介してください")
    output = model.invoke(messages)

    print("output ", output)

if __name__ == '__main__':
    """
    usage)
    python llm_with_langchain.py --img_num 163
    """
    args = parser()
    main(args)