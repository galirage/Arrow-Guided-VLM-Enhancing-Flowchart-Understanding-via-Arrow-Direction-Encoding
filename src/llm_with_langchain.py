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

    imgNum_prompt_dict = {
                        #   '163': "この画像でGasだと判明した場合、次のprocessはどうなりますか？",
                          '163': "この画像でSystem Updateがなされた場合、次のprocessは何ですか？",
                        #   '166': "この画像でDesignされたものがTestに通らない場合、どのようなprocessとなりますか？",
                          '166': "この画像でPrototypeでない場合、次のprocessは何ですか？",
                        #   '203': "この画像でRepair the coded charging schedule の次はどのようなprocessとなりますか？",
                          '203': "この画像でHave all particle been processedでNoとなるとどのprocessとなりますか？",
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