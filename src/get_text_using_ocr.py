import requests
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import argparse
import os

def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_path', '-igp', type=str, default="../images/flowchart-example166.png", help='image path')
    parser.add_argument('--output_dir', '-od', type=str, default='../output/', help='path to output directory')
    return parser.parse_args()



def main(args:argparse.Namespace) -> None:
    # .envファイルの読み込み
    load_dotenv()

    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")

    # 画像ファイルのパス
    image_path = args.img_path

    # APIリクエストヘッダー
    headers = {
    "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
    "Content-Type": "application/octet-stream"
    }

    # APIのエンドポイント (OCR)
    ocr_url = AZURE_ENDPOINT + "vision/v3.2/ocr"

    # 画像の読み込みとリクエスト送信
    with open(image_path, "rb") as image_file:
        response = requests.post(ocr_url, headers=headers, data=image_file)

    # APIのレスポンス解析
    response.raise_for_status()
    result = response.json()
    print("get result !!")
    print("type(result) ", type(result))

    # テキストと座標情報を取得
    extracted_text = []
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for region in result.get("regions", []):
        for line in region.get("lines", []):
            line_text = " ".join([word["text"] for word in line["words"]])
            extracted_text.append(line_text)

            # 座標を取得して枠を描画
            for word in line["words"]:
                bbox = list(map(int, word["boundingBox"].split(",")))
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    # OCR結果を保存
    image.save(os.path.join(args.output_dir, 'out_img_{}'.format(args.img_path.rsplit('/', 1)[1])))

    # OCR結果のテキストを表示
    print("Extracted Text:")
    print("\n".join(extracted_text))


if __name__ == '__main__':
    """
    usage)
    python get_text_using_ocr.py --img_path ../images/flowchart-example163.png
    """
    args = parser()
    main(args)