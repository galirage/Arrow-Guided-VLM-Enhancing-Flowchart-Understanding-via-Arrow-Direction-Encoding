import requests
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import argparse
import os
import datetime

def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--img_path', '-igp', type=str, default="../images/flowchart-example166.png", help='image path')
    parser.add_argument('--output_dir', '-od', type=str, default='../output/', help='path to output directory')
    parser.add_argument('--ocr_precision', '-orp', type=str, default='low', help='low: vision/v3.2/ocr, high: vision/v3.2/read/analyze')

    return parser.parse_args()

def create_filename_with_timestamp() -> str:
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")  # 14桁のタイムスタンプを作成
    time_str = f"{timestamp}"
    return time_str

def response_low_precision(args, ocr_url, headers):
    # 画像の読み込みとリクエスト送信
    with open(args.img_path, "rb") as image_file:
        response = requests.post(ocr_url, headers=headers, data=image_file)
    print("type(response) ", type(response))
    # APIのレスポンス解析
    response.raise_for_status()
    result = response.json()
    print("get result !!")
    print("type(result) ", type(result))
    print("result.keys(), ", result.keys())

    # テキストと座標情報を取得
    extracted_text = []
    image = Image.open(args.img_path)
    # print("image.size, ", image.size)
    draw = ImageDraw.Draw(image)

    for region in result.get("regions", []):
        for line in region.get("lines", []):
            line_text = " ".join([word["text"] for word in line["words"]])
            extracted_text.append(line_text)

            # 座標を取得して枠を描画
            for word in line["words"]:
                bbox = list(map(int, word["boundingBox"].split(",")))
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                # print(f"(x, y):({x}, {y}). (w, h):({w}, {h}). word:{word}")
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

    # OCR結果を保存
    timestamp = create_filename_with_timestamp()
    image.save(os.path.join(args.output_dir, 'out_img_{}_{}'.format(timestamp, args.img_path.rsplit('/', 1)[1])))

    # OCR結果のテキストを表示
    print("Extracted Text:")
    print("\n".join(extracted_text))
    return image, result

def response_high_precision(args, ocr_url, headers):
    with open(args.img_path, "rb") as image_file:
        response = requests.post(ocr_url, headers=headers, data=image_file)
    # APIのレスポンス解析
    response.raise_for_status()
    operation_location = response.headers.get("Operation-Location")
    if not operation_location:
        raise Exception("Operation-Location header not found.")
    
    # OCR結果の取得
    headers.pop("Content-Type")
    while True:
        result_response = requests.get(operation_location, headers=headers)
        result_response.raise_for_status()
        result = result_response.json()
        if result.get("status") in ["succeeded", "failed"]:
            break
    if result.get("status") != "succeeded":
        raise Exception(f"OCR failed: {result}")
    print("get result !!")
    print("type(result) ", type(result))

    # テキストと座標情報を取得
    extracted_text = []
    image = Image.open(args.img_path)
    draw = ImageDraw.Draw(image)

    if args.ocr_precision == 'low':
        # low precision の場合の処理
        if result and "regions" in result:
          for region in result["regions"]:
              for line in region["lines"]:
                  text = " ".join([word["text"] for word in line["words"]])
                  extracted_text.append(text)
                  # 座標情報の描画
                  for word in line["words"]:
                      x, y, w, h = word["boundingBox"].split(",")
                      draw.rectangle([(int(x), int(y)), (int(x) + int(w), int(y) + int(h))], outline="red", width=2)
    elif args.ocr_precision == 'high':
      #high precision の場合の処理
        if result and "analyzeResult" in result and "lines" in result["analyzeResult"]:
          for line in result["analyzeResult"]["lines"]:
            extracted_text.append(line["text"])
            # 座標情報の描画
            polygon = line["boundingBox"]
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, polygon)
            draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline="blue", width=2)

    print("Extracted Text:", extracted_text)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save("output.jpg")

def main(args:argparse.Namespace) -> None:
    # .envファイルの読み込み
    load_dotenv()

    CV_ENDPOINT = os.getenv("CV_ENDPOINT")
    CV_API_KEY = os.getenv("CV_API_KEY")

    # APIリクエストヘッダー
    headers = {
    "Ocp-Apim-Subscription-Key": CV_API_KEY,
    "Content-Type": "application/octet-stream"
    }

    # APIのエンドポイント (OCR)
    if args.ocr_precision == 'low':
        ocr_url = CV_ENDPOINT + "vision/v3.2/ocr"
        response_low_precision(args, ocr_url, headers)
    elif args.ocr_precision == 'high':
        ocr_url = CV_ENDPOINT + "vision/v3.2/read/analyze"
        response_high_precision(args,ocr_url, headers)
    
    


if __name__ == '__main__':
    """
    usage)
    python get_text_using_ocr.py --img_path ../images/flowchart-example163.png --ocr_precision low
    """
    args = parser()
    main(args)