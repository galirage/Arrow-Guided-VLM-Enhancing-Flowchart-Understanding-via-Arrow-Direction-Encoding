import json
import os
import argparse
import xml.etree.ElementTree as ET


def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='devide_anno', help='process name')
    parser.add_argument('--img_num', '-ign', type=int, default=163, help='image number')
    parser.add_argument('--xml_path', '-xp', type=str, default="../images/flowchart-example163.xml", help='image path')
    parser.add_argument('--output_dir', '-od', type=str, default='output/', help='path to output directory')
    return parser.parse_args()


def parse_voc_xml(xml_path, category_mapping):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    file_name = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    annotations = []
    ann_id = 1  # ファイルごとにリセット
    image_id = 1  # 各ファイル内では常に 1

    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if name not in category_mapping:
            continue
        category_id = category_mapping[name]

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        w = xmax - xmin
        h = ymax - ymin
        area = w * h

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [xmin, ymin, w, h],
            "area": area,
            "segmentation": [],
            "iscrowd": 0
        })
        ann_id += 1

    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

    return image_info, annotations

def convert_voc_folder_to_coco(voc_folder: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    category_mapping = {
        "text": 1,
        "arrow": 2,
        "terminator": 3,
        "data": 4,
        "process": 5,
        "decision": 6,
        "connection": 7,
        "arrow_start": 8,
        "arrow_end": 9
    }

    categories = [{"id": id_, "name": name} for name, id_ in category_mapping.items()]

    for file in os.listdir(voc_folder):
        if not file.endswith('.xml'):
            continue

        xml_path = os.path.join(voc_folder, file)
        image_info, anns = parse_voc_xml(xml_path, category_mapping)

        # ここで毎回初期化（重要！）
        coco = {
            "images": [image_info],
            "annotations": anns,
            "categories": categories
        }

        output_filename = os.path.splitext(file)[0] + '.json'
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

        print(f" Saved: {output_path}")


def split_coco_by_image(coco_data: dict, output_dir: str):
    # os.makedirs(output_dir, exist_ok=True)

    # 画像ごとに分割処理
    for image_info in coco_data["images"]:
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        base_name = os.path.splitext(file_name)[0]  # "flowchart-example001"
        output_path = os.path.join(output_dir, f"{base_name}.json")

        # 対応するアノテーションのみ抽出
        annotations_for_image = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == image_id
        ]

        # 構造を保存
        image_coco_data = {
            "images": [image_info],
            "annotations": annotations_for_image
        }

        # JSONとして保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(image_coco_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {output_path}")

if __name__ == '__main__':
    """
    usage)
    python ocr_nodes.py --img_path ../images/flowchart-example163.png
    """
    args = parser()
    
    if args.process_name == "devide_anno":
        # instances_custom.jsonにはarrow_start, arrow_endがないので機能しない！！
        # 全ての画像の情報を含んだcoco形式のinstances_custom.jsonに対し、
        # ここの画像ごとに分けて保存する
        with open("../json/instances_custom.json", "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        split_coco_by_image(coco_data, output_dir="../json")

    elif args.process_name == "pascal2coco":
        convert_voc_folder_to_coco(
            voc_folder='../xml/',
            output_dir='../json/'
        )

