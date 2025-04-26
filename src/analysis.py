import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.schema.results import QAEvaluationSet


def load_evaluation_results(filepath: str) -> Dict[str, bool]:
    """
    評価結果CSVファイルを読み込む関数

    Args:
        filepath: CSVファイルのパス

    Returns:
        モデル出力をキー、正誤判定を値とする辞書
    """
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["model_output"]] = row["is_correct"].lower() == "true"
    return results


def analyze_question_types(
    qa_set: QAEvaluationSet,
    ocr_results: Dict[str, bool],
    no_ocr_results: Dict[str, bool],
) -> Dict[int, List[Tuple[str, str, str, str, bool, bool]]]:
    """
    質問タイプごとの回答を分析する関数

    Args:
        qa_set: 評価データセット
        ocr_results: OCRありモデルの評価結果
        no_ocr_results: OCRなしモデルの評価結果

    Returns:
        質問タイプごとの[質問, 正解, OCRあり予測, OCRなし予測, OCRあり正誤, OCRなし正誤]のリスト
    """
    type_results = defaultdict(list)

    for item in qa_set.root:
        # 質問タイプごとに結果を格納
        type_results[item.question_type].append((
            item.question,
            item.answer_collect,
            item.answer_from_llm_with_dec_ocr,
            item.answer_from_llm_with_no_dec_ocr,
            ocr_results.get(item.answer_from_llm_with_dec_ocr, False),
            no_ocr_results.get(item.answer_from_llm_with_no_dec_ocr, False),
        ))

    return type_results


def analyze_differences(
    qa_set: QAEvaluationSet,
    ocr_results: Dict[str, bool],
    no_ocr_results: Dict[str, bool],
) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """
    OCRの有無による回答の差分を分析する関数

    Args:
        qa_set: 評価データセット
        ocr_results: OCRありモデルの評価結果
        no_ocr_results: OCRなしモデルの評価結果

    Returns:
        差分の種類ごとの[質問, 正解, OCRあり予測, OCRなし予測]のリスト
    """
    differences = {
        "both_correct": [],
        "both_wrong": [],
        "only_ocr_correct": [],
        "only_no_ocr_correct": [],
    }

    for item in qa_set.root:
        ocr_correct = ocr_results.get(item.answer_from_llm_with_dec_ocr, False)
        no_ocr_correct = no_ocr_results.get(item.answer_from_llm_with_no_dec_ocr, False)

        if ocr_correct and no_ocr_correct:
            differences["both_correct"].append((
                item.question,
                item.answer_collect,
                item.answer_from_llm_with_dec_ocr,
                item.answer_from_llm_with_no_dec_ocr,
            ))
        elif not ocr_correct and not no_ocr_correct:
            differences["both_wrong"].append((
                item.question,
                item.answer_collect,
                item.answer_from_llm_with_dec_ocr,
                item.answer_from_llm_with_no_dec_ocr,
            ))
        elif ocr_correct and not no_ocr_correct:
            differences["only_ocr_correct"].append((
                item.question,
                item.answer_collect,
                item.answer_from_llm_with_dec_ocr,
                item.answer_from_llm_with_no_dec_ocr,
            ))
        else:
            differences["only_no_ocr_correct"].append((
                item.question,
                item.answer_collect,
                item.answer_from_llm_with_dec_ocr,
                item.answer_from_llm_with_no_dec_ocr,
            ))

    return differences


def print_type_analysis(
    type_results: Dict[int, List[Tuple[str, str, str, str, bool, bool]]],
) -> None:
    """質問タイプごとの分析結果を出力する関数"""
    print("\n=== 質問タイプごとの分析 ===")

    type_descriptions = {
        1: "次のステップを尋ねる質問",
        2: "条件分岐の結果を尋ねる質問",
        3: "特定のオブジェクトの前のステップを尋ねる質問",
    }

    for q_type, results in type_results.items():
        print(
            f"\n質問タイプ {q_type}: {type_descriptions.get(q_type, '不明な質問タイプ')}"
        )
        print(f"質問数: {len(results)}")

        # OCRありの正答率を計算
        ocr_correct = sum(1 for _, _, _, _, is_correct, _ in results if is_correct)
        ocr_accuracy = (ocr_correct / len(results)) * 100

        # OCRなしの正答率を計算
        no_ocr_correct = sum(1 for _, _, _, _, _, is_correct in results if is_correct)
        no_ocr_accuracy = (no_ocr_correct / len(results)) * 100

        print(f"OCRあり正答率: {ocr_accuracy:.2f}%")
        print(f"OCRなし正答率: {no_ocr_accuracy:.2f}%")
        print(f"正答率の差: {ocr_accuracy - no_ocr_accuracy:.2f}%")

        # 誤答の例を表示
        print("\n誤答の例:")
        wrong_answers = [
            (q, c, o1, o2) for q, c, o1, o2, oc, nc in results if not (oc and nc)
        ]
        for i, (question, correct, ocr_pred, no_ocr_pred) in enumerate(
            wrong_answers[:3], 1
        ):
            print(f"\n例{i}:")
            print(f"質問: {question}")
            print(f"正解: {correct}")
            print(f"OCRあり: {ocr_pred}")
            print(f"OCRなし: {no_ocr_pred}")


def print_difference_analysis(
    differences: Dict[str, List[Tuple[str, str, str, str]]],
) -> None:
    """差分の分析結果を出力する関数"""
    print("\n=== OCRの有無による差分の分析 ===")

    total = sum(len(items) for items in differences.values())

    print(f"\n総質問数: {total}")
    for category, items in differences.items():
        percentage = (len(items) / total) * 100
        print(f"\n{category}:")
        print(f"件数: {len(items)} ({percentage:.2f}%)")

        if category in ["only_ocr_correct", "only_no_ocr_correct"]:
            print("\n代表的な例:")
            for i, (question, correct, ocr_pred, no_ocr_pred) in enumerate(
                items[:3], 1
            ):
                print(f"\n例{i}:")
                print(f"質問: {question}")
                print(f"正解: {correct}")
                print(f"OCRあり: {ocr_pred}")
                print(f"OCRなし: {no_ocr_pred}")


def load_categories_from_csv(filepath: str) -> Dict[Tuple[int, int], str]:
    """
    質問と回答事例.csvからカテゴリー情報を読み込む関数

    Args:
        filepath: CSVファイルのパス

    Returns:
        (image_num, question_type)をキー、categoryを値とする辞書
    """
    categories = {}
    print(f"\nReading categories from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (
                    int(row["img_file_name"]),
                    int(row["question_type"]),
                )
                categories[key] = row["category"]
                print(f"Loaded category: {key} -> {row['category']}")
            except (KeyError, ValueError) as e:
                print(f"Error processing row: {row}")
                print(f"Error: {e}")
    print(f"\nTotal categories loaded: {len(categories)}")
    return categories


def merge_categories_with_qa_data(
    qa_data: list, categories: Dict[Tuple[int, int], str]
) -> list:
    """
    QAデータにカテゴリー情報を結合する関数

    Args:
        qa_data: 元のQAデータ
        categories: カテゴリー情報の辞書

    Returns:
        カテゴリー情報が追加されたQAデータ
    """
    print("\nMerging categories with QA data")
    print(f"Total QA items: {len(qa_data)}")
    matched_count = 0

    for item in qa_data:
        key = (
            item["image_num"],
            item["question_type"],
        )
        if key in categories:
            item["category"] = categories[key]
            matched_count += 1
            print(f"Matched category for key: {key} -> {categories[key]}")
        else:
            print(f"No category found for key: {key}")
            # デフォルトのカテゴリーを設定
            item["category"] = "unknown"

    print(f"\nTotal matches: {matched_count} out of {len(qa_data)} items")
    return qa_data


def analyze_by_category(
    qa_set: QAEvaluationSet,
    ocr_results: Dict[str, bool],
    no_ocr_results: Dict[str, bool],
) -> Dict[str, List[Tuple[str, str, str, str, bool, bool]]]:
    """
    カテゴリーごとの回答を分析する関数

    Args:
        qa_set: 評価データセット
        ocr_results: OCRありモデルの評価結果
        no_ocr_results: OCRなしモデルの評価結果

    Returns:
        カテゴリーごとの[質問, 正解, OCRあり予測, OCRなし予測, OCRあり正誤, OCRなし正誤]のリスト
    """
    category_results = defaultdict(list)

    for item in qa_set.root:
        category_results[item.category].append((
            item.question,
            item.answer_collect,
            item.answer_from_llm_with_dec_ocr,
            item.answer_from_llm_with_no_dec_ocr,
            ocr_results.get(item.answer_from_llm_with_dec_ocr, False),
            no_ocr_results.get(item.answer_from_llm_with_no_dec_ocr, False),
        ))

    return category_results


def analyze_category_type_combination(
    qa_set: QAEvaluationSet,
    ocr_results: Dict[str, bool],
    no_ocr_results: Dict[str, bool],
) -> Dict[Tuple[str, int], List[Tuple[str, str, str, str, bool, bool]]]:
    """
    カテゴリーと質問タイプの組み合わせごとの回答を分析する関数

    Args:
        qa_set: 評価データセット
        ocr_results: OCRありモデルの評価結果
        no_ocr_results: OCRなしモデルの評価結果

    Returns:
        (カテゴリー, 質問タイプ)ごとの[質問, 正解, OCRあり予測, OCRなし予測, OCRあり正誤, OCRなし正誤]のリスト
    """
    combination_results = defaultdict(list)

    for item in qa_set.root:
        key = (item.category, item.question_type)
        combination_results[key].append((
            item.question,
            item.answer_collect,
            item.answer_from_llm_with_dec_ocr,
            item.answer_from_llm_with_no_dec_ocr,
            ocr_results.get(item.answer_from_llm_with_dec_ocr, False),
            no_ocr_results.get(item.answer_from_llm_with_no_dec_ocr, False),
        ))

    return combination_results


def print_category_analysis(
    category_results: Dict[str, List[Tuple[str, str, str, str, bool, bool]]],
) -> None:
    """カテゴリーごとの分析結果を出力する関数"""
    print("\n=== カテゴリーごとの分析 ===")

    for category, results in category_results.items():
        print(f"\nカテゴリー: {category}")
        print(f"質問数: {len(results)}")

        # OCRありの正答率を計算
        ocr_correct = sum(1 for _, _, _, _, is_correct, _ in results if is_correct)
        ocr_accuracy = (ocr_correct / len(results)) * 100

        # OCRなしの正答率を計算
        no_ocr_correct = sum(1 for _, _, _, _, _, is_correct in results if is_correct)
        no_ocr_accuracy = (no_ocr_correct / len(results)) * 100

        print(f"OCRあり正答率: {ocr_accuracy:.2f}%")
        print(f"OCRなし正答率: {no_ocr_accuracy:.2f}%")
        print(f"正答率の差: {ocr_accuracy - no_ocr_accuracy:.2f}%")

        # 誤答の例を表示
        print("\n誤答の例:")
        wrong_answers = [
            (q, c, o1, o2) for q, c, o1, o2, oc, nc in results if not (oc and nc)
        ]
        for i, (question, correct, ocr_pred, no_ocr_pred) in enumerate(
            wrong_answers[:2], 1
        ):
            print(f"\n例{i}:")
            print(f"質問: {question}")
            print(f"正解: {correct}")
            print(f"OCRあり: {ocr_pred}")
            print(f"OCRなし: {no_ocr_pred}")


def print_category_type_analysis(
    combination_results: Dict[
        Tuple[str, int], List[Tuple[str, str, str, str, bool, bool]]
    ],
) -> None:
    """カテゴリーと質問タイプの組み合わせごとの分析結果を出力する関数"""
    print("\n=== カテゴリーと質問タイプの組み合わせ分析 ===")

    type_descriptions = {
        1: "次のステップを尋ねる質問",
        2: "条件分岐の結果を尋ねる質問",
        3: "特定のオブジェクトの前のステップを尋ねる質問",
    }

    # 結果を見やすくソート
    sorted_keys = sorted(combination_results.keys(), key=lambda x: (x[0], x[1]))

    for key in sorted_keys:
        category, q_type = key
        results = combination_results[key]

        print(f"\nカテゴリー: {category}")
        print(
            f"質問タイプ: {q_type} ({type_descriptions.get(q_type, '不明な質問タイプ')})"
        )
        print(f"質問数: {len(results)}")

        # OCRありの正答率を計算
        ocr_correct = sum(1 for _, _, _, _, is_correct, _ in results if is_correct)
        ocr_accuracy = (ocr_correct / len(results)) * 100

        # OCRなしの正答率を計算
        no_ocr_correct = sum(1 for _, _, _, _, _, is_correct in results if is_correct)
        no_ocr_accuracy = (no_ocr_correct / len(results)) * 100

        print(f"OCRあり正答率: {ocr_accuracy:.2f}%")
        print(f"OCRなし正答率: {no_ocr_accuracy:.2f}%")
        print(f"正答率の差: {ocr_accuracy - no_ocr_accuracy:.2f}%")


def print_overall_analysis(
    qa_set: QAEvaluationSet,
    ocr_results: Dict[str, bool],
    no_ocr_results: Dict[str, bool],
) -> None:
    """全体の分析結果を出力する関数"""
    print("\n=== 全体の分析結果 ===")

    total = len(qa_set.root)

    # OCRありの正答率を計算
    ocr_correct = sum(
        1
        for item in qa_set.root
        if ocr_results.get(item.answer_from_llm_with_dec_ocr, False)
    )
    ocr_accuracy = (ocr_correct / total) * 100

    # OCRなしの正答率を計算
    no_ocr_correct = sum(
        1
        for item in qa_set.root
        if no_ocr_results.get(item.answer_from_llm_with_no_dec_ocr, False)
    )
    no_ocr_accuracy = (no_ocr_correct / total) * 100

    print(f"\n総質問数: {total}")
    print(f"OCRあり正答数: {ocr_correct}")
    print(f"OCRなし正答数: {no_ocr_correct}")
    print(f"OCRあり正答率: {ocr_accuracy:.2f}%")
    print(f"OCRなし正答率: {no_ocr_accuracy:.2f}%")
    print(f"正答率の差: {ocr_accuracy - no_ocr_accuracy:.2f}%")

    # カテゴリーごとの分布
    print("\nカテゴリーごとの質問数:")
    category_counts = defaultdict(int)
    for item in qa_set.root:
        category_counts[item.category] += 1

    for category, count in sorted(category_counts.items()):
        percentage = (count / total) * 100
        print(f"{category}: {count}件 ({percentage:.2f}%)")

    # 質問タイプごとの分布
    print("\n質問タイプごとの質問数:")
    type_counts = defaultdict(int)
    for item in qa_set.root:
        type_counts[item.question_type] += 1

    type_descriptions = {
        1: "次のステップを尋ねる質問",
        2: "条件分岐の結果を尋ねる質問",
        3: "特定のオブジェクトの前のステップを尋ねる質問",
    }

    for q_type, count in sorted(type_counts.items()):
        percentage = (count / total) * 100
        print(
            f"タイプ{q_type} ({type_descriptions.get(q_type, '不明')}): {count}件 ({percentage:.2f}%)"
        )


def main():
    # データの読み込み
    qa_data = json.loads(
        Path("data/q_a_result_predict.json").read_text(encoding="utf-8")
    )

    # カテゴリー情報の読み込みと結合
    categories = load_categories_from_csv("data/質問と回答事例.csv")
    qa_data_with_categories = merge_categories_with_qa_data(qa_data, categories)

    # 結果の保存
    output_path = "data/q_a_result_predict_with_categories.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data_with_categories, f, ensure_ascii=False, indent=2)

    # 以降の分析のためにQAEvaluationSetを作成
    qa_set = QAEvaluationSet.model_validate(qa_data_with_categories)

    # 評価結果の読み込み
    ocr_results = load_evaluation_results(
        "results/model_with_dec_ocr_evaluation_results.csv"
    )
    no_ocr_results = load_evaluation_results(
        "results/model_with_no_dec_ocr_evaluation_results.csv"
    )

    # 全体の分析
    print_overall_analysis(qa_set, ocr_results, no_ocr_results)

    # 質問タイプごとの分析
    type_results = analyze_question_types(qa_set, ocr_results, no_ocr_results)
    print_type_analysis(type_results)

    # 差分の分析
    differences = analyze_differences(qa_set, ocr_results, no_ocr_results)
    print_difference_analysis(differences)

    # カテゴリーごとの分析
    category_results = analyze_by_category(qa_set, ocr_results, no_ocr_results)
    print_category_analysis(category_results)

    # カテゴリーと質問タイプの組み合わせ分析
    combination_results = analyze_category_type_combination(
        qa_set, ocr_results, no_ocr_results
    )
    print_category_type_analysis(combination_results)


if __name__ == "__main__":
    main()
