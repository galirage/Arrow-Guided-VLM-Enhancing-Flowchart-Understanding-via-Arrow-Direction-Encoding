import csv
from pathlib import Path

from src.evaluator import workflow
from src.schema import EvaluationData, EvaluationDataset, EvaluationResult


def save_results_to_csv(
    results: list[EvaluationResult], filename: str = "evaluation_results.csv"
):
    """評価結果をCSVファイルに保存する関数"""
    if not results:
        print("評価結果がありません。CSVファイルは作成されません。")
        return

    # ディレクトリが存在するか確認し、なければ作成
    file_path = Path(filename)
    if file_path.parent != Path("."):
        file_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(EvaluationResult.model_fields.keys())

    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result.model_dump())
    print(f"評価結果を {filename} に保存しました。")


def calculate_accuracy_percentage(results: list[EvaluationResult]) -> float:
    """Compute accuracy (percentage of correct answers) from evaluation results."""
    if not results:
        return 0.0

    num_correct = sum(1 for r in results if r.is_correct)
    accuracy = (num_correct / len(results)) * 100
    return accuracy


def evaluate_dataset(
    dataset: EvaluationDataset,
) -> list[EvaluationResult]:
    """
    EvaluationDatasetを評価し、結果のリストを返す。

    Args:
        dataset: 評価対象のデータセット。

    Returns:
        評価結果のリスト。
    """
    results: list[EvaluationResult] = []

    for data_item in dataset.items:
        try:
            workflow_output = workflow.invoke(
                input={
                    "question": data_item.question,
                    "image_path": data_item.image_path,
                    "reference_answer": data_item.reference_answer,
                    "model_output": data_item.model_output,
                }
            )

            # workflow_output には image_path が含まれていないため、元のデータから追加
            complete_output = {
                **workflow_output,
                "image_path": data_item.image_path,  # EvaluationResult が必要とするフィールド
                "category": data_item.category,
                "question_type": data_item.question_type,
            }

            result = EvaluationResult(**complete_output)
            results.append(result)
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise e

    accuracy = calculate_accuracy_percentage(results)
    print(
        f"正答率: {accuracy:.2f}% ({len(results)} 件中 {sum(r.is_correct for r in results)} 件正解)"
    )

    return results


def save_model_comparison_to_csv(
    model_results: dict[str, tuple[list[EvaluationResult], float]],
    filename: str = "model_comparison.csv",
):
    """
    複数モデルの評価結果を整形して1つのCSVに保存する関数

    Args:
        model_results: {モデル名: (評価結果リスト, 正答率)} の辞書
        filename: 出力CSVファイル名
    """
    # ディレクトリが存在するか確認し、なければ作成
    file_path = Path(filename)
    if file_path.parent != Path("."):
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # ヘッダー行
        writer.writerow(["Model", "Accuracy (%)", "Correct", "Total", "Error Rate (%)"])

        # 各モデルの結果行
        for model_name, (results, accuracy) in model_results.items():
            correct_count = sum(1 for r in results if r.is_correct)
            total_count = len(results)
            error_rate = 100.0 - accuracy

            writer.writerow([
                model_name,
                f"{accuracy:.2f}",
                correct_count,
                total_count,
                f"{error_rate:.2f}",
            ])

    print(f"モデル比較結果を {filename} に保存しました。")


# --- 実行例 ---
if __name__ == "__main__":
    import json
    from pathlib import Path

    from src.schema.results import QAEvaluationSet

    # スキーマ読み込み
    data = json.loads(
        Path("data/q_a_result_predict_with_categories.json").read_text(encoding="utf-8")
    )
    qa_set = QAEvaluationSet.model_validate(data)

    model_with_no_dec_ocr_data = [
        EvaluationData(
            question=qa_set.root[i].question,
            category=qa_set.root[i].category,
            question_type=qa_set.root[i].question_type,
            image_path=qa_set.root[i].image_path,
            reference_answer=qa_set.root[i].answer_collect,
            model_output=qa_set.root[i].answer_from_llm_with_no_dec_ocr,
        )
        for i in range(len(qa_set.root))
    ]

    model_with_dec_ocr_data = [
        EvaluationData(
            question=qa_set.root[i].question,
            category=qa_set.root[i].category,
            question_type=qa_set.root[i].question_type,
            image_path=qa_set.root[i].image_path,
            reference_answer=qa_set.root[i].answer_collect,
            model_output=qa_set.root[i].answer_from_llm_with_dec_ocr,
        )
        for i in range(len(qa_set.root))
    ]

    # 各モデルの評価実行
    print("Model with no dec ocr の評価を実行中...")
    model_with_no_dec_ocr_results = evaluate_dataset(
        EvaluationDataset(items=model_with_no_dec_ocr_data)
    )
    model_with_no_dec_ocr_accuracy = calculate_accuracy_percentage(
        model_with_no_dec_ocr_results
    )
    save_results_to_csv(
        model_with_no_dec_ocr_results,
        "results/model_with_no_dec_ocr_evaluation_results-001.csv",
    )

    print("\nModel with dec ocr の評価を実行中...")
    model_with_dec_ocr_results = evaluate_dataset(
        EvaluationDataset(items=model_with_dec_ocr_data)
    )
    model_with_dec_ocr_accuracy = calculate_accuracy_percentage(
        model_with_dec_ocr_results
    )
    save_results_to_csv(
        model_with_dec_ocr_results,
        "results/model_with_dec_ocr_evaluation_results-001.csv",
    )

    # モデル比較結果をCSVに保存
    model_comparison = {
        "Model A": (model_with_no_dec_ocr_results, model_with_no_dec_ocr_accuracy),
        "Model B": (model_with_dec_ocr_results, model_with_dec_ocr_accuracy),
    }

    save_model_comparison_to_csv(
        model_comparison, "results/model_comparison_results-001.csv"
    )
