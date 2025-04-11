import csv

from .evaluator import workflow

# State の変更に合わせてインポートを整理
from .schema import EvaluationData, EvaluationDataset, EvaluationResult


def save_results_to_csv(
    results: list[EvaluationResult], filename: str = "evaluation_results.csv"
):
    """評価結果をCSVファイルに保存する関数"""
    if not results:
        print("評価結果がありません。CSVファイルは作成されません。")
        return

    # EvaluationResultの全フィールド名を取得 (順序はモデル定義依存)
    fieldnames = list(EvaluationResult.model_fields.keys())

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            # 全フィールドのデータを書き込む
            writer.writerow(result.model_dump())
    print(f"評価結果を {filename} に保存しました。")


def evaluate_dataset(
    dataset: EvaluationDataset,
):
    """
    EvaluationDatasetを評価し、結果をCSVに保存するパイプライン関数。

    Args:
        dataset: 評価対象のデータセット。

    Returns:
        評価結果のリスト。
    """
    results: list[EvaluationResult] = []

    for data_item in dataset.items:
        try:
            # workflow は State に準拠した辞書を返す想定
            workflow_output = workflow.invoke(
                input={
                    "question": data_item.question,
                    "image": data_item.image,
                    "reference_answer": data_item.reference_answer,
                    "model_output": data_item.model_output,
                }
            )
            result = EvaluationResult(**workflow_output)
            results.append(result)
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise e

    save_results_to_csv(results)


# --- 実行例 ---
if __name__ == "__main__":
    sample_data = [
        EvaluationData(
            question="質問1",
            image=b"",
            reference_answer="模範回答1",
            model_output="モデル出力1",
        ),
        EvaluationData(
            question="質問2",
            image=b"",
            reference_answer="模範回答2",
            model_output="モデル出力2",
        ),
    ]

    sample_dataset = EvaluationDataset(items=sample_data)

    evaluation_results = evaluate_dataset(sample_dataset)
