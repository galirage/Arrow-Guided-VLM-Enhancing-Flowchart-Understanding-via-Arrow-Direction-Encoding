import argparse
import json
import os


def parser() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="lllm_args")
    parser.add_argument(
        "--process_name", "-pn", type=str, default="load_pdf", help="process name"
    )
    parser.add_argument(
        "--dataset_dir",
        "-dsd",
        type=str,
        default="../dataset/DocVQA/",
        help="pdf file to load",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        default="output/",
        help="path to output directory",
    )

    return parser.parse_args()


def check_DocVQA_dataset(args: argparse.Namespace) -> None:
    """
    Checks the DocVQA dataset annotations, specifically looking for 'flow' in questions or answers.

    Args:
        args: Command line arguments, expected to have a 'dataset_dir' attribute.
    """
    # TODO: Verify the dataset path or make it more robust (e.g., relative to project root or configurable).
    # TODO: Execution test failed with FileNotFoundError. Resolve the dataset path issue noted above.
    # check annotations
    annotation_path = os.path.join(args.dataset_dir, "qas/train.json")
    print(f"Checking annotations in: {annotation_path}")
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {annotation_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {annotation_path}")
        return

    if "data" not in data:
        print("Error: 'data' key not found in the JSON structure.")
        return

    print("Scanning for 'flow' in questions and answers...")
    found_count = 0
    for i, data1 in enumerate(data["data"]):
        # Ensure keys exist before accessing them
        question = data1.get("question", "")
        answers = data1.get("answers", [])

        if "flow" in question:
            print(f"Found 'flow' in question (item {i}): {question}")
            found_count += 1
        for answer1 in answers:
            if isinstance(answer1, str) and "flow" in answer1:
                print(f"Found 'flow' in answer (item {i}): {answer1}")
                found_count += 1

    print(f"Finished scanning. Found {found_count} occurrences of 'flow'.")


if __name__ == "__main__":
    """
    usage)
    python check_dataset.py --process_name check_docvqa --dataset_dir <実際のデータセットディレクトリへのパス>
    """
    args = parser()
    if args.process_name == "check_docvqa":
        """
        check DocVQA Dataset
        https://www.docvqa.org/datasets
        """
        check_DocVQA_dataset(args)
