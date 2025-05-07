# フローチャート検出プロジェクト

## 概要

本リポジトリは、[論文タイトル（必要であれば追加）]の研究で使用されたフローチャート図検出のためのプログラムコードを管理するものです。

プロジェクトは主に以下の2つの要素で構成されています。

-   `notebooks/`: フローチャート図検出のための深層学習モデルに関する実験を行ったJupyter Notebookが含まれます。これらのNotebookは、個々に必要な環境構築手順や依存関係リスト（例: ディレクトリ内の`requirements.txt`）を持つ場合があります。
-   `src/`: 大規模言語モデル（LLM）関連の処理（パーサー、評価ツールなど）を行うPythonスクリプトが含まれます。

## 環境構築

### LLM関連スクリプト (`src/`)

`src/`ディレクトリ内のPythonスクリプトは、標準的なPython仮想環境 (`venv`) を使用します。

**前提条件:**
-   Python 3.11 以降

**セットアップ手順:**

1.  プロジェクトのルートディレクトリに移動します。
    ```bash
    cd gg-rq-rag-flowchat-detection
    ```

2.  仮想環境を作成します。
    ```bash
    python -m venv .venv
    ```

3.  仮想環境を有効化します。
    -   macOS/Linuxの場合:
        ```bash
        source .venv/bin/activate
        ```
    -   Windowsの場合:
        ```bash
        .venv\Scripts\activate
        ```

4.  `src`ディレクトリ用の依存関係をインストールします。（`src/requirements.txt`が存在する場合）
    ```bash
    pip install -r src/requirements.txt
    ```
    *(注意: この`requirements.txt`ファイルは必要に応じて作成してください)*

### 深層学習関連Notebook (`notebooks/`)

`notebooks/`ディレクトリ内の各Jupyter Notebookに記載されている指示に従って、特定の実験に必要な環境や依存関係をセットアップしてください。もし`notebooks/requirements.txt`ファイルが提供されている場合は、任意の環境（例: conda, venv）で`pip`を使用して依存関係をインストールできます。

## 開発ツール

`src/`ディレクトリのコードについては、リンター/フォーマッターとして`ruff`、テストツールとして`pytest`の使用を推奨します。これらは作成した`.venv`環境内にインストールしてください。

```bash
pip install ruff pytest
```

VSCodeを使用している場合、`charliermarsh.ruff`拡張機能をインストールすると、コード保存時の自動フォーマットやリアルタイムのlintチェックが有効になります。

## OCR, detection -> LLM の実行手順
```bash
cd src
python graph.py --process_name main --img_path PATH/TO/FLOW-CHART-IMAGE-FILE
# ex)
python graph.py --process_name main --img_path ../images/flowchart-example179.png
```