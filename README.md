# Flow-chart detection project
## Trees
- notebook: jupyter notebooks to detect flow-chart diagrams
- src: several programs
      - check_dataset.py: python programs to check datasets

## 環境構築

このプロジェクトは[Rye](https://rye-up.com/)を使用して環境を管理しています。

### 前提条件
- Python 3.11.9
- Rye (インストール方法: https://rye-up.com/guide/installation/)

### セットアップ手順

1. リポジトリをクローン
```bash
git clone [repository-url]
cd gg-rq-rag-flowchat-detection
```

2. 依存関係のインストール
```bash
rye sync
```

3. 仮想環境のアクティベート
```bash
.venv/bin/activate
```

### 開発用ツール
このプロジェクトでは、コードの品質管理とフォーマットに`ruff`を使用しています。
`rye sync`を実行すると、開発用の依存関係として`ruff`がインストールされます。

VSCodeを使用している場合は、`charliermarsh.ruff`拡張機能をインストールすると、保存時のフォーマットやリアルタイムでのリントチェックが有効になります（`.vscode/settings.json`で設定済み）。

以下のツールも利用可能です：
- `pytest`: テスト実行

### 依存関係の追加

新しい依存関係を追加する場合は、以下のいずれかの方法で行います。

1.  **`pyproject.toml`を編集**: `[tool.rye.dependencies]` (本番用) または `[tool.rye.dev-dependencies]` (開発用) にパッケージ名とバージョン指定を追加し、以下のコマンドを実行します。
    ```bash
    rye sync
    ```

2.  **`rye add`コマンドを使用**:
    - 本番用依存関係を追加する場合:
      ```bash
      rye add <package_name>
      # 例: rye add requests
      ```
    - 開発用依存関係を追加する場合:
      ```bash
      rye add --dev <package_name>
      # 例: rye add --dev pytest-mock
      ```
    `rye add`コマンドを実行すると、`pyproject.toml`が自動的に更新され、`rye sync`相当の処理も行われます。
