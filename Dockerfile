# 軽量なPythonベースのイメージを使用（Ubuntu 22.04 ベース）
FROM python:3.10-slim

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pythonのアップグレードとエイリアス設定
RUN pip install --no-cache-dir --upgrade pip \
    && echo "alias python='python3'" >> ~/.bashrc \
    && echo "alias pip='pip3'" >> ~/.bashrc

# 必要なPythonパッケージをインストール（LLM, RAG 向け）
RUN pip install --no-cache-dir langchain==0.2.1
RUN pip install --no-cache-dir langchain-openai==0.3.8
RUN pip install --no-cache-dir faiss-cpu
RUN pip install --no-cache-dir tiktoken
RUN pip install --no-cache-dir sentence_transformers
RUN pip install --no-cache-dir wandb
RUN pip install --no-cache-dir openai==0.28.0
RUN pip install --no-cache-dir pandas==2.2.2
RUN pip install --no-cache-dir langchain-community
RUN pip install --no-cache-dir openpyxl==3.1.2
RUN pip install --no-cache-dir python-docx==1.1.2
RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir python-dotenv==0.9.0   # dotenv ではなく python-dotenv を使う
RUN pip install --no-cache-dir opencv-python==4.11.0.86
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get install -y libgl1-mesa-glx
RUN pip install twine==6.1.0
RUN pip install build==1.2.2.post1
RUN pip install openai==1.70.0
RUN pip install azure-storage-blob
RUN pip install azure-identity
RUN pip install langgraph==0.3.26
RUN pip install typing_extensions==4.13.1
RUN pip install azure-ai-formrecognizer==3.3.3
RUN pip install matplotlib==3.10.1

# Azure CLI 用の環境変数の設定（非対話モードでの Azure CLI の動作を有効化）
ENV DEBIAN_FRONTEND=noninteractive

# Azure CLI に必要なパッケージのインストール
# RUN apt-get update && apt-get install -y \
#     curl \
#     gnupg \
#     && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# RUN pip install openai==1.70.0
# RUN pip install numpy==2.2.4

# Azure CLI の動作確認
# RUN az --version

# 作業ディレクトリの設定
WORKDIR /root/share

# ポートの公開（FastAPI などを使用する場合）
# EXPOSE 8000

RUN ln -s /root/share /external_code

# コンテナ起動時のデフォルトコマンド
CMD ["bash"]
