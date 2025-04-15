FROM python:3.11-bullseye

# タイムゾーン設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 必要なパッケージを確実にインストール
RUN apt-get update && \
    apt-get install -y python3-venv python3-pip vim && \
    apt-get clean
# RUN python3 -m venv --help
# WORKDIR /root/share
# WORKDIR app/
WORKDIR /opt/venv
COPY . .
# COPY requirements.txt .

# 仮想環境の作成とパッケージインストール
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /root/share
CMD [ "bash" ]



