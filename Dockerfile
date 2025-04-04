FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# RUN pip install --upgrade pip

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y apt-file
RUN apt-file update

RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y curl
RUN apt-get install -y vim

# python
RUN apt-get install -y python3-pip

# alias
RUN echo alias python="python3" >> /root/.bashrc
RUN echo alias pip="pip3" >> /root/.bashrc

# Install python packages
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install poetry

# Install VIM
RUN apt-get update
RUN apt-get install -y apt-file
RUN apt-file update
RUN apt-get install -y vim

# insatll langchain, and else
RUN pip install langchain==0.1.14
RUN pip install faiss-gpu tiktoken sentence_transformers
RUN pip install wandb
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
RUN pip install flash_attn
RUN pip install openai==0.28.0
RUN pip install pandas==2.2.2
RUN pip install -U langchain-community
RUN pip install openpyxl==3.1.2
RUN pip install python-docx==1.1.2
RUN pip install pypdf==5.1.0
RUN pip install pdfplumber==0.11.5
RUN pip install python-pptx==1.0.2
RUN pip install langchain-openai==0.3.7
RUN pip install dotenv==0.9.9
RUN pip install matplotlib==3.10.1
RUN pip install azure-ai-formrecognizer==3.3.3
RUN pip install langgraph==0.3.22

WORKDIR /root/share

EXPOSE 8080/TCP

