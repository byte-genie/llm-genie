FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*
## install curl
RUN apt-get update && apt -y install curl
WORKDIR /code
RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir --upgrade awslambdaric
## install gcloud
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get install -y google-cloud-sdk
## install img libraries
RUN apt-get install ffmpeg libsm6 libxext6  -y
## install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
## create conda env
conda create -n -y llm-genie python=3.10
conda activate llm-genie
## install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
## create dir for storing trained models
mkdir -p ~/data
## copy code
COPY ./ /code