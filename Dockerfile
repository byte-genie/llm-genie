FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*
## install curl
RUN apt-get update && apt -y install curl
WORKDIR /code
RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir --upgrade awslambdaric
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update
RUN apt-get install -y google-cloud-sdk
## install img libraries
RUN apt-get install ffmpeg libsm6 libxext6  -y
## install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
## install apt-get libs
RUN apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates
## install conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    rm -rf /var/lib/apt/lists/*
## create conda env
RUN conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n llm-genie && \
    conda install python=3.10 pip && \
    conda activate llm-genie
## install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install einops transformers_stream_generator
## install deepspeed
RUN pip install deepspeed
# ENV STAGE_DIR=/tmp
# RUN mkdir -p ${STAGE_DIR}
# RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
# RUN pip install triton
# RUN cd ${STAGE_DIR}/DeepSpeed && \
#     git checkout . && \
#     git checkout master && \
#     DS_BUILD_OPS=1 pip install .
# RUN rm -rf ${STAGE_DIR}/DeepSpeed
# RUN python -c "import deepspeed; print(deepspeed.__version__)" && ds_report
## create dir for storing trained models
RUN mkdir -p /data
## copy code
COPY ./ /code