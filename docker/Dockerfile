FROM tensorflow/tensorflow:1.15.4-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends git python3-pip python3-tk \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN git clone https://github.com/PRBonn/OverlapNet.git /root/OverlapNet
RUN pip3 install -r /root/OverlapNet/requirements.txt

WORKDIR /root/OverlapNet
