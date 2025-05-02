FROM ubuntu:22.04

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.10 && \
    apt-get install -y python3-pip

RUN pip install "torchdata>=0.8.0" \
    "datasets>=2.21.0" \
    "tomli>=1.1.0" \
    "tensorboard" \
    "tiktoken" \
    "blobfile" \
    "tabulate" \
    "wandb" \
    "fsspec" \
    "tyro" \
    "transformers>=4.51.1" \
    "einops" \
    "sentencepiece" \
    "Pillow"


# symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python


