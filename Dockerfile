FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

# Avoid interactive prompts from apt-get
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3-pip python3.8-dev python3-opencv build-essential && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    pip install --upgrade pip

RUN pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

COPY . /simplify-x
WORKDIR /simplify-x

RUN pip install -r requirements.txt && \
    pip install dependencies/mesh_intersection-0.1.0-cp38-cp38-linux_x86_64.whl && \
    mv dependencies/conversions.py /usr/local/lib/python3.8/dist-packages/torchgeometry/core/conversions.py && \
    pip cache purge && \
    apt-get remove --purge -y python3-pip python3.8-dev build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 3000

CMD ["python", "app.py"]
