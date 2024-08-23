FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
ENV http_proxy="xx"

MAINTAINER Zeynep Pehlivan
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*
# Set working directory

# Set working directory
WORKDIR /usr/src/app

RUN pip install h5py-cache

# Install TorchNet, a high-level framework for PyTorch.
RUN pip install torchnet

# Install Requests, a Python library for making HTTP requests.
RUN pip install requests


# Install Graphviz.
RUN pip install graphviz

RUN pip install scipy

RUN pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.8.0+cu102.html \
 && pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.8.0+cu102.html \
 && pip install --no-index torch-cluster -f https://data.pyg.org/whl/torch-1.8.0+cu102.html \
 && pip install --no-index torch-spline-conv -f https://data.pyg.org/whl/torch-1.8.0+cu102.html \
 && pip install torch-geometric

RUN pip install tensorboard
RUN pip install torchtext
RUN pip install graph4nlp-cu102
RUN pip install transformers
# Set the default command to python3.
COPY ./dockertemp.py /usr/src/app
ENTRYPOINT ["python3"]
CMD [ "dockertemp.py", "/usr/src/temp/config.json" ]
