# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.8    (apt)
# pytorch       1.4.0  (pip)
# ==================================================================
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update
# ==================================================================
# tools
# ------------------------------------------------------------------
RUN apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
	    nano \
        libx11-dev \
        fish \
        libsparsehash-dev \
        software-properties-common \
        libsm6 \
        libxext6 \
        libxrender-dev

# ==================================================================
# python
# ------------------------------------------------------------------
WORKDIR /CaDDN
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-pip \
        python3-setuptools
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install --upgrade -r requirements.txt

# ==================================================================
# Environment
# ------------------------------------------------------------------
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# nvidia runtime
COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04 \
 /usr/lib/x86_64-linux-gnu \
 /usr/lib/x86_64-linux-gnu

COPY --from=nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04 \
 /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
 /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
 ldconfig && \
 echo '/usr/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
 echo '/usr/$LIB/libEGL.so.1' >> /etc/ld.so.preload

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# ==================================================================
# File Structure
# ------------------------------------------------------------------
WORKDIR /CaDDN
COPY pcdet pcdet
COPY setup.py setup.py
RUN python setup.py develop
RUN mkdir checkpoints && mkdir data && mkdir output && && mkdir tools