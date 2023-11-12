FROM osrf/ros:humble-desktop-jammy

# USE BASH
SHELL ["/bin/bash", "-c"]

# RUN LINE BELOW TO REMOVE debconf ERRORS (MUST RUN BEFORE ANY apt-get CALLS)
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# essential packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install cuda
RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
    && mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.2-535.104.05-1_amd64.deb \
    && cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    cuda \
    && rm -rf /var/lib/apt/lists/*

# install nvidia hpc sdk (contains cuBLAS)
RUN curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    nvhpc-23-7 \
    && rm -rf /var/lib/apt/lists/*

# personal packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    vim \
    gdb \
    gdbserver \
    && rm -rf /var/lib/apt/lists/*

RUN echo "export PATH=$PATH:/usr/local/cuda/bin" >> ~/.bashrc && \
    echo "source /home/dev_ws/install/setup.bash" >> ~/.bashrc
    
WORKDIR /home/dev_ws
