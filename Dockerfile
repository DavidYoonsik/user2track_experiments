ARG BASEIMG

FROM nvidia/cuda:10.0-base-centos7 as base
FROM $BASEIMG as devel

# Install CUDA & cuDNN
ENV CUDNN_VERSION 7.6.5.32
ENV CUDA_VERSION 10.0.130
ENV CUDA_PKG_VERSION 10-0-${CUDA_VERSION}-1

ENV WORKDIR=/opt/app
WORKDIR $WORKDIR

COPY --from=base /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA
COPY --from=base /etc/yum.repos.d/cuda.repo /etc/yum.repos.d/cuda.repo

RUN yum install -y \
    cuda-cudart-$CUDA_PKG_VERSION \
    cuda-compat-10-0 \
    && \
    ln -s cuda-10.0 /usr/local/cuda \
    && \
    rm -rf /var/cache/yum/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# MCP GPU - Tesla V100 8GB * 2 Devices
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"

RUN yum install -y \
    cuda-libraries-$CUDA_PKG_VERSION \
    cuda-nvtx-$CUDA_PKG_VERSION \
    && \
    rm -rf /var/cache/yum/*

RUN yum install -y \
    cuda-nvml-dev-$CUDA_PKG_VERSION \
    cuda-command-line-tools-$CUDA_PKG_VERSION \
    cuda-libraries-dev-$CUDA_PKG_VERSION \
    cuda-minimal-build-$CUDA_PKG_VERSION \
    && \
    rm -rf /var/cache/yum/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

RUN CUDNN_DOWNLOAD_SUM=28355e395f0b2b93ac2c83b61360b35ba6cd0377e44e78be197b6b61b4b492ba && \
    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v7.6.5/cudnn-10.0-linux-x64-v7.6.5.32.tgz -O && \
    echo "$CUDNN_DOWNLOAD_SUM  cudnn-10.0-linux-x64-v7.6.5.32.tgz" | sha256sum -c - && \
    tar --no-same-owner -xzf cudnn-10.0-linux-x64-v7.6.5.32.tgz -C /usr/local && \
    rm -f cudnn-10.0-linux-x64-v7.6.5.32.tgz && \
    ldconfig

FROM devel

# Install YUM Package
RUN bash -lc "yum install -y bzip2-devel xz-devel sqlite sqlite-devel jq sshpass"

# AWS
ENV AWS_DEFAULT_REGION ap-northeast-2
ENV AWS_DEFAULT_OUTPUT json

# Copy sources
COPY src $WORKDIR/src

# Copy resources
COPY res $WORKDIR/res

# Copy SQL
COPY sql $WORKDIR/sql

# Make directory
RUN mkdir -p $WORKDIR/checkpoint && \
    mkdir -p $WORKDIR/data && \
    mkdir -p $WORKDIR/data_meta && \
    mkdir -p $WORKDIR/data_w2v && \
    mkdir -p $WORKDIR/data_demo

# Install packages
COPY requirements.txt $WORKDIR/
RUN bash -lc "res/install-python.sh && \
              rm -f $WORKDIR/res/install-python.sh && \
              echo export PYTHON_ENV_PATH='$(dirname $(dirname $(pyenv which python)))' >> ~/.bashrc && \
              python -m pip install --upgrade pip && \
              python -m pip install -r ${WORKDIR}/requirements.txt"

# Copy .sh
COPY runner.sh _spark-sql.sh remote-runner.sh $WORKDIR/
RUN chmod 755 ${WORKDIR}/runner.sh && \
    chmod 755 ${WORKDIR}/_spark-sql.sh && \
    chmod 755 ${WORKDIR}/remote-runner.sh

# Entry
ENTRYPOINT [ "" ]

# CMD
CMD [ "bash", "-l" ]