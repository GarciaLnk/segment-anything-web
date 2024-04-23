FROM nvidia/cuda:11.8.0-base-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND="noninteractive"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV MINIFORGE_NAME=Miniforge3
ENV MINIFORGE_VERSION=23.3.1-1
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

COPY . /app/
WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    curl=7.81.0-1ubuntu1.15 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSLo /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh" && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    mamba env create -f scripts/environment.yml && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/mamba.sh" >> ~/.bashrc

EXPOSE 3000
EXPOSE 8000

# use a wrapper for quick testing, cba with compose
CMD ["./wrapper.sh"]
