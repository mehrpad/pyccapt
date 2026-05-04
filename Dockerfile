###############################################################################
# PyCCAPT calibration container
#
# Ships the calibration extra plus JupyterLab so users can pull a single image
# and start running the tutorial notebooks without setting up a conda env.
#
# Build:
#     docker build -t pyccapt .
# Run JupyterLab on http://localhost:8888 (no token):
#     docker run --rm -p 8888:8888 -v "$PWD":/work pyccapt
###############################################################################

FROM mambaorg/micromamba:1.5.8

LABEL org.opencontainers.image.source="https://github.com/mmonajem/pyccapt"
LABEL org.opencontainers.image.description="PyCCAPT calibration tools and tutorials"
LABEL org.opencontainers.image.licenses="MIT"

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
WORKDIR /home/$MAMBA_USER/pyccapt

# Install Python + the heavier scientific deps from conda-forge first so the
# pip step that follows only has to compile the small remainder.
RUN micromamba install -y -n base -c conda-forge \
        python=3.11 \
        pip \
        h5py \
        hdf5 \
        pytables \
        numpy \
        pandas \
        scipy \
        matplotlib \
        numba \
        jupyterlab \
        ipywidgets \
        ipympl \
 && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy package source and install the calibration extra.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /home/$MAMBA_USER/pyccapt
RUN pip install --no-cache-dir ".[calibration]"

EXPOSE 8888

# Default command: JupyterLab on all interfaces, no token (suitable for local
# Docker-only use; put a reverse proxy in front for anything internet-facing).
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''", \
     "--notebook-dir=/home/mambauser/pyccapt/pyccapt/calibration/tutorials/jupyter_files"]
