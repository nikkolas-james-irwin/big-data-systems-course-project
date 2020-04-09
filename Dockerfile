# --------------------------------------------------------------------------------------
#
# Authors: Brianna Blain-Castelli, Nikkolas Irwin, Adam Cassell, and Andrew Munoz
# Date: 04/01/2020
# Purpose: Build a Big Data application using a Conda environment and Docker.
# Command To Build Image: docker image build -t cars:latest .
#
# --------------------------------------------------------------------------------------

# Pull the base image
ARG BASE_IMAGE=jupyter/minimal-notebook:dc9744740e12@sha256:0dc8e7bd46d7dbf27c255178ef2d92bb8ca888d32776204d19b5c23f741c1414
ARG ROOT_IMAGE=${BASE_IMAGE}
FROM ${BASE_IMAGE}

# Specify the image maintainer
LABEL maintainer="Nikkolas Irwin <nikkolasjirwin@nevada.unr.edu>, \
Brianna Blain-Castelli <bblaincastelli@unr.edu>, \
Andrew Munoz <amunoz24@nevada.unr.edu>, \
Adam Cassell <a.t.cassell@gmail.com>"

ENV CONDA_DEFAULT_ENV base

RUN conda install --yes --name base \
    matplotlib \
    plotly \
    pyspark && \
    conda clean --all --force-pkgs-dirs --yes

# Set-up font cache for matplotlib by importing for the first time
# XDG_CACHE_HOME defines the base directory relative to which user specific non-essential data files should be stored.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
# RUN MPLBACKEND=Agg python -c "import matplotlib.py" && \
#     fix-permissions /home/$NB_USER

USER ${NB_UID}
