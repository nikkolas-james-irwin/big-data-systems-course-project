# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Containerized Amazon Recommender System (CARS) Project:
#
#   Authors: Brianna Blain-Castelli, Nikkolas Irwin, Adam Cassell, and Andrew Munoz
#   Date: 04/01/2020
#   Purpose: Build a Big Data application using a Conda environment and Docker.
#   Course: CS 636 Big Data Systems
#   Project: CARS is an application that builds a recommender system from datasets provided by
#            UCSD (see citation below). 
#
#   Dataset URL: https://nijianmo.github.io/amazon/index.html
#
#   ***IMPORTANT*** You must download the dataset files for a particular category to your local machine yourself due
#                   to their size. As long as your dataset files are in the same directory as the Dockerfile, then
#                   they will be added to the volume and usable by the container as expected.
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   Citation: Justifying recommendations using distantly-labeled reviews and fined-grained aspects
#             Jianmo Ni, Jiacheng Li, Julian McAuley
#             Empirical Methods in Natural Language Processing (EMNLP), 2019
#             PDF: http://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
#   Example Usage: The commands listed below provide examples with an image called cars with the optional tag latest.
#                  The commands listed below also use an example container called cars_ctnr
#
#       1. Command to Build Image: 
#
#           docker image build -t cars:latest .
#
#       2. Command to Run Container and Build Volume:
#
#           docker container run -d -p 8888:8888 -it --name cars_container --mount source=cars_local_volume,target=/home/jovyan/work cars
#
#       3. Command to View Currently Running Jupyter Notebook Server (Outside of Container):
#
#           docker container exec -it cars_container jupyter notebook list
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Base Image Attribution (jupyter/minimal-notebook):
#
#   Copyright (c) Jupyter Development Team.
#   Distributed under the terms of the Modified BSD License.
#   OS/ARCH: linux/amd64
#   GitHub: https://github.com/jupyter/docker-stacks/tree/master/minimal-notebook
#   DockerHub: https://hub.docker.com/layers/jupyter/minimal-notebook/latest/images/sha256-0dc8e7bd46d7dbf27c255178ef2d92bb8ca888d32776204d19b5c23f741c1414?context=explore
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Pull the base image (jupyter/minimal-notebook) from DockerHub
ARG BASE_IMAGE=jupyter/minimal-notebook:dc9744740e12@sha256:0dc8e7bd46d7dbf27c255178ef2d92bb8ca888d32776204d19b5c23f741c1414
ARG ROOT_IMAGE=${BASE_IMAGE}
FROM ${BASE_IMAGE}

# Specify the CARS image maintainers
LABEL maintainer="Nikkolas Irwin <nikkolasjirwin@nevada.unr.edu>, \
Brianna Blain-Castelli <bblaincastelli@unr.edu>, \
Andrew Munoz <amunoz24@nevada.unr.edu>, \
Adam Cassell <a.t.cassell@gmail.com>"

# Set the user to root during dependency management and installations
USER root

# Set the default Conda environment to the base environment
ENV CONDA_DEFAULT_ENV base

# Update apt and install, locate and Open JDK 8
RUN apt-get -y update && \
    apt-get install -y locate && \
    apt-get install --no-install-recommends -y openjdk-8-jre-headless ca-certificates-java && \
    rm -rf /var/lib/apt/lists/*

# Using pinned spec conda==4.8.2, install explicit dependencies with Conda and then clean tarballs
RUN conda install --yes --name base \
    matplotlib \
    plotly \
    pyspark && \
    conda clean --all --force-pkgs-dirs --yes

# Set the environmental variables for Spark/PySpark
ENV APACHE_SPARK_VERSION=2.4.5
ENV PYSPARK_SUBMIT_ARGS="--master local[*] pyspark-shell"
ENV PYSPARK_PYTHON=python3

# Set the environmental variables for Jupyter Notebook
ENV PYSPARK_DRIVER_PYTHON="jupyter"
ENV PYSPARK_DRIVER_PYTHON_OPTS="notebook"

# Change the user back to the Jupyter Notebook user provided by the jupyter/minimal-notebook
USER ${NB_UID}

# Create the volume mount point to the Jupyter Notebook user's directory
VOLUME /home/jovyan/work

# Copy the dataset files from the current directory into the volume
COPY . /home/jovyan/work
