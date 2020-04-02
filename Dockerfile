# --------------------------------------------------------------------------------------
#
# Authors: Brianna Blain-Castelli, Nikkolas Irwin, Adam Cassell, and Andrew Munoz
# Date: 04/01/2020
# Purpose: Build a Big Data application using a Conda environment and Docker.
# Command To Build Image: docker image build -t cars:latest .
#
# --------------------------------------------------------------------------------------

# Pull the base image
FROM continuumio/miniconda3

# Specify the image maintainer
LABEL maintainer="Nikkolas Irwin <nikkolasjirwin@nevada.unr.edu>, \
Brianna Blain-Castelli <bblaincastelli@unr.edu>, \
Andrew Munoz <amunoz24@nevada.unr.edu>, \
Adam Cassell <a.t.cassell@gmail.com>"

# Set the working directory for the rest of the instructions in the file
WORKDIR /app

# Copy 'environment.yml' from the build context
COPY environment.yml .

# Create the conda environment and activate the environment
RUN conda update -y -n base -c defaults conda && \
    conda env create --file environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cars", "-v", "/bin/bash", "-c"]

# Set environmental variables
ENV PATH /opt/conda/envs/cars/bin:$PATH
ENV CONDA_DEFAULT_ENV cars

# Copy the application from the build context
COPY . .

CMD ["python"]