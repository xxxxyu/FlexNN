# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Avoid interactive prompts in docker build
ARG DEBIAN_FRONTEND=noninteractive

# Change package sources to Tsinghua mirror (optional for Chinese users)
# ADD sources.list /etc/apt/

# Install dependencies
# For setting up SDKs: wget ca-certificates unzip
# For building FlexNN: git cmake libprotobuf-dev protobuf-compiler libomp-dev
RUN apt-get update && \
    apt-get install -y build-essential git cmake libprotobuf-dev protobuf-compiler libomp-dev wget ca-certificates unzip && \
    rm -rf /var/lib/apt/lists/*

# Install the following dependencies:
# matplotlib
# numpy
# pandas
# scipy
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install matplotlib numpy pandas scipy && \
    rm -rf /var/lib/apt/lists/*