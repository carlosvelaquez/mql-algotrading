#!/bin/bash

sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev -y
pip3 install pip==19.0
pip install --user stable-baselines[mpi] tensorflow-gpu==1.14 pandas numpy pyzmq gym

wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run --silent --toolkit --driver

scp -i "C:/Users/imado/.ssh/carlosv_nvirginia.pem" train.py ubuntu@ec2-184-72-163-219.compute-1.amazonaws.com:train.py
scp -i "C:/Users/imado/.ssh/carlosv_nvirginia.pem" prodigy.py ubuntu@ec2-184-72-163-219.compute-1.amazonaws.com:prodigy.py