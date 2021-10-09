#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>/home/ubuntu/log.out 2>&1
# Everything below will go to the file 'log.out':

apt-get update
apt-get install python3-pip cmake libopenmpi-dev python3-dev zlib1g-dev libsm6 libxext6 libfontconfig1 libxrender1 -y

wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sh cuda_10.2.89_440.33.01_linux.run --silent --driver --toolkit

su - ubuntu -c "pip3 install pip==19.0 && pip install --user gast==0.2.2 tensorflow-gpu==1.14 pandas numpy gym zmq stable-baselines[mpi] && cd ~ && wget https://imadomldatasets.s3.amazonaws.com/history.zip"

ssh -i "C:/Users/imado/.ssh/carlosv_nvirginia.pem" ubuntu@ec2-54-234-173-237.compute-1.amazonaws.com
scp -i "C:/Users/imado/.ssh/carlosv_nvirginia.pem" hellbent.py ubuntu@ec2-54-234-173-237.compute-1.amazonaws.com:hellbent.py
scp -i "C:/Users/imado/.ssh/carlosv_nvirginia.pem" train.py ubuntu@ec2-54-234-173-237.compute-1.amazonaws.com:train.py