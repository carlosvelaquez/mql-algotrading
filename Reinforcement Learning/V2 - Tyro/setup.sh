ssh -i "C:/Users/imado/.ssh/carlosv.pem" ubuntu@18.191.242.255
scp -i "C:/Users/imado/.ssh/carlosv.pem" tyro.py ubuntu@18.191.242.255:tyro.py
scp -i "C:/Users/imado/.ssh/carlosv.pem" main.py ubuntu@18.191.242.255:main.py

coach -p ./main.py -n 4 -s 18000

#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>/home/ubuntu/log.out 2>&1
# Everything below will go to the file 'log.out':

apt-get update
apt-get install python3-pip -y

wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sh cuda_10.2.89_440.33.01_linux.run --silent --driver --toolkit

wget https://imadomldatasets.s3.amazonaws.com/lib.deb
dpkg -i lib.deb

su - ubuntu -c "cd && pip3 install pip==19.0 && pip install --user rl_coach && wget https://imadomldatasets.s3.amazonaws.com/history.csv"
reboot