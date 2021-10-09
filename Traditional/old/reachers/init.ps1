#Invoke-Expression "scp -i ~/.ssh/carlosv_nvirginia.pem train.py ubuntu@${env:SERVER_ADDRESS}:~"
#Invoke-Expression "scp -i ~/.ssh/carlosv_nvirginia.pem dataset.csv ubuntu@${env:SERVER_ADDRESS}:~"
ssh -i ~/.ssh/carlosv_nvirginia.pem ubuntu@$env:SERVER_ADDRESS "sudo apt-get update -y && sudo apt-get upgrade -y && sudo apt-get install build-essential -y && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/ubuntu/miniconda && PATH=`$PATH:~/home/ubuntu/miniconda/bin && /home/ubuntu/miniconda/bin/conda init bash && echo '/home/ubuntu/miniconda/bin/conda install tensorflow-gpu scikit-learn pandas -y && wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run && sudo sh cuda_10.2.89_440.33.01_linux.run --silent --driver --toolkit && sudo reboot\' >> olo.sh && exec bash olo.sh"
 


