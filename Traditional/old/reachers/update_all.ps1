Invoke-Expression "scp -i ~/.ssh/carlosv_nvirginia.pem train.py ubuntu@${env:SERVER_ADDRESS}:~"
Invoke-Expression "scp -i ~/.ssh/carlosv_nvirginia.pem dataset.csv ubuntu@${env:SERVER_ADDRESS}:~"