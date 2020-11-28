PEM=$1
SERVER_DNS=$2
sudo scp -r -i ~/.ssh/$PEM.pem $SERVER_DNS:~/screenwriter/screenwriter/tensorboard-logs ./screenwriter/tensorboard-logs
sudo chmod 777 ./screenwriter/tensorboard-logs