PEM=$1
SERVER_DNS=$2
CHECKPOINT_NAME=$3
sudo scp -r -i ~/.ssh/$PEM.pem $SERVER_DNS:~/screenwriter/screenwriter/checkpoints/$CHECKPOINT_NAME ./screenwriter/checkpoints
sudo chmod 777 ./screenwriter/checkpoints