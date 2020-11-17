PEM=$1
PROJECT_DIR=$2
SERVER_DNS=$3
sudo scp -r -i ~/.ssh/$PEM.pem $PROJECT_DIR $SERVER_DNS:~/