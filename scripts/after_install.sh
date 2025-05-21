#!/bin/bash
DIR="/home/ec2-user/fastapi-app"
echo "AfterInstall: Instalando dependências"
sudo chown -R ec2-user:ec2-user ${DIR}
cd ${DIR}

# Instalar dependências Python usando pip
pip install -r requirements.txt