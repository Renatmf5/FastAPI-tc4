#!/bin/bash
DIR="/home/ec2-user/fastapi-app"

echo "ApplicationStart: Iniciando a aplicação" | tee -a /home/ec2-user/fastapi-app/deploy.log

# Adicionar ~/.local/bin ao PATH
export PATH=$PATH:/home/ec2-user/.local/bin
echo "PATH atualizado: $PATH" | tee -a /home/ec2-user/fastapi-app/deploy.log

# Conceder permissões
sudo chmod -R 777 ${DIR}

# Navegar para o diretório da aplicação
cd ${DIR}

# Definir variável de ambiente
export ENV=production

# Construir a imagem Docker
echo "Construindo a imagem Docker" | tee -a ${DIR}/deploy.log
sudo docker build -t fastapi-app . | tee -a ${DIR}/deploy.log

# Parar e remover contêineres antigos
echo "Parando e removendo contêineres antigos" | tee -a ${DIR}/deploy.log
sudo docker stop fastapi-container || true
sudo docker rm fastapi-container || true

# Aplicar setcap ao binário do Python
PYTHON_BIN=$(readlink -f $(which python3))
sudo setcap 'cap_net_bind_service=+ep' $PYTHON_BIN
echo "Permissões setcap aplicadas ao Python" | tee -a /home/ec2-user/fastapi-app/deploy.log
# Iniciar o contêiner Docker
echo "Iniciando o contêiner Docker" | tee -a ${DIR}/deploy.log
sudo docker run -d --name fastapi-container -p 80:80 fastapi-app | tee -a ${DIR}/deploy.log

echo "Aplicação iniciada com sucesso no contêiner Docker" | tee -a ${DIR}/deploy.log


