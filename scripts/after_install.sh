#!/bin/bash
DIR="/home/ec2-user/fastapi-app"
echo "AfterInstall: Instalando dependências"
sudo chown -R ec2-user:ec2-user ${DIR}
cd ${DIR}

echo "AfterInstall: Preparando o ambiente Docker" | tee -a ${DIR}/deploy.log

# Atualizar o sistema e instalar o Docker, se necessário
echo "Verificando se o Docker está instalado" | tee -a ${DIR}/deploy.log
if ! command -v docker &> /dev/null; then
    echo "Docker não está instalado. Instalando Docker..." | tee -a ${DIR}/deploy.log
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user
else
    echo "Docker já está instalado" | tee -a ${DIR}/deploy.log
    sudo service docker start
fi

# Garantir permissões para o diretório do projeto
echo "Garantindo permissões para o diretório ${DIR}" | tee -a ${DIR}/deploy.log
sudo chmod -R 777 ${DIR}

# Limpar imagens e contêineres antigos (opcional)
echo "Limpando imagens e contêineres antigos" | tee -a ${DIR}/deploy.log
sudo docker system prune -af || true

echo "AfterInstall: Ambiente Docker preparado com sucesso" | tee -a ${DIR}/deploy.log