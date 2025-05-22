# Usar uma imagem base do Python
FROM python:3.10-slim

# Definir o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiar os arquivos do projeto para o contêiner
COPY . /app

# Instalar dependências do sistema (se necessário)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Aplicar permissões para permitir que o Python escute na porta 80
RUN apt-get update && apt-get install -y libcap2-bin && \
    setcap 'cap_net_bind_service=+ep' $(readlink -f $(which python3))

# Expor a porta 80 para o host
EXPOSE 80

ENV S3_BUCKET_NAME=models-bucket-tc4
# Comando para iniciar o servidor FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]