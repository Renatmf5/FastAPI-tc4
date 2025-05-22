#!/bin/bash
echo "ApplicationStop: Parando o contêiner Docker"

# Parar o contêiner
sudo docker stop fastapi-container || true