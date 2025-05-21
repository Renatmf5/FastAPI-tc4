from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import boto3
import json
import os
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

# Inicializa o cliente S3
s3_client = boto3.client("s3")

@router.get("/getMetrics")
async def get_open_orders():
    try:
        # Baixa o arquivo JSON do S3
        response = s3_client.get_object(Bucket=os.getenv("S3_BUCKET_NAME"), Key='backtests/backtests_results.json')
        file_content = response["Body"].read().decode("utf-8")
        
        # Carrega o conteúdo do JSON
        data = json.loads(file_content)
        
        #print(f"Posições retornadas pela API: {positions}")  # Log de depuração
        if not isinstance(data, list):
            data = [data]  # Garante que sempre retorna uma lista
        return JSONResponse(content=data, media_type="application/json")
    except Exception as e:
        print(f"Erro ao obter metricas: {e}")
        return None


@router.get("/getCarteiras")
async def get_open_orders():
    try:
        # Baixa o arquivo JSON do S3
        response = s3_client.get_object(Bucket=os.getenv("S3_BUCKET_NAME"), Key='backtests/historico_carteiras.json')
        file_content = response["Body"].read().decode("utf-8")
        
        # Carrega o conteúdo do JSON
        data = json.loads(file_content)
        
        #print(f"Posições retornadas pela API: {positions}")  # Log de depuração
        if not isinstance(data, list):
            data = [data]  # Garante que sempre retorna uma lista
        return JSONResponse(content=data, media_type="application/json")
    except Exception as e:
        print(f"Erro ao obter metricas: {e}")
        return None