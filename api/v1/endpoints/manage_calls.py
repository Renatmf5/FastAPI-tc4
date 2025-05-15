from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from api.utils.data_handler import DataAnalitcsHandler
import boto3

from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

# Inicializa o cliente S3
s3_client = boto3.client("s3")

@router.get("/Predict/{ticker}")
async def get_predict_by_ticker(ticker: str):
    """
    Endpoint para obter previsões com base no ticker fornecido.
    """
    try:
        dicionario_indicadores = {
            'indicadores': {
                'ebit_dl',
                'ROIC',
                'EBIT_Ativos',
                'ROE',
                'PSR'
            }
        }
        handler = DataAnalitcsHandler(ticker=f'{ticker}.SA', **dicionario_indicadores)
        resultado = handler.gera_prediction()

        return JSONResponse(content=resultado, media_type="application/json")
    except Exception as e:
        print(f"Erro ao obter previsões para o ticker '{ticker}': {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao obter previsões.")