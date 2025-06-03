from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from api.utils.data_handler import DataAnalitcsHandler
import boto3
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Inicializa o cliente S3
s3_client = boto3.client("s3")

# Lock para permitir apenas 1 requisição ativa
lock = asyncio.Lock()

# Executor para rodar funções bloqueantes em thread separada
executor = ThreadPoolExecutor(max_workers=1)  # Apenas 1 thread ativa para proteger contra rate-limit


@router.get("/Predict/{ticker}")
async def get_predict_by_ticker(ticker: str):
    """
    Endpoint para obter previsões com base no ticker fornecido.
    Limita a apenas 1 requisição ativa. Se uma já estiver em execução, retorna erro 429 (Too Many Requests).
    """
    # Tenta adquirir o lock imediatamente (sem esperar)
    locked = lock.locked()

    if locked:
        raise HTTPException(
            status_code=429,
            detail="Outra requisição já está em processamento. Tente novamente em alguns instantes."
        )

    try:
        # Adquire o lock
        async with lock:
            dicionario_indicadores = {
                'indicadores': {
                    'ebit_dl',
                    'ROIC',
                    'EBIT_Ativos',
                    'ROE',
                    'PSR'
                }
            }

            handler = DataAnalitcsHandler(
                ticker=f'{ticker}.SA',
                **dicionario_indicadores
            )

            # Executa função bloqueante no executor
            loop = asyncio.get_event_loop()
            resultado = await loop.run_in_executor(executor, handler.gera_prediction)

            return JSONResponse(content=resultado, media_type="application/json")

    except Exception as e:
        print(f"Erro ao obter previsões para o ticker '{ticker}': {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao obter previsões.")
