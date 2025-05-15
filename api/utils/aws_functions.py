import io
import joblib
import tempfile
import boto3
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

def buscar_modelo_no_s3(symbol, model_path, modelo = 'pkl'):

    s3 = boto3.client('s3')
    try:
        # Construir a Key do modelo
        key = f'models/{symbol}/{model_path}/{model_path}_model.{modelo}'
        response = s3.get_object(Bucket='models-bucket-tc4', Key=key)
        content = response['Body'].read()
        
        # Verificar o tipo de modelo e carregar adequadamente
        if modelo == 'keras':
            # Criar um arquivo temporário para salvar o modelo
            with tempfile.NamedTemporaryFile(suffix='.keras') as temp_file:
                temp_file.write(content)  # Escrever o conteúdo no arquivo temporário
                temp_file.flush()  # Garantir que os dados sejam gravados
                model = tf.keras.models.load_model(temp_file.name)  # Carregar o modelo Keras
        else:
            model = joblib.load(io.BytesIO(content))  # Carregar modelo joblib
        
        print(f"Modelo carregado com sucesso: {key}")
        return model
    except Exception as e:
        print(f"Erro ao buscar o modelo no S3: {e}")
        return None

def buscar_indicador(indicador):

    s3 = boto3.client('s3')
    try:
        # Construir a Key do modelo
        key = f'indicadores/{indicador}.parquet'
        response = s3.get_object(Bucket='datalake-tc4', Key=key)
        indicador_df = pd.read_parquet(io.BytesIO(response['Body'].read()))

                
        print(f"indicador carregado carregado com sucesso: {key}")
        return indicador_df
    except Exception as e:
        print(f"Erro ao buscar o modelo no S3: {e}")
        return None

def ler_parametros_scaler_do_s3(symbol, scaler_path):
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket='models-bucket-tc4', Key=f'models/{symbol}/scaler/{symbol}_{scaler_path}')
        content = response['Body'].read()
        scaler = joblib.load(io.BytesIO(content))
        
        # Verificar o tipo do scaler
        if isinstance(scaler, StandardScaler):
            print(f"Scaler carregado: StandardScaler para {symbol}")
            return scaler
        elif isinstance(scaler, MinMaxScaler):
            print(f"Scaler carregado: MinMaxScaler para {symbol}")
            return scaler
        else:
            print("O objeto carregado não é um scaler válido.")
            return None
    except Exception as e:
        print(f"Erro ao obter parâmetros do scaler do S3: {e}")
        return None