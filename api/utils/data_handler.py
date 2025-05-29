import os
import boto3
from curl_cffi import requests as curl_cffi_request
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from .aws_functions import buscar_modelo_no_s3, ler_parametros_scaler_do_s3, buscar_indicador
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import socket
pd.set_option('future.no_silent_downcasting', True)


class DataAnalitcsHandler:
    
    def __init__(self, ticker, **kargs):
        self.ticker = ticker
        self.indicadores = kargs
        self.bucket_name = 'models-bucket-tc4'
        self.df_dados = pd.DataFrame() 

    def get_cdi_history(self, start_date, end_date):
        """
        Consulta o histórico do CDI entre as datas especificadas.
        """
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
            df["valor"] = pd.to_numeric(df["valor"]) / 100
            df = df.rename(columns={"valor": "retorno_cdi"})
            return df
        else:
            print(f"Erro ao acessar a API do Banco Central: {response.status_code}")
            return None

    def get_yahoo_finance_data(self, ticker, start_date, end_date):
        """
        Consulta os dados de cotação de um ticker no Yahoo Finance.
        """
        try:
            print(f"Analisando a variável 'ticker': valor={repr(ticker)}, tipo={type(ticker)}")
            session = curl_cffi_request.Session(impersonate="chrome110")
            """
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            session.headers.update(headers)
            """

            ticker_obj = yf.Ticker(ticker, session=session)
            
            # Buscar o histórico de preços
            df_ticker = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
            # Configurar o agente de usuário globalmente no yfinance
            #session = requests.Session(impersonate=False)
            #print(f"Buscando dados para o ticker: {ticker}")
            df_ticker = yf.download(tickers=ticker,start=start_date,end=end_date,auto_adjust=False,progress=False,threads=True)
            # Verificar se o DataFrame está vazio
            if df_ticker.empty:
                print(f"Sem dados para {ticker}")
                if ticker != "^BVSP":
                    print("Carregando dados do S3...")
                    # Inicializar o cliente S3
                    s3_client = boto3.client('s3')
                    bucket_name = "datalake-tc4"
                    file_key = "acoes_cotacoes.parquet"

                    # Baixar o arquivo do S3
                    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                    df_s3 = pd.read_parquet(response['Body'])

                    # Filtrar os dados pelo ticker
                    df_ticker = df_s3[df_s3['ticker'] == ticker]

                    # Ordenar pela data mais recente e pegar os últimos 300 registros
                    df_ticker = df_ticker.sort_values(by='data', ascending=False).head(300)

                    # Verificar se ainda está vazio após o filtro
                    if df_ticker.empty:
                        print(f"Sem dados disponíveis para o ticker {ticker} no arquivo S3.")
                        return pd.DataFrame()  # Retorna um DataFrame vazio
                    
                elif ticker == "^BVSP":
                    print("Carregando dados do S3...")
                    # Inicializar o cliente S3
                    s3_client = boto3.client('s3')
                    bucket_name = "datalake-tc4"
                    file_key = "IBOV.parquet"

                    # Baixar o arquivo do S3
                    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                    df_s3 = pd.read_parquet(response['Body'])

                    # Ordenar pela data mais recente e pegar os últimos 300 registros
                    df_ticker = df_ticker.sort_values(by='data', ascending=False).head(300)

                    # Verificar se ainda está vazio após o filtro
                    if df_ticker.empty:
                        print(f"Sem dados disponíveis para o ticker {ticker} no arquivo S3.")
                        return pd.DataFrame()  # Retorna um DataFrame vazio


            # Resetar o índice
            df_ticker = df_ticker.reset_index()

            # Tratar MultiIndex, se presente
            if isinstance(df_ticker.columns, pd.MultiIndex):
                df_ticker.columns = df_ticker.columns.get_level_values(0)

            # Renomear colunas
            df_ticker = df_ticker.rename(columns={
                'Date': 'data',
                'Adj Close': 'preco_fechamento_ajustado',
                'Close': 'close',
                'High': 'high',
                'Low': 'low',
                'Open': 'open',
                'Volume': 'volume'
            })

            # Adicionar o ticker como coluna
            df_ticker['ticker'] = ticker

            # Selecionar as colunas finais
            df_ticker = df_ticker[['data', 'ticker', 'preco_fechamento_ajustado',
                                'close', 'high', 'low', 'open','volume']]

            return df_ticker

        except Exception as e:
            print(f"Erro ao buscar {ticker}: {e}")
            return pd.DataFrame() 

    
    
    def cria_data_frame(self):
        def otimizar_tipos(df):
            """
            Otimiza os tipos de dados de um DataFrame para reduzir o consumo de memória.
            """
            for col in df.select_dtypes(include=['float64']).columns:
                df.loc[:, col] = df[col].astype('float32')
            for col in df.select_dtypes(include=['int64']).columns:
                df.loc[:, col] = df[col].astype('int32')
            return df
        """
        Cria um DataFrame com os dados de cotação da ação especificada, IBOV e CDI.
        """
        # Definir período de 60 dias
        end_date = datetime.today()
        start_date = end_date - timedelta(days=500)
        start_date_str = start_date.strftime("%d/%m/%Y")
        end_date_str = end_date.strftime("%d/%m/%Y")
        #yf.utils.set_user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        # Buscar dados da ação
        cotacoes_acao = self.get_yahoo_finance_data(self.ticker, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))

        # Buscar dados do IBOV
        cotacoes_ibov = self.get_yahoo_finance_data("^BVSP", start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
        cotacoes_ibov = cotacoes_ibov.rename(columns={'preco_fechamento_ajustado': 'fechamento_ibov'})
        cotacoes_ibov['pct_ibov'] = cotacoes_ibov['fechamento_ibov'].pct_change()
        cotacoes_ibov = cotacoes_ibov[['data', 'fechamento_ibov', 'pct_ibov']]

        # Buscar dados do CDI
        cotacoes_cdi = self.get_cdi_history(start_date_str, end_date_str)
        if cotacoes_cdi is None:
            raise Exception("Erro ao buscar os dados do CDI.")

        # Concatenar os dados
        df_merged = pd.merge(cotacoes_acao, cotacoes_ibov, on='data', how='left')
        df_merged = pd.merge(df_merged, cotacoes_cdi, on='data', how='left')
        # Remover o sufixo "SA" do ticker
        df_merged['ticker'] = df_merged['ticker'].str.replace('.SA', '', regex=False)

        # Selecionar as colunas finais
        self.df_dados = df_merged[[
            'data', 'ticker', 'preco_fechamento_ajustado', 'high', 'low', 
            'open', 'volume', 'pct_ibov', 'fechamento_ibov', 'retorno_cdi'
        ]]

        self.df_dados = otimizar_tipos(self.df_dados)
        
        return self.df_dados
    # Garantir que o DataFrame esteja ordenado por 'data' dentro de cada 'ticker'
    
   
    def calcular_variacao_indicador(self, indicador):
        """
        Calcula a variação percentual de um indicador e adiciona uma nova coluna ao DataFrame.
        
        :param indicador: Nome do indicador existente no DataFrame.
        """
        coluna_variacao = f"{indicador}_variacao"
        
        # Verificar se o indicador existe no DataFrame
        if indicador not in self.df_dados.columns:
            raise Exception(f"O indicador '{indicador}' não existe no DataFrame.")

        grupo = self.df_dados.groupby('ticker', group_keys=False)
        self.df_dados[coluna_variacao] = grupo[indicador].pct_change(fill_method=None)
        self.df_dados.loc[self.df_dados[coluna_variacao] == 0.0, coluna_variacao] = pd.NA
        self.df_dados[coluna_variacao] = grupo[coluna_variacao].ffill()

    def adicionar_indicadores(self):
        # Verificar se o DataFrame `df_dados` foi criado
        if not hasattr(self, 'df_dados') or self.df_dados.empty:
            raise Exception("O DataFrame `df_dados` não foi criado. Execute `cria_data_frame` primeiro.")
        
        def calcula_indicadores_tecnicos():
            # Cálculo do RSI
            self.calcular_rsi(periodo=14)
            self.calcular_variacao_indicador(f'RSI_14')  # Calcular variação do RSI

            # Indicador de momento para 1 mês e 6 meses
            self.fazer_indicador_momento(meses=1)
            self.calcular_variacao_indicador('momento_1_meses')  # Calcular variação do momento de 1 mês
            self.fazer_indicador_momento(meses=6)
            self.calcular_variacao_indicador('momento_6_meses')  # Calcular variação do momento de 6 meses

            # Média móvel proporcional (curta: 7 dias, longa: 40 dias)
            self.media_movel_proporcao(mm_curta=7, mm_longa=40)
            self.calcular_variacao_indicador('mm_7_40')  # Calcular variação da proporção de médias móveis
            
        def calcular_beta_252():
            """
            Calcula o Beta de 252 dias para o ticker em relação ao IBOV e concatena ao DataFrame `self.df_dados`.
            """
            # Verificar se os dados necessários estão disponíveis
            if 'fechamento_ibov' not in self.df_dados.columns or 'preco_fechamento_ajustado' not in self.df_dados.columns:
                raise Exception("Os dados necessários para calcular o Beta não estão disponíveis no DataFrame.")

            # Calcular os retornos do ticker
            grupo = self.df_dados.groupby('ticker', group_keys=False)
            self.df_dados['retorno_ticker'] = grupo['preco_fechamento_ajustado'].ffill().pct_change()

            # Tratar valores inválidos diretamente
            self.df_dados.loc[self.df_dados['retorno_ticker'].isin([0, np.inf]), 'retorno_ticker'] = pd.NA
            
            # Garantir que todos os valores de retorno_ticker sejam preenchidos com ffill
            self.df_dados['retorno_ticker'] = grupo['retorno_ticker'].ffill()

            # Filtrar apenas as colunas necessárias
            df_beta = self.df_dados[['data', 'ticker', 'retorno_ticker', 'pct_ibov']].dropna().sort_values(by=['ticker', 'data'])

            # Lista para armazenar os betas calculados
            lista_betas = []

            # Calcular o Beta para cada ticker
            tickers = df_beta['ticker'].unique()
            for ticker in tickers:
                dados_ticker = df_beta[df_beta['ticker'] == ticker]

                if len(dados_ticker) >= 252:  # Garantir que há dados suficientes para calcular o Beta
                    # Usar uma regressão linear para calcular o Beta
                    X = dados_ticker['pct_ibov']
                    y = dados_ticker['retorno_ticker']
                    X = sm.add_constant(X)  # Adicionar uma constante para o modelo
                    modelo = RollingOLS(endog=y, exog=X, window=252, min_nobs=200)
                    resultados = modelo.fit()

                    # Extrair os valores do Beta
                    dados_ticker['beta_252'] = resultados.params['pct_ibov']
                    lista_betas.append(dados_ticker[['data', 'ticker', 'beta_252']])

            # Concatenar os resultados
            if lista_betas:
                df_betas = pd.concat(lista_betas, ignore_index=True)

                # Fazer o merge com o DataFrame principal
                self.df_dados = pd.merge(self.df_dados, df_betas, on=['data', 'ticker'], how='left')
            
            # Calcular a variação do beta_252
            grupo = self.df_dados.groupby('ticker', group_keys=False)
            self.df_dados['beta_252_variacao'] = grupo['beta_252'].ffill().pct_change()

            # Substituir valores 0.0 por NaN e preencher valores NaN com forward fill
            self.df_dados['beta_252_variacao'] = (self.df_dados['beta_252_variacao']
                .replace(0.0, pd.NA)
                .groupby(self.df_dados['ticker'])
                .ffill()
            )
        def calcula_indicadores_contabeis():
            """
            Função para calcular indicadores contábeis.
            """
            lista_indicadores_sem_rep = []
    
            # Ajuste no loop para o formato do dicionário fornecido
            indicadores = self.indicadores.get('indicadores', set())
            for indicador in indicadores:
                if indicador not in lista_indicadores_sem_rep:
                    lista_indicadores_sem_rep.append(indicador)
                    lendo_indicador = buscar_indicador(indicador)  # Buscar indicador no S3
                    if lendo_indicador is not None:
                        # Converter colunas para os tipos corretos
                        lendo_indicador['data'] = pd.to_datetime(lendo_indicador['data'])
                        lendo_indicador['ticker'] = lendo_indicador['ticker'].astype(str)
                        lendo_indicador['valor'] = lendo_indicador['valor'].astype(float)
                        
                        # Filtrar apenas os últimos 2 anos de dados
                        dois_anos_atras = pd.Timestamp.now() - pd.DateOffset(years=2)
                        lendo_indicador = lendo_indicador[lendo_indicador['data'] >= dois_anos_atras]

                        # Filtrar apenas os indicadores para os tickers presentes em self.df_dados
                        tickers_presentes = self.df_dados['ticker'].unique()
                        lendo_indicador = lendo_indicador[lendo_indicador['ticker'].isin(tickers_presentes)]

                        # Calcular a variação entre trimestres
                        lendo_indicador['variacao'] = lendo_indicador.groupby('ticker')['valor'].pct_change()
                        lendo_indicador['variacao'] = lendo_indicador['variacao'].replace(0.0, pd.NA)
                        
                        lendo_indicador['variacao'] = lendo_indicador.groupby('ticker')['variacao'].ffill()

                        # Selecionar as colunas finais
                        lendo_indicador = lendo_indicador[['data', 'ticker', 'valor', 'variacao']]
                        lendo_indicador.columns = ['data', 'ticker', indicador, f'{indicador}_variacao']

                        # Realizar o merge_asof com o DataFrame existente
                        self.df_dados['data'] = pd.to_datetime(self.df_dados['data'], errors='coerce')
                        self.df_dados.sort_values(by=['ticker', 'data'], inplace=True)
                        lendo_indicador.sort_values(by=['ticker', 'data'], inplace=True)
                        self.df_dados = pd.merge_asof(
                            self.df_dados,
                            lendo_indicador,
                            by='ticker',
                            on='data',
                            direction='backward'
                        )

            # Garantir que o DataFrame esteja ordenado por 'data' dentro de cada 'ticker'
            self.df_dados = self.df_dados.sort_values(by=['ticker', 'data']).reset_index(drop=True)
        
        

        calcula_indicadores_tecnicos()
        calcular_beta_252()
        calcula_indicadores_contabeis()
    
        # Aplicar a função para cada grupo de 'ticker'
        #self.df_dados = self.df_dados.groupby('ticker', group_keys=False).apply(calcula_signal_target)
        
        # Remover a coluna temporária 'preco_fechamento_futuro'
        print('passei aqui')
        #self.df_dados = self.df_dados.drop(columns=['preco_fechamento_futuro'])
        
        self.df_dados = self.df_dados.groupby('ticker', group_keys=False).apply(lambda x: x.tail(21))

        return self.df_dados

    def calcular_rsi(self, periodo=14):
        """
        Calcula o Índice de Força Relativa (RSI) e adiciona ao DataFrame.
        """
        delta = self.df_dados['preco_fechamento_ajustado'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        self.df_dados[f'RSI_{periodo}'] = 100 - (100 / (1 + rs))

    def fazer_indicador_momento(self, meses):
        """
        Calcula o indicador de momento para o período especificado (em meses) e adiciona ao DataFrame.
        """
        periodo = meses * 21  # Aproximadamente 21 dias úteis por mês
        coluna_momento = f'momento_{meses}_meses'
        self.df_dados[coluna_momento] = self.df_dados.groupby('ticker')['preco_fechamento_ajustado'].pct_change(periods=periodo)
        self.df_dados.loc[self.df_dados[coluna_momento] == 0, coluna_momento] = np.nan
        self.df_dados.loc[self.df_dados[coluna_momento] == np.inf, coluna_momento] = np.nan

    def media_movel_proporcao(self, mm_curta, mm_longa):
        """
        Calcula a proporção entre médias móveis curtas e longas e adiciona ao DataFrame.
        """
        self.df_dados['media_curta'] = self.df_dados.groupby('ticker')['preco_fechamento_ajustado'].rolling(
            window=mm_curta, min_periods=int(mm_curta * 0.8)).mean().reset_index(0, drop=True)
        self.df_dados['media_longa'] = self.df_dados.groupby('ticker')['preco_fechamento_ajustado'].rolling(
            window=mm_longa, min_periods=int(mm_longa * 0.8)).mean().reset_index(0, drop=True)
        self.df_dados['mm_7_40'] = self.df_dados['media_curta'] / self.df_dados['media_longa']
        self.df_dados.drop(columns=['media_curta', 'media_longa'], inplace=True)

        # Remover valores NaN
        self.df_dados.dropna(subset=['mm_7_40'], inplace=True)
        
    def gera_prediction(self):
        """
        Gera previsões com base nos dados e indicadores técnicos.
        Retorna um JSON com as predições.
        """
        try:
            # Criar o DataFrame com os dados e indicadores técnicos
            self.cria_data_frame()
            df_com_indicadores = self.adicionar_indicadores()

            # Formatar o ticker
            ticker_formatado = self.ticker.replace(".SA", "").replace(".sa", "")

            # Carregar scaler e modelos do S3
            lstm_scaler = ler_parametros_scaler_do_s3(ticker_formatado, 'lstm_scaler.pkl')
            model_regressor = buscar_modelo_no_s3(ticker_formatado, 'lstm_regression', modelo='keras')
            model_classification = buscar_modelo_no_s3(ticker_formatado, 'lstm_classification', modelo='keras')

            # Verificar se os modelos foram carregados corretamente
            if lstm_scaler is None or model_regressor is None or model_classification is None:
                raise Exception("Erro ao carregar scaler ou modelos do S3.")
            
            print(lstm_scaler.feature_names_in_)
            preco_fechamento_ajustado = df_com_indicadores['preco_fechamento_ajustado'].values[-1]
            df_com_indicadores = df_com_indicadores[lstm_scaler.feature_names_in_].ffill().bfill()
            # Escalar os dados
            scaled_data = lstm_scaler.transform(df_com_indicadores)
            # Redimensionar os dados para [batch_size, time_steps, features]
            scaled_data = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])
        
            # Fazer predições
            lstm_classification = model_classification.predict(scaled_data)
            lstm_classification = (lstm_classification > 0.5).astype(int).flatten()
            lstm_regression = model_regressor.predict(scaled_data).flatten()
            
            # Ajustar valores muito próximos de zero em lstm_regression[0]
            if 0.0 <= lstm_regression[0] < 0.1:
                lstm_regression[0] = 0.1  # Ajustar para 0.1% se for positivo e próximo de zero
            elif -0.1 < lstm_regression[0] < 0.0:
                lstm_regression[0] = -0.1  # Ajustar para -0.1% se for negativo e próximo de zero


            # Preparar o retorno em JSON
            predictions = []
            predictions.append({
                "data": datetime.today().strftime("%d/%m/%Y"),  # Data atual no formato dia/mês/ano
                "target": preco_fechamento_ajustado.round(2),
                "classificacao": "Compra" if int(lstm_classification[0]) == 1 else "Venda",
                "variacao": f"{round(float(lstm_regression[0]), 2)}"
            })
            print(predictions[0])

            # Retornar o JSON com o ticker como chave
            return {
                ticker_formatado: predictions
            }

        except Exception as e:
            print(f"Erro ao gerar previsões: {e}")
            return {
                "error": str(e)
            }


if __name__ == "__main__":
    dicionario_indicadores = {
            'indicadores': {
                'ebit_dl',
                'ROIC',
                'EBIT_Ativos',
                'ROE',
                'PSR'
            }
        }
    handler = DataAnalitcsHandler(ticker="AZUL4.SA", **dicionario_indicadores)
    resultado = handler.gera_prediction()


