import pandas as pd
import requests
import statsmodels.api as sm
from datetime import datetime

print('Iniciando o script da Apresentação...')

# ---------------------------------------------------------
# PASSO 1: Consumindo dados via API do Banco Central (BCB)
# ---------------------------------------------------------
print('-> 1. Buscando dados da taxa SELIC via API do Banco Central...')
# Código 4390: Taxa de juros - Selic acumulada no mês (% a.m.)
url_bcb = 'https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados?formato=json'
resposta = requests.get(url_bcb)
dados_json = resposta.json()

# Convertendo o JSON para um DataFrame do Pandas
df_api = pd.DataFrame(dados_json)

# Limpando e formatando os dados da API
# O BCB retorna strings, então convertemos para datetime e float
df_api['Data_API'] = pd.to_datetime(df_api['data'], format='%d/%m/%Y')
df_api['Selic_API'] = df_api['valor'].astype(float)
df_api = df_api[['Data_API', 'Selic_API']] # Mantendo apenas as colunas úteis

print(f'   Sucesso! {len(df_api)} registros importados da API.')

# ---------------------------------------------------------
# PASSO 2: Carregando a base de dados local
# ---------------------------------------------------------
print('-> 2. Carregando dados locais do arquivo IPCA.xlsx...')
df_local = pd.read_excel('IPCA.xlsx')
# Garantindo que a coluna Data também seja datetime
df_local['Data'] = pd.to_datetime(df_local['Data'])
print(f'   Sucesso! {len(df_local)} registros locais importados.')

# ---------------------------------------------------------
# PASSO 3: Realizando o MERGE (Cruzamento de Dados)
# ---------------------------------------------------------
print('-> 3. Realizando o MERGE entre os dados da API e os dados locais...')
# Vamos unir as duas tabelas usando as colunas de data correspondentes
df_final = pd.merge(left=df_local, 
                    right=df_api, 
                    left_on='Data', 
                    right_on='Data_API', 
                    how='inner') # Inner join: mantém apenas as datas que existem em ambas

# Removendo linhas com valores nulos nas variáveis que usaremos
df_final = df_final.dropna(subset=['Var_IPCA_Trans', 'Preco_Barril', 'Cambio', 'Selic_API'])
print(f'   Merge concluído! A base final ficou com {len(df_final)} linhas combinadas.')

# ---------------------------------------------------------
# PASSO 4: Criando e Estimando o Modelo Econométrico (OLS)
# ---------------------------------------------------------
print('-> 4. Estimando o modelo (IPCA Transportes vs. Petróleo, Câmbio e Selic da API)...\n')
# Variável Dependente (Y)
Y = df_final['Var_IPCA_Trans']

# Variáveis Explicativas (X)
X = df_final[['Preco_Barril', 'Cambio', 'Selic_API']]
X = sm.add_constant(X) # Adiciona o intercepto (alfa) na regressão

# Treinando o modelo Ordinary Least Squares (OLS)
modelo = sm.OLS(Y, X).fit()

# Exibindo o resultado final para o professor ver
print('='*60)
print('         RESULTADO DO MODELO (MERGE API + LOCAL)         ')
print('='*60)
print(modelo.summary().tables[1])
print('='*60)
print('Apresentação concluída com sucesso!')
