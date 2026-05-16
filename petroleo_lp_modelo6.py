# -*- coding: utf-8 -*-
"""
petroleo_lp_modelo6.py

Modelo de Local Projections para o TCC:
"Efeitos dos Choques do Preço do Petróleo sobre a Inflação no Brasil:
Evidências via Combustíveis, Petrobras, VAR e Local Projections entre 2003 e 2026"

Objetivo do script:
1. Estimar Local Projections acumuladas para o canal:
   petróleo em reais -> combustíveis -> IPCA Geral e IPCA Transportes.
2. Usar 3 defasagens como especificação principal.
3. Gerar robustez com 6 e 12 defasagens.
4. Dividir a ação/institucionalidade da Petrobras em três regimes:
   - 2003-2014: maior intervenção, suavização ou represamento.
   - 2015-2022: realinhamento e maior aderência à paridade internacional.
   - 2023 em diante: nova estratégia comercial da Petrobras.
5. Estimar testes de Wald para diferenças entre regimes.
6. Rodar LP-IV com Oil Supply News Shock como robustez de identificação,
   caso a série esteja disponível na base.
7. Exportar tabelas, gráficos e resumos em arquivos.

Observação importante:
Este script foi escrito para ser robusto a pequenas diferenças nos nomes das colunas.
Mesmo assim, revise a seção CONFIGURAÇÕES caso sua planilha use nomes diferentes.
"""

# ============================================================
# 0. PACOTES
# ============================================================

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================

# Caminho padrão usado no TCC.
# Se o arquivo não estiver nesse caminho, coloque o caminho correto aqui.
ARQUIVO_EXCEL = r"C:/Users/pedro/OneDrive/Documentos/TCC/IPCA.xlsx"

# Caso o arquivo esteja na mesma pasta do script, o código tenta este nome também.
ARQUIVO_EXCEL_ALTERNATIVO = "IPCA.xlsx"

# Aba do Excel.
# Use None para a primeira aba.
NOME_ABA = None

# Pasta de saída.
PASTA_SAIDA = Path("output_petroleo_lp_modelo6")
PASTA_TABELAS = PASTA_SAIDA / "tabelas"
PASTA_GRAFICOS = PASTA_SAIDA / "graficos"
PASTA_RESUMOS = PASTA_SAIDA / "resumos"

for pasta in [PASTA_SAIDA, PASTA_TABELAS, PASTA_GRAFICOS, PASTA_RESUMOS]:
    pasta.mkdir(parents=True, exist_ok=True)

# Horizonte máximo das respostas.
H_MAX = 24

# Horizonte principal de interpretação no TCC.
H_PRINCIPAL = 12

# Modelo principal e robustez.
LAGS_PRINCIPAL = 3
LAGS_ROBUSTEZ = [6, 12]

# Intervalo de confiança usado no TCC.
NIVEL_IC = 0.90
Z_IC = stats.norm.ppf(1 - (1 - NIVEL_IC) / 2)

# HAC/Newey-West:
# "h_plus_1": usa maxlags = h + 1.
# "min_3_h_plus_1": usa maxlags = max(3, h + 1).
MODO_HAC = "min_3_h_plus_1"

# Usar expectativa de inflação em nível como padrão.
# Isso evita tratar expectativa, que é taxa, como índice de preço.
USAR_EXPECTATIVA_EM_NIVEL = True

# Usar Selic em nível como padrão.
# Isso é defensável porque política monetária atua principalmente pelo patamar da taxa.
USAR_SELIC_EM_NIVEL = True

# Salvar gráficos.
SALVAR_GRAFICOS = True


# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def limpar_nome_coluna(nome: str) -> str:
    """
    Padroniza nomes de colunas para reduzir erros de digitação,
    acentos e diferenças de caixa.
    """
    nome = str(nome).strip()
    nome = nome.replace("\n", " ").replace("\r", " ")
    nome = re.sub(r"\s+", "_", nome)
    return nome


def normalizar_para_busca(texto: str) -> str:
    """
    Normaliza texto para busca flexível de colunas.
    """
    texto = str(texto).lower()
    substituicoes = {
        "á": "a", "à": "a", "ã": "a", "â": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u",
        "ç": "c"
    }
    for antigo, novo in substituicoes.items():
        texto = texto.replace(antigo, novo)
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto


def encontrar_coluna(df: pd.DataFrame, candidatos, obrigatoria=True):
    """
    Encontra uma coluna da base com base em uma lista de nomes candidatos.
    Retorna o nome real da coluna no DataFrame.
    """
    mapa = {normalizar_para_busca(c): c for c in df.columns}

    for cand in candidatos:
        cand_norm = normalizar_para_busca(cand)
        if cand_norm in mapa:
            return mapa[cand_norm]

    # Busca parcial.
    for cand in candidatos:
        cand_norm = normalizar_para_busca(cand)
        for col_norm, col_real in mapa.items():
            if cand_norm in col_norm or col_norm in cand_norm:
                return col_real

    if obrigatoria:
        raise ValueError(
            f"Não encontrei nenhuma coluna compatível com: {candidatos}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )
    return None


def dlog_100(serie: pd.Series) -> pd.Series:
    """
    Variação logarítmica multiplicada por 100.
    """
    serie = pd.to_numeric(serie, errors="coerce")
    serie = serie.where(serie > 0)
    return 100 * (np.log(serie) - np.log(serie.shift(1)))


def diff_pontos(serie: pd.Series) -> pd.Series:
    """
    Diferença simples, útil para taxas em pontos percentuais.
    """
    serie = pd.to_numeric(serie, errors="coerce")
    return serie.diff()


def padronizar(serie: pd.Series) -> pd.Series:
    """
    Padroniza uma variável para média zero e desvio-padrão um.
    """
    serie = pd.to_numeric(serie, errors="coerce")
    desvio = serie.std(skipna=True)
    if desvio == 0 or np.isnan(desvio):
        return serie * np.nan
    return (serie - serie.mean(skipna=True)) / desvio


def maxlags_hac(h: int) -> int:
    """
    Define o bandwidth do HAC/Newey-West por horizonte.
    Em respostas acumuladas há sobreposição temporal, então o HAC precisa
    crescer com o horizonte.
    """
    if MODO_HAC == "h_plus_1":
        return max(1, h + 1)
    if MODO_HAC == "min_3_h_plus_1":
        return max(3, h + 1)
    return max(1, h + 1)


def criar_lags(df: pd.DataFrame, variaveis, lags: int) -> pd.DataFrame:
    """
    Cria defasagens das variáveis informadas.
    """
    out = pd.DataFrame(index=df.index)
    for var in variaveis:
        if var not in df.columns:
            continue
        for lag in range(1, lags + 1):
            out[f"{var}_lag{lag}"] = df[var].shift(lag)
    return out


def resposta_acumulada(df: pd.DataFrame, var: str, h: int) -> pd.Series:
    """
    Soma acumulada da variável de resposta entre t e t+h.
    """
    acumulada = pd.Series(0.0, index=df.index)
    for j in range(0, h + 1):
        acumulada = acumulada + df[var].shift(-j)
    return acumulada


def resposta_pontual(df: pd.DataFrame, var: str, h: int) -> pd.Series:
    """
    Resposta pontual no horizonte h.
    """
    return df[var].shift(-h)


def estimar_ols_hac(y: pd.Series, X: pd.DataFrame, h: int):
    """
    Estima OLS com erros-padrão HAC/Newey-West.
    """
    dados = pd.concat([y.rename("y"), X], axis=1)
    dados = dados.replace([np.inf, -np.inf], np.nan).dropna()

    if dados.shape[0] < X.shape[1] + 10:
        return None, dados

    y_est = dados["y"]
    X_est = sm.add_constant(dados.drop(columns=["y"]), has_constant="add")

    try:
        modelo = sm.OLS(y_est, X_est).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags_hac(h)}
        )
        return modelo, dados
    except Exception:
        return None, dados


def intervalo_confianca(coef, se):
    """
    Retorna intervalo de confiança de 90%.
    """
    return coef - Z_IC * se, coef + Z_IC * se


def plotar_resposta(tabela: pd.DataFrame, titulo: str, arquivo_saida: Path):
    """
    Gera gráfico de resposta acumulada ou pontual.
    """
    if tabela.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(tabela["h"], tabela["coef"], linewidth=2)
    ax.fill_between(
        tabela["h"],
        tabela["ic_inf"],
        tabela["ic_sup"],
        alpha=0.25
    )
    ax.axhline(0, linewidth=1)
    ax.set_title(titulo)
    ax.set_xlabel("Horizonte em meses")
    ax.set_ylabel("Resposta acumulada")
    ax.grid(True, alpha=0.3)

    nota = (
        f"Choque de 1 desvio-padrão. IC {int(NIVEL_IC*100)}%. "
        f"HAC/Newey-West: {MODO_HAC}. "
        f"Modelo principal: {LAGS_PRINCIPAL} defasagens."
    )
    fig.text(0.01, 0.01, nota, fontsize=8)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(arquivo_saida, dpi=300)
    plt.close(fig)


def nome_seguro(texto: str) -> str:
    """
    Cria nomes seguros para arquivos.
    """
    texto = normalizar_para_busca(texto)
    return texto[:160]


# ============================================================
# 3. CARREGAMENTO DA BASE
# ============================================================

def carregar_base() -> pd.DataFrame:
    """
    Carrega a planilha principal do TCC.
    """
    caminho = Path(ARQUIVO_EXCEL)

    if not caminho.exists():
        caminho_alt = Path(ARQUIVO_EXCEL_ALTERNATIVO)
        if caminho_alt.exists():
            caminho = caminho_alt
        else:
            raise FileNotFoundError(
                "Não encontrei o arquivo Excel.\n"
                f"Tentei:\n{ARQUIVO_EXCEL}\n{ARQUIVO_EXCEL_ALTERNATIVO}\n"
                "Altere a variável ARQUIVO_EXCEL no início do script."
            )

    if NOME_ABA is None:
        df = pd.read_excel(caminho)
    else:
        df = pd.read_excel(caminho, sheet_name=NOME_ABA)

    df.columns = [limpar_nome_coluna(c) for c in df.columns]

    col_data = encontrar_coluna(df, ["Data", "date", "periodo", "mes"], obrigatoria=True)
    df[col_data] = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[col_data]).copy()
    df = df.sort_values(col_data)
    df = df.set_index(col_data)
    df.index.name = "Data"

    # Remove duplicatas de mês, caso existam.
    df = df[~df.index.duplicated(keep="last")]

    return df


# ============================================================
# 4. PREPARAÇÃO DAS VARIÁVEIS DO TCC
# ============================================================

def preparar_variaveis(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis transformadas usadas nas LPs.
    """
    df = df_raw.copy()

    # Colunas principais com busca flexível.
    col_petroleo = encontrar_coluna(
        df,
        ["Preco_Barril", "Brent", "Petroleo", "DCOILBRENTEU", "preco_petroleo"],
        obrigatoria=True
    )

    col_cambio = encontrar_coluna(
        df,
        ["Cambio", "Câmbio", "Taxa_Cambio", "USD_BRL", "Dolar"],
        obrigatoria=True
    )

    col_diesel = encontrar_coluna(
        df,
        ["Oleo_diesel", "Óleo_diesel", "Diesel", "oleo diesel"],
        obrigatoria=True
    )

    col_gasolina = encontrar_coluna(
        df,
        ["Gasolina", "Gasolina_consumidor", "Gasolina_C", "GasolinaBrasil_media"],
        obrigatoria=True
    )

    col_etanol = encontrar_coluna(
        df,
        ["Etanol", "Alcool", "Álcool"],
        obrigatoria=True
    )

    col_refinaria = encontrar_coluna(
        df,
        [
            "GasolinaABrasil_media",
            "GasolinaA",
            "Gasolina_A",
            "Gasolina_refinaria",
            "Gasolina_de_refinaria",
            "Preco_refinaria",
            "Preço_refinaria"
        ],
        obrigatoria=True
    )

    col_ipca_geral = encontrar_coluna(
        df,
        ["IPCA_Geral", "IPCA_Brasil", "Var_IPCA_Brasil", "IPCA", "IPCA Brasil"],
        obrigatoria=True
    )

    col_ipca_trans = encontrar_coluna(
        df,
        ["IPCA_Trans", "IPCA_Transportes", "Var_IPCA_trans", "IPCA trans", "Transportes"],
        obrigatoria=True
    )

    col_atividade = encontrar_coluna(
        df,
        ["Atividade", "IBC_BR", "IBC-Br", "IBCBr", "atividade_economica"],
        obrigatoria=True
    )

    col_selic = encontrar_coluna(
        df,
        ["Selic", "Meta_Selic", "Selic_meta", "Taxa_Selic"],
        obrigatoria=True
    )

    col_expectativa = encontrar_coluna(
        df,
        ["Expectativa_Inflacao", "Expectativa_Inflação", "Focus_Inflacao_12m", "Focus_IPCA_12m"],
        obrigatoria=True
    )

    col_stringency = encontrar_coluna(
        df,
        ["Stringency", "Stringency_Index", "Oxford_Stringency", "stringency_index"],
        obrigatoria=False
    )

    col_news = encontrar_coluna(
        df,
        [
            "Oil_Supply_News_Shock",
            "Oil Supply News Shock",
            "Supply_News_Shock",
            "Kanzig",
            "Kanzig_Shock",
            "OSNS"
        ],
        obrigatoria=False
    )

    # Converte numéricas.
    for col in [
        col_petroleo, col_cambio, col_diesel, col_gasolina, col_etanol,
        col_refinaria, col_ipca_geral, col_ipca_trans, col_atividade,
        col_selic, col_expectativa
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if col_stringency is not None:
        df[col_stringency] = pd.to_numeric(df[col_stringency], errors="coerce")
    else:
        df["Stringency_Index_ausente"] = 0.0
        col_stringency = "Stringency_Index_ausente"

    if col_news is not None:
        df[col_news] = pd.to_numeric(df[col_news], errors="coerce")

    # Variáveis em nível com nomes padronizados.
    df["brent_usd_nivel"] = df[col_petroleo]
    df["cambio_nivel"] = df[col_cambio]
    df["brent_brl_nivel"] = df["brent_usd_nivel"] * df["cambio_nivel"]

    df["diesel_nivel"] = df[col_diesel]
    df["gasolina_consumidor_nivel"] = df[col_gasolina]
    df["etanol_nivel"] = df[col_etanol]
    df["gasolina_refinaria_nivel"] = df[col_refinaria]

    df["ipca_geral_original"] = df[col_ipca_geral]
    df["ipca_transportes_original"] = df[col_ipca_trans]

    df["atividade_nivel"] = df[col_atividade]
    df["selic_nivel"] = df[col_selic]
    df["expectativa_inflacao_nivel"] = df[col_expectativa]
    df["stringency"] = df[col_stringency]

    if col_news is not None:
        df["oil_supply_news_shock"] = df[col_news]

    # Transformações principais.
    df["dlog_brent_usd"] = dlog_100(df["brent_usd_nivel"])
    df["dlog_cambio"] = dlog_100(df["cambio_nivel"])
    df["dlog_brent_brl"] = dlog_100(df["brent_brl_nivel"])

    df["dlog_diesel"] = dlog_100(df["diesel_nivel"])
    df["dlog_gasolina_consumidor"] = dlog_100(df["gasolina_consumidor_nivel"])
    df["dlog_etanol"] = dlog_100(df["etanol_nivel"])
    df["dlog_gasolina_refinaria"] = dlog_100(df["gasolina_refinaria_nivel"])

    # IPCA:
    # Se a série parecer índice em nível, usa dlog.
    # Se parecer taxa mensal já pronta, preserva.
    # Regra prática: valores médios muito altos indicam índice em nível.
    if df["ipca_geral_original"].median(skipna=True) > 20:
        df["ipca_geral"] = dlog_100(df["ipca_geral_original"])
    else:
        df["ipca_geral"] = df["ipca_geral_original"]

    if df["ipca_transportes_original"].median(skipna=True) > 20:
        df["ipca_transportes"] = dlog_100(df["ipca_transportes_original"])
    else:
        df["ipca_transportes"] = df["ipca_transportes_original"]

    # Atividade: como controle final do TCC.
    df["dlog_atividade"] = dlog_100(df["atividade_nivel"])

    # Selic e expectativa.
    if USAR_SELIC_EM_NIVEL:
        df["selic_controle"] = df["selic_nivel"]
    else:
        df["selic_controle"] = diff_pontos(df["selic_nivel"])

    if USAR_EXPECTATIVA_EM_NIVEL:
        df["expectativa_controle"] = df["expectativa_inflacao_nivel"]
    else:
        df["expectativa_controle"] = diff_pontos(df["expectativa_inflacao_nivel"])

    # Dummies mensais.
    df["mes"] = df.index.month
    dummies_mes = pd.get_dummies(df["mes"], prefix="mes", drop_first=True, dtype=float)
    df = pd.concat([df, dummies_mes], axis=1)

    # Regimes Petrobras em três partes.
    df["regime_2003_2014"] = ((df.index >= "2003-01-01") & (df.index <= "2014-12-31")).astype(int)
    df["regime_2015_2022"] = ((df.index >= "2015-01-01") & (df.index <= "2022-12-31")).astype(int)
    df["regime_2023_2026"] = (df.index >= "2023-01-01").astype(int)

    # Descrição dos regimes para exportar.
    regimes = pd.DataFrame({
        "regime": ["2003-2014", "2015-2022", "2023 em diante"],
        "interpretacao": [
            "Maior intervenção, suavização ou represamento dos preços.",
            "Realinhamento e maior aderência à paridade internacional.",
            "Nova estratégia comercial da Petrobras."
        ],
        "observacao": [
            "Regime empírico aproximado, não representa regra formal única.",
            "Inclui transição e vigência mais forte da PPI; não deve ser tratado como homogêneo perfeito.",
            "Período curto; deve ser interpretado com cautela por menor número de observações."
        ]
    })
    regimes.to_excel(PASTA_TABELAS / "descricao_regimes_petroleo_lp_modelo6.xlsx", index=False)

    # Amostra a partir de 2003.
    df = df[df.index >= "2003-01-01"].copy()

    return df


# ============================================================
# 5. LOCAL PROJECTIONS PRINCIPAL
# ============================================================

def montar_matriz_controles(
    df: pd.DataFrame,
    response: str,
    impulse: str,
    lags: int,
    incluir_controles=True
) -> pd.DataFrame:
    """
    Controles finais do TCC:
    1. Atividade econômica.
    2. Selic.
    3. Expectativas de inflação.
    4. Dummies mensais.
    5. Stringency Index.
    6. Defasagens da variável dependente.
    7. Defasagens do choque.

    Observação:
    Também defasamos os controles macroeconômicos para evitar simultaneidade excessiva.
    """
    controles = pd.DataFrame(index=df.index)

    # Defasagens da variável dependente e do choque.
    vars_lag = [response, impulse]

    # Controles macroeconômicos finais.
    if incluir_controles:
        macro_controles = ["dlog_atividade", "selic_controle", "expectativa_controle", "stringency"]
        vars_lag += macro_controles

    controles = pd.concat([controles, criar_lags(df, vars_lag, lags)], axis=1)

    # Dummies mensais contemporâneas.
    dummies_mes = [c for c in df.columns if c.startswith("mes_")]
    controles = pd.concat([controles, df[dummies_mes]], axis=1)

    return controles


def local_projection(
    df: pd.DataFrame,
    impulse: str,
    response: str,
    lags: int = LAGS_PRINCIPAL,
    h_max: int = H_MAX,
    acumulada: bool = True,
    tag_modelo: str = "principal"
) -> pd.DataFrame:
    """
    Estima Local Projection para uma combinação impulso-resposta.
    """
    resultados = []

    impulse_std = f"{impulse}_std"
    dados = df.copy()
    dados[impulse_std] = padronizar(dados[impulse])

    for h in range(0, h_max + 1):
        if acumulada:
            y = resposta_acumulada(dados, response, h)
        else:
            y = resposta_pontual(dados, response, h)

        X_base = pd.DataFrame(index=dados.index)
        X_base[impulse_std] = dados[impulse_std]

        controles = montar_matriz_controles(
            dados,
            response=response,
            impulse=impulse,
            lags=lags,
            incluir_controles=True
        )

        X = pd.concat([X_base, controles], axis=1)

        modelo, dados_est = estimar_ols_hac(y, X, h)

        if modelo is None or impulse_std not in modelo.params.index:
            continue

        coef = modelo.params[impulse_std]
        se = modelo.bse[impulse_std]
        t_stat = modelo.tvalues[impulse_std]
        p_valor = modelo.pvalues[impulse_std]
        ic_inf, ic_sup = intervalo_confianca(coef, se)

        resultados.append({
            "modelo": tag_modelo,
            "tipo": "LP acumulada" if acumulada else "LP pontual",
            "impulso": impulse,
            "resposta": response,
            "h": h,
            "lags": lags,
            "coef": coef,
            "erro_padrao_hac": se,
            "t": t_stat,
            "p_valor": p_valor,
            "ic_inf": ic_inf,
            "ic_sup": ic_sup,
            "significativo_90": (ic_inf > 0) or (ic_sup < 0),
            "n_obs": int(modelo.nobs),
            "r2": modelo.rsquared,
            "hac_maxlags": maxlags_hac(h)
        })

    return pd.DataFrame(resultados)


# ============================================================
# 6. LOCAL PROJECTIONS COM 3 REGIMES PETROBRAS
# ============================================================

def wald_diferenca_parametros(modelo, nome_a: str, nome_b: str):
    """
    Teste de Wald para H0: beta_a = beta_b.
    Retorna estatística, p-valor e diferença.
    """
    params = list(modelo.params.index)
    if nome_a not in params or nome_b not in params:
        return np.nan, np.nan, np.nan

    idx_a = params.index(nome_a)
    idx_b = params.index(nome_b)

    R = np.zeros((1, len(params)))
    R[0, idx_a] = 1
    R[0, idx_b] = -1

    try:
        teste = modelo.wald_test(R, scalar=True)
        stat = float(teste.statistic)
        pval = float(teste.pvalue)
        diff = float(modelo.params[nome_a] - modelo.params[nome_b])
        return stat, pval, diff
    except Exception:
        return np.nan, np.nan, np.nan


def local_projection_regimes_petrobras(
    df: pd.DataFrame,
    impulse: str,
    response: str,
    lags: int = LAGS_PRINCIPAL,
    h_max: int = H_MAX,
    acumulada: bool = True,
    tag_modelo: str = "regimes_petrobras_3"
):
    pass


def local_projection_regimes_petrobras(
    df: pd.DataFrame,
    impulse: str,
    response: str,
    lags: int = LAGS_PRINCIPAL,
    h_max: int = H_MAX,
    acumulada: bool = True,
    tag_modelo: str = "regimes_petrobras_3"
):
    """
    Estima LP com três regimes Petrobras:
    - 2003-2014
    - 2015-2022
    - 2023 em diante

    O modelo estima interações entre o choque padronizado e cada regime.
    Depois testa diferenças par-a-par por Wald.
    """
    resultados = []
    resultados_wald = []

    dados = df.copy()
    impulse_std = f"{impulse}_std"
    dados[impulse_std] = padronizar(dados[impulse])

    regime_cols = {
        "2003_2014": "regime_2003_2014",
        "2015_2022": "regime_2015_2022",
        "2023_2026": "regime_2023_2026"
    }

    interacoes = {}
    for nome_regime, col_regime in regime_cols.items():
        nome_interacao = f"{impulse_std}_x_{nome_regime}"
        dados[nome_interacao] = dados[impulse_std] * dados[col_regime]
        interacoes[nome_regime] = nome_interacao

    for h in range(0, h_max + 1):
        if acumulada:
            y = resposta_acumulada(dados, response, h)
        else:
            y = resposta_pontual(dados, response, h)

        X_base = dados[list(interacoes.values())].copy()

        controles = montar_matriz_controles(
            dados,
            response=response,
            impulse=impulse,
            lags=lags,
            incluir_controles=True
        )

        X = pd.concat([X_base, controles], axis=1)

        modelo, dados_est = estimar_ols_hac(y, X, h)

        if modelo is None:
            continue

        for nome_regime, nome_interacao in interacoes.items():
            if nome_interacao not in modelo.params.index:
                continue

            coef = modelo.params[nome_interacao]
            se = modelo.bse[nome_interacao]
            t_stat = modelo.tvalues[nome_interacao]
            p_valor = modelo.pvalues[nome_interacao]
            ic_inf, ic_sup = intervalo_confianca(coef, se)

            resultados.append({
                "modelo": tag_modelo,
                "tipo": "LP acumulada por regime Petrobras" if acumulada else "LP pontual por regime Petrobras",
                "impulso": impulse,
                "resposta": response,
                "regime": nome_regime,
                "h": h,
                "lags": lags,
                "coef": coef,
                "erro_padrao_hac": se,
                "t": t_stat,
                "p_valor": p_valor,
                "ic_inf": ic_inf,
                "ic_sup": ic_sup,
                "significativo_90": (ic_inf > 0) or (ic_sup < 0),
                "n_obs": int(modelo.nobs),
                "r2": modelo.rsquared,
                "hac_maxlags": maxlags_hac(h)
            })

        pares = [
            ("2003_2014", "2015_2022"),
            ("2015_2022", "2023_2026"),
            ("2003_2014", "2023_2026")
        ]

        for a, b in pares:
            stat, pval, diff = wald_diferenca_parametros(
                modelo,
                interacoes[a],
                interacoes[b]
            )
            resultados_wald.append({
                "modelo": tag_modelo,
                "impulso": impulse,
                "resposta": response,
                "h": h,
                "lags": lags,
                "comparacao": f"{a} vs {b}",
                "diferenca_coef": diff,
                "wald_stat": stat,
                "p_valor_wald": pval,
                "diferenca_significativa_10": bool(pval < 0.10) if pd.notna(pval) else False,
                "n_obs": int(modelo.nobs),
                "hac_maxlags": maxlags_hac(h)
            })

    return pd.DataFrame(resultados), pd.DataFrame(resultados_wald)


def plotar_regimes(tabela: pd.DataFrame, titulo: str, arquivo_saida: Path):
    """
    Gráfico com as respostas por três regimes Petrobras.
    """
    if tabela.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for regime in tabela["regime"].unique():
        sub = tabela[tabela["regime"] == regime].sort_values("h")
        ax.plot(sub["h"], sub["coef"], linewidth=2, label=regime)
        ax.fill_between(sub["h"], sub["ic_inf"], sub["ic_sup"], alpha=0.15)

    ax.axhline(0, linewidth=1)
    ax.set_title(titulo)
    ax.set_xlabel("Horizonte em meses")
    ax.set_ylabel("Resposta acumulada")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Regime Petrobras")

    nota = (
        f"Choque de 1 desvio-padrão. IC {int(NIVEL_IC*100)}%. "
        "Regimes: 2003-2014, 2015-2022 e 2023 em diante. "
        "Interpretação cautelosa para 2023+ por menor amostra."
    )
    fig.text(0.01, 0.01, nota, fontsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(arquivo_saida, dpi=300)
    plt.close(fig)


# ============================================================
# 7. LP-IV COM OIL SUPPLY NEWS SHOCK
# ============================================================

def local_projection_iv(
    df: pd.DataFrame,
    impulse: str,
    instrument: str,
    response: str,
    lags: int = LAGS_PRINCIPAL,
    h_max: int = H_MAX,
    acumulada: bool = True,
    tag_modelo: str = "lp_iv_oil_supply_news"
) -> pd.DataFrame:
    """
    LP-IV simples por 2SLS manual:
    1º estágio: impulso padronizado ~ instrumento padronizado + controles
    2º estágio: resposta acumulada ~ impulso previsto + controles

    Observação:
    A inferência do 2º estágio usa HAC, mas esta implementação manual não corrige
    todos os detalhes de generated regressor como pacotes IV especializados.
    Para TCC, serve como robustez operacional. Para artigo, recomenda-se linearmodels.iv.IV2SLS.
    """
    resultados = []

    if instrument not in df.columns:
        print(f"[AVISO] Instrumento {instrument} não encontrado. LP-IV não será estimada.")
        return pd.DataFrame()

    dados = df.copy()
    impulse_std = f"{impulse}_std"
    instrument_std = f"{instrument}_std"

    dados[impulse_std] = padronizar(dados[impulse])
    dados[instrument_std] = padronizar(dados[instrument])

    for h in range(0, h_max + 1):
        if acumulada:
            y = resposta_acumulada(dados, response, h)
        else:
            y = resposta_pontual(dados, response, h)

        controles = montar_matriz_controles(
            dados,
            response=response,
            impulse=impulse,
            lags=lags,
            incluir_controles=True
        )

        # Primeiro estágio.
        X1 = pd.concat([dados[[instrument_std]], controles], axis=1)
        y1 = dados[impulse_std]

        modelo_1, dados_1 = estimar_ols_hac(y1, X1, h)

        if modelo_1 is None or instrument_std not in modelo_1.params.index:
            continue

        X1_est = sm.add_constant(dados_1.drop(columns=["y"]), has_constant="add")
        previsto = pd.Series(modelo_1.predict(X1_est), index=dados_1.index, name=f"{impulse}_iv_previsto")

        # F do primeiro estágio para o instrumento.
        try:
            f_test = modelo_1.f_test(f"{instrument_std} = 0")
            f_stat = float(f_test.fvalue)
            f_pval = float(f_test.pvalue)
        except Exception:
            # Como alternativa, t² do coeficiente do instrumento.
            t_inst = modelo_1.tvalues[instrument_std]
            f_stat = float(t_inst ** 2)
            f_pval = float(modelo_1.pvalues[instrument_std])

        # Segundo estágio.
        X2 = pd.concat([previsto, controles], axis=1)
        modelo_2, dados_2 = estimar_ols_hac(y, X2, h)

        if modelo_2 is None or f"{impulse}_iv_previsto" not in modelo_2.params.index:
            continue

        nome_iv = f"{impulse}_iv_previsto"
        coef = modelo_2.params[nome_iv]
        se = modelo_2.bse[nome_iv]
        t_stat = modelo_2.tvalues[nome_iv]
        p_valor = modelo_2.pvalues[nome_iv]
        ic_inf, ic_sup = intervalo_confianca(coef, se)

        resultados.append({
            "modelo": tag_modelo,
            "tipo": "LP-IV acumulada" if acumulada else "LP-IV pontual",
            "impulso_instrumentado": impulse,
            "instrumento": instrument,
            "resposta": response,
            "h": h,
            "lags": lags,
            "coef_iv": coef,
            "erro_padrao_hac": se,
            "t": t_stat,
            "p_valor": p_valor,
            "ic_inf": ic_inf,
            "ic_sup": ic_sup,
            "significativo_90": (ic_inf > 0) or (ic_sup < 0),
            "f_primeiro_estagio": f_stat,
            "p_primeiro_estagio": f_pval,
            "r2_primeiro_estagio": modelo_1.rsquared,
            "n_obs": int(modelo_2.nobs),
            "hac_maxlags": maxlags_hac(h)
        })

    return pd.DataFrame(resultados)


# ============================================================
# 8. EXECUÇÃO DOS MODELOS
# ============================================================

def executar_modelos():
    """
    Executa todo o pipeline do modelo 6.
    """
    print("Carregando base...")
    df_raw = carregar_base()

    print("Preparando variáveis...")
    df = preparar_variaveis(df_raw)

    # Salva base transformada.
    df.to_excel(PASTA_TABELAS / "base_transformada_petroleo_lp_modelo6.xlsx")

    # Variáveis principais.
    impulsos_petroleo = ["dlog_brent_brl"]
    combustiveis = [
        "dlog_diesel",
        "dlog_gasolina_consumidor",
        "dlog_etanol",
        "dlog_gasolina_refinaria"
    ]
    inflacoes = ["ipca_geral", "ipca_transportes"]

    # --------------------------------------------------------
    # 8.1 Modelo principal: petróleo -> combustíveis
    # --------------------------------------------------------
    resultados_lp = []

    print("Estimando LP principal: petróleo em reais -> combustíveis...")
    for impulso in impulsos_petroleo:
        for resposta in combustiveis:
            tab = local_projection(
                df,
                impulse=impulso,
                response=resposta,
                lags=LAGS_PRINCIPAL,
                h_max=H_MAX,
                acumulada=True,
                tag_modelo="principal_3_lags_petroleo_combustiveis"
            )
            resultados_lp.append(tab)

            if SALVAR_GRAFICOS and not tab.empty:
                titulo = f"LP acumulada: {impulso} → {resposta}"
                arquivo = PASTA_GRAFICOS / f"lp_principal_{nome_seguro(impulso)}_para_{nome_seguro(resposta)}.png"
                plotar_resposta(tab, titulo, arquivo)

    # --------------------------------------------------------
    # 8.2 Modelo principal: combustíveis -> inflação
    # --------------------------------------------------------
    print("Estimando LP principal: combustíveis -> IPCA Geral e IPCA Transportes...")
    for impulso in combustiveis:
        for resposta in inflacoes:
            tab = local_projection(
                df,
                impulse=impulso,
                response=resposta,
                lags=LAGS_PRINCIPAL,
                h_max=H_MAX,
                acumulada=True,
                tag_modelo="principal_3_lags_combustiveis_ipca"
            )
            resultados_lp.append(tab)

            if SALVAR_GRAFICOS and not tab.empty:
                titulo = f"LP acumulada: {impulso} → {resposta}"
                arquivo = PASTA_GRAFICOS / f"lp_principal_{nome_seguro(impulso)}_para_{nome_seguro(resposta)}.png"
                plotar_resposta(tab, titulo, arquivo)

    # --------------------------------------------------------
    # 8.3 Modelo principal: petróleo em reais -> inflação
    # --------------------------------------------------------
    print("Estimando LP principal: petróleo em reais -> inflação...")
    for impulso in impulsos_petroleo:
        for resposta in inflacoes:
            tab = local_projection(
                df,
                impulse=impulso,
                response=resposta,
                lags=LAGS_PRINCIPAL,
                h_max=H_MAX,
                acumulada=True,
                tag_modelo="principal_3_lags_petroleo_ipca"
            )
            resultados_lp.append(tab)

            if SALVAR_GRAFICOS and not tab.empty:
                titulo = f"LP acumulada: {impulso} → {resposta}"
                arquivo = PASTA_GRAFICOS / f"lp_principal_{nome_seguro(impulso)}_para_{nome_seguro(resposta)}.png"
                plotar_resposta(tab, titulo, arquivo)

    resultados_lp_df = pd.concat(resultados_lp, ignore_index=True) if resultados_lp else pd.DataFrame()
    resultados_lp_df.to_excel(PASTA_TABELAS / "resultados_lp_principal_3_lags.xlsx", index=False)

    # Tabela dos horizontes mais importantes.
    if not resultados_lp_df.empty:
        tabela_horizontes = resultados_lp_df[resultados_lp_df["h"].isin([3, 6, 12, 24])].copy()
        tabela_horizontes.to_excel(PASTA_TABELAS / "tabela_lp_principal_h3_h6_h12_h24.xlsx", index=False)

        tabela_h12 = resultados_lp_df[resultados_lp_df["h"] == H_PRINCIPAL].copy()
        tabela_h12 = tabela_h12.sort_values(["significativo_90", "resposta", "impulso"], ascending=[False, True, True])
        tabela_h12.to_excel(PASTA_TABELAS / "ranking_resultados_lp_h12.xlsx", index=False)

    # --------------------------------------------------------
    # 8.4 Robustez com 6 e 12 defasagens
    # --------------------------------------------------------
    print("Estimando robustez com 6 e 12 defasagens...")
    resultados_robustez = []

    relacoes_principais_robustez = [
        ("dlog_brent_brl", "dlog_diesel"),
        ("dlog_brent_brl", "dlog_gasolina_refinaria"),
        ("dlog_brent_brl", "ipca_transportes"),
        ("dlog_brent_brl", "ipca_geral"),
        ("dlog_gasolina_consumidor", "ipca_transportes"),
        ("dlog_etanol", "ipca_transportes"),
        ("dlog_gasolina_refinaria", "ipca_transportes"),
        ("dlog_gasolina_consumidor", "ipca_geral")
    ]

    for lags in LAGS_ROBUSTEZ:
        for impulso, resposta in relacoes_principais_robustez:
            tab = local_projection(
                df,
                impulse=impulso,
                response=resposta,
                lags=lags,
                h_max=H_MAX,
                acumulada=True,
                tag_modelo=f"robustez_{lags}_lags"
            )
            resultados_robustez.append(tab)

    resultados_robustez_df = pd.concat(resultados_robustez, ignore_index=True) if resultados_robustez else pd.DataFrame()
    resultados_robustez_df.to_excel(PASTA_TABELAS / "resultados_lp_robustez_6_12_lags.xlsx", index=False)

    if not resultados_robustez_df.empty:
        resultados_robustez_df[resultados_robustez_df["h"].isin([3, 6, 12, 24])].to_excel(
            PASTA_TABELAS / "tabela_lp_robustez_6_12_lags_h3_h6_h12_h24.xlsx",
            index=False
        )

    # --------------------------------------------------------
    # 8.5 Petrobras em três regimes
    # --------------------------------------------------------
    print("Estimando LP por três regimes Petrobras...")
    resultados_regimes = []
    resultados_wald = []

    relacoes_regimes = [
        ("dlog_brent_brl", "dlog_diesel"),
        ("dlog_brent_brl", "dlog_gasolina_refinaria"),
        ("dlog_gasolina_refinaria", "ipca_transportes"),
        ("dlog_gasolina_consumidor", "ipca_transportes"),
        ("dlog_etanol", "ipca_transportes"),
        ("dlog_brent_brl", "ipca_transportes"),
        ("dlog_brent_brl", "ipca_geral")
    ]

    for impulso, resposta in relacoes_regimes:
        tab_reg, tab_wald = local_projection_regimes_petrobras(
            df,
            impulse=impulso,
            response=resposta,
            lags=LAGS_PRINCIPAL,
            h_max=H_MAX,
            acumulada=True,
            tag_modelo="regimes_petrobras_3_periodos"
        )

        resultados_regimes.append(tab_reg)
        resultados_wald.append(tab_wald)

        if SALVAR_GRAFICOS and not tab_reg.empty:
            titulo = f"LP por regimes Petrobras: {impulso} → {resposta}"
            arquivo = PASTA_GRAFICOS / f"lp_regimes_{nome_seguro(impulso)}_para_{nome_seguro(resposta)}.png"
            plotar_regimes(tab_reg, titulo, arquivo)

    resultados_regimes_df = pd.concat(resultados_regimes, ignore_index=True) if resultados_regimes else pd.DataFrame()
    resultados_wald_df = pd.concat(resultados_wald, ignore_index=True) if resultados_wald else pd.DataFrame()

    resultados_regimes_df.to_excel(PASTA_TABELAS / "resultados_lp_regimes_petrobras_3_periodos.xlsx", index=False)
    resultados_wald_df.to_excel(PASTA_TABELAS / "testes_wald_regimes_petrobras_3_periodos.xlsx", index=False)

    if not resultados_regimes_df.empty:
        resultados_regimes_df[resultados_regimes_df["h"].isin([3, 6, 12, 24])].to_excel(
            PASTA_TABELAS / "tabela_regimes_petrobras_h3_h6_h12_h24.xlsx",
            index=False
        )

    if not resultados_wald_df.empty:
        resultados_wald_df[resultados_wald_df["h"].isin([3, 6, 12, 24])].to_excel(
            PASTA_TABELAS / "tabela_wald_regimes_h3_h6_h12_h24.xlsx",
            index=False
        )

    # --------------------------------------------------------
    # 8.6 LP-IV com Oil Supply News Shock
    # --------------------------------------------------------
    print("Estimando LP-IV, se Oil Supply News Shock estiver disponível...")
    resultados_iv = []

    if "oil_supply_news_shock" in df.columns:
        relacoes_iv = [
            ("dlog_brent_brl", "oil_supply_news_shock", "dlog_diesel"),
            ("dlog_brent_brl", "oil_supply_news_shock", "dlog_gasolina_refinaria"),
            ("dlog_brent_brl", "oil_supply_news_shock", "ipca_transportes"),
            ("dlog_brent_brl", "oil_supply_news_shock", "ipca_geral")
        ]

        for impulso, instrumento, resposta in relacoes_iv:
            tab = local_projection_iv(
                df,
                impulse=impulso,
                instrument=instrumento,
                response=resposta,
                lags=LAGS_PRINCIPAL,
                h_max=H_MAX,
                acumulada=True,
                tag_modelo="lp_iv_oil_supply_news_3_lags"
            )
            resultados_iv.append(tab)

            if SALVAR_GRAFICOS and not tab.empty:
                tab_plot = tab.rename(columns={"coef_iv": "coef"})
                titulo = f"LP-IV: {impulso} instrumentado por {instrumento} → {resposta}"
                arquivo = PASTA_GRAFICOS / f"lp_iv_{nome_seguro(impulso)}_para_{nome_seguro(resposta)}.png"
                plotar_resposta(tab_plot, titulo, arquivo)

    resultados_iv_df = pd.concat(resultados_iv, ignore_index=True) if resultados_iv else pd.DataFrame()
    resultados_iv_df.to_excel(PASTA_TABELAS / "resultados_lp_iv_oil_supply_news.xlsx", index=False)

    if not resultados_iv_df.empty:
        resultados_iv_df[resultados_iv_df["h"].isin([3, 6, 12, 24])].to_excel(
            PASTA_TABELAS / "tabela_lp_iv_h3_h6_h12_h24.xlsx",
            index=False
        )

    # --------------------------------------------------------
    # 8.7 Resumo automático para colar no TCC/código
    # --------------------------------------------------------
    print("Gerando resumo automático...")
    gerar_resumo_txt(resultados_lp_df, resultados_robustez_df, resultados_regimes_df, resultados_wald_df, resultados_iv_df)

    print("\nConcluído.")
    print(f"Arquivos salvos em: {PASTA_SAIDA.resolve()}")


# ============================================================
# 9. RESUMO AUTOMÁTICO
# ============================================================

def formatar_sig(row, coluna_coef="coef"):
    sig = "significativo" if row.get("significativo_90", False) else "não significativo"
    return (
        f"{row.get('impulso', row.get('impulso_instrumentado'))} -> {row['resposta']}: "
        f"coef={row[coluna_coef]:.3f}, "
        f"IC90=[{row['ic_inf']:.3f}; {row['ic_sup']:.3f}], {sig}"
    )


def gerar_resumo_txt(lp, rob, regimes, wald, iv):
    """
    Gera arquivo de texto com bullets enxutos para documentação do modelo.
    """
    linhas = []

    linhas.append("RESUMO DO petroleo_lp_modelo6.py")
    linhas.append("=" * 70)
    linhas.append("")
    linhas.append("O que foi feito:")
    linhas.append("1. O modelo principal usa Local Projections acumuladas com 3 defasagens.")
    linhas.append("2. O choque principal é o Brent em reais, construído como Brent em dólar multiplicado pelo câmbio.")
    linhas.append("3. Os controles finais são atividade econômica, Selic, expectativas de inflação, dummies mensais, Stringency Index, defasagens da variável dependente e defasagens do choque.")
    linhas.append("4. A inferência usa erros-padrão HAC/Newey-West, com bandwidth crescente por horizonte.")
    linhas.append("5. A ação da Petrobras foi dividida em três regimes: 2003-2014, 2015-2022 e 2023 em diante.")
    linhas.append("6. Foram gerados testes de Wald para comparar diferenças entre regimes.")
    linhas.append("7. Foram geradas robustez com 6 e 12 defasagens.")
    linhas.append("8. Se a série Oil Supply News Shock estiver na base, o script estima LP-IV como robustez de identificação.")
    linhas.append("")

    if lp is not None and not lp.empty:
        linhas.append(f"Resultados principais no horizonte de {H_PRINCIPAL} meses:")
        h12 = lp[lp["h"] == H_PRINCIPAL].copy()
        h12 = h12.sort_values("significativo_90", ascending=False)
        for _, row in h12.head(20).iterrows():
            linhas.append("- " + formatar_sig(row, "coef"))
        linhas.append("")

    if rob is not None and not rob.empty:
        linhas.append("Robustez com 6 e 12 defasagens:")
        linhas.append("- Resultados salvos em resultados_lp_robustez_6_12_lags.xlsx.")
        linhas.append("- Use essa tabela para verificar se sinal e significância permanecem próximos ao modelo principal.")
        linhas.append("")

    if regimes is not None and not regimes.empty:
        linhas.append("Regimes Petrobras:")
        linhas.append("- 2003-2014: maior intervenção, suavização ou represamento dos preços.")
        linhas.append("- 2015-2022: realinhamento e maior aderência à paridade internacional.")
        linhas.append("- 2023 em diante: nova estratégia comercial da Petrobras.")
        linhas.append("- O período 2023+ tem menos observações, então deve ser interpretado com cautela.")
        linhas.append("")

    if wald is not None and not wald.empty:
        linhas.append("Testes de Wald:")
        wald_sel = wald[(wald["h"].isin([6, 12])) & (wald["diferenca_significativa_10"] == True)]
        if wald_sel.empty:
            linhas.append("- Não houve diferenças fortes nos horizontes 6 e 12 meses, ao nível de 10%, nas relações selecionadas.")
        else:
            for _, row in wald_sel.head(20).iterrows():
                linhas.append(
                    f"- {row['impulso']} -> {row['resposta']}, h={int(row['h'])}, "
                    f"{row['comparacao']}: p-Wald={row['p_valor_wald']:.3f}"
                )
        linhas.append("")

    if iv is not None and not iv.empty:
        linhas.append("LP-IV com Oil Supply News Shock:")
        hsel = iv[iv["h"].isin([3, 6, 12])].copy()
        for _, row in hsel.head(20).iterrows():
            sig = "significativo" if row["significativo_90"] else "não significativo"
            linhas.append(
                f"- {row['impulso_instrumentado']} IV -> {row['resposta']}, h={int(row['h'])}: "
                f"coef={row['coef_iv']:.3f}, IC90=[{row['ic_inf']:.3f}; {row['ic_sup']:.3f}], "
                f"{sig}, F 1º estágio={row['f_primeiro_estagio']:.2f}"
            )
        linhas.append("")

    linhas.append("Como escrever no TCC:")
    linhas.append("O modelo deve ser apresentado como evidência dinâmica reduzida. A interpretação causal forte fica restrita ao exercício de robustez com Oil Supply News Shock. O resultado central esperado é que choques no petróleo em reais afetam mais claramente combustíveis e IPCA Transportes, enquanto o IPCA Geral tende a apresentar resposta menor, menos persistente ou menos precisa.")
    linhas.append("")

    caminho = PASTA_RESUMOS / "resumo_modelo6_para_tcc.txt"
    caminho.write_text("\n".join(linhas), encoding="utf-8")


# ============================================================
# 10. EXECUTAR
# ============================================================

if __name__ == "__main__":
    executar_modelos()
