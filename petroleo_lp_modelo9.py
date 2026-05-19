# -*- coding: utf-8 -*-
"""
petroleo_lp_modelo9.py

MODELO 9 - LOCAL PROJECTIONS COM CHOQUE EXTERNO DE OFERTA DE PETRÓLEO

Objetivo
--------
Este script adiciona uma camada de identificação externa ao seu TCC/artigo.

Ele usa a série "Oil supply news shock" como choque externo de oferta de petróleo.
A série vem do arquivo:

    oilSupplyNewsShocks_2025M06.xlsx

O Modelo 9 possui duas partes:

1. Modelo 9A - Local Projections diretas:
   Oil supply news shock -> combustíveis
   Oil supply news shock -> IPCA Geral
   Oil supply news shock -> IPCA Transportes

2. Modelo 9B - LP-IV:
   Usa Oil supply news shock como instrumento para dln_petroleo_brl.

   Primeiro estágio:
       dln_petroleo_brl_t = pi * OilSupplyNewsShock_t + controles_t + erro_t

   Segundo estágio:
       y_{t+h} = beta_h * dln_petroleo_brl_instrumentado_t + controles_t + erro_{t+h}

Como interpretar
----------------
- Se o primeiro estágio for forte, o LP-IV melhora a identificação do choque de petróleo.
- Se o primeiro estágio for fraco, use o Modelo 9A como evidência externa e trate o LP-IV com cautela.
- O Modelo 9 deve ser tratado como robustez de identificação, não como substituto do Modelo 3 e do Modelo 4.

Observação
----------
Este código tenta usar linearmodels.iv.IV2SLS, caso esteja instalado.
Se linearmodels não estiver instalado, ele roda um 2SLS manual como aproximação:
primeiro estima o petróleo previsto e depois usa esse petróleo previsto na LP com HAC.
Para artigo, o ideal é instalar linearmodels:

    pip install linearmodels
"""

# =============================================================================
# 0. PACOTES
# =============================================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

try:
    from linearmodels.iv import IV2SLS
    TEM_LINEARMODELS = True
except Exception:
    TEM_LINEARMODELS = False


# =============================================================================
# 1. CONFIGURAÇÕES GERAIS
# =============================================================================

ARQUIVO_IPCA = r"C:\Users\pedro\OneDrive\Documentos\TCC\IPCA.xlsx"
ABA_IPCA = 0

ARQUIVO_CHOQUE_OFERTA = r"C:\Users\pedro\OneDrive\Documentos\TCC\oilSupplyNewsShocks_2025M06.xlsx"
ABA_CHOQUE = "Monthly"

OUTPUT_DIR = Path("output_petroleo_lp_modelo9")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_INICIO = "2003-01-01"
DATA_FIM = "2025-06-01"   # a série de choque vai até 2025M06 no arquivo enviado
DATA_CORTE_POLITICA_PETROBRAS = "2016-09-01"

H_MAX = 12
H_PRINCIPAL = 12

LAGS_Y = 3
LAGS_SHOCK = 3
LAGS_CONTROLS = 3

CONF = 0.90
Z_CRIT = 1.645

USAR_HAC = True
USAR_DUMMIES_MENSAIS = True
PADRONIZAR_CHOQUE_EXTERNO = True
PADRONIZAR_PETROLEO_ENDOGENO = True

MIN_OBS = 60

# =============================================================================
# 1.1 LOCALIZAÇÃO AUTOMÁTICA DOS ARQUIVOS
# =============================================================================

def localizar_arquivo(nome_arquivo, caminho_configurado=None):
    """
    Procura um arquivo em locais prováveis:
    1. caminho configurado no código;
    2. pasta atual;
    3. pasta onde o script está salvo;
    4. pastas TCC e TCC_python dentro de Documentos.
    """
    candidatos = []

    if caminho_configurado:
        candidatos.append(Path(caminho_configurado))

    candidatos.append(Path(nome_arquivo))
    candidatos.append(Path.cwd() / nome_arquivo)

    try:
        candidatos.append(Path(__file__).resolve().parent / nome_arquivo)
    except Exception:
        pass

    home = Path.home()
    candidatos.append(home / "OneDrive" / "Documentos" / "TCC" / nome_arquivo)
    candidatos.append(home / "OneDrive" / "Documentos" / "TCC_python" / nome_arquivo)
    candidatos.append(home / "Documents" / "TCC" / nome_arquivo)
    candidatos.append(home / "Documents" / "TCC_python" / nome_arquivo)

    # Evita repetição.
    vistos = set()
    candidatos_unicos = []
    for c in candidatos:
        try:
            key = str(c.resolve())
        except Exception:
            key = str(c)
        if key not in vistos:
            vistos.add(key)
            candidatos_unicos.append(c)

    for c in candidatos_unicos:
        if c.exists():
            return str(c)

    msg = [
        f"Não encontrei o arquivo: {nome_arquivo}",
        "",
        "Locais testados:"
    ]
    msg += [f"- {c}" for c in candidatos_unicos]
    msg += [
        "",
        "Como resolver:",
        "1. coloque o arquivo na mesma pasta do script; ou",
        "2. ajuste o caminho nas variáveis ARQUIVO_IPCA e ARQUIVO_CHOQUE_OFERTA."
    ]
    raise FileNotFoundError("\n".join(msg))



# =============================================================================
# 2. NOMES DAS COLUNAS
# =============================================================================

CONFIG_COLUNAS = {
    "data": ["Data", "data", "DATE", "Date"],

    "petroleo": ["Preco_Barril", "Petroleo", "Petróleo", "Brent", "DCOILBRENTEU", "preco_barril"],
    "cambio": ["Cambio", "cambio", "USDBRL", "Dolar", "Dólar", "Taxa_Cambio"],

    "gasolina_refinaria": [
        "GasolinaABrasil_media", "GasolinaA", "GasolinaA_nivel",
        "Gasolina_A", "Gasolina_Refinaria", "Preco_Refinaria"
    ],
    "gasolina": ["Gasolina", "Gasolina_nivel", "Gasolina_consumidor", "Preco_Gasolina"],
    "etanol": ["Etanol", "Etanol_nivel", "Preco_Etanol"],
    "diesel": ["Oleo_diesel", "Oleo_diesel_nivel", "Diesel", "Preco_Diesel"],

    "ipca_geral": ["IPCA_Geral_nivel", "IPCA_Brasil", "Var_IPCA_Brasil", "IPCA_Geral", "IPCA"],
    "ipca_transporte": ["IPCA_Trans_nivel", "Var_IPCA_trans", "IPCA_Transporte", "IPCA_Trans"],

    "atividade": ["Atividade", "IBC_BR", "IBC_Br", "IBC-BR", "IBC"],
    "selic": ["Selic", "SELIC", "Meta_Selic", "selic"],
    "expectativa": ["Expectativa_Inflacao", "Focus_IPCA_12m", "IPCA_Focus_12m", "Expectativa"],
    "stringency": ["Stringency", "stringency", "Stringency_Index", "Oxford_Stringency"]
}


# =============================================================================
# 3. FUNÇÕES AUXILIARES
# =============================================================================

def encontrar_coluna(df, candidatos, obrigatoria=False, nome_logico=""):
    cols = list(df.columns)

    for c in candidatos:
        if c in cols:
            return c

    cols_lower = {str(c).lower(): c for c in cols}
    for c in candidatos:
        if str(c).lower() in cols_lower:
            return cols_lower[str(c).lower()]

    normalizadas = {
        str(c).lower().replace(" ", "").replace("_", "").replace("-", ""): c
        for c in cols
    }
    for c in candidatos:
        chave = str(c).lower().replace(" ", "").replace("_", "").replace("-", "")
        if chave in normalizadas:
            return normalizadas[chave]

    if obrigatoria:
        raise ValueError(
            f"Não encontrei a coluna obrigatória '{nome_logico}'. "
            f"Candidatos testados: {candidatos}. "
            f"Colunas disponíveis: {cols}"
        )

    return None


def safe_log_diff(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s.where(s > 0)).diff()


def diff_se_precisa(s):
    """
    Para IPCA:
    - Se parecer índice em nível/base 100, usa Δlog * 100.
    - Se parecer inflação mensal já pronta, mantém a série.
    """
    s = pd.to_numeric(s, errors="coerce")
    med = s.dropna().median()

    if pd.notna(med) and med > 20:
        return 100 * safe_log_diff(s)

    return s


def padronizar(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s
    return s / sd


def criar_lags(df, var, n_lags, prefix=None):
    if prefix is None:
        prefix = var

    lag_cols = []
    for L in range(1, n_lags + 1):
        col = f"{prefix}_lag{L}"
        df[col] = df[var].shift(L)
        lag_cols.append(col)

    return lag_cols


def parse_data_mensal_choque(x):
    """
    Converte datas no formato 1975M01 para timestamp mensal.
    Também aceita datas já interpretáveis pelo pandas.
    """
    if pd.isna(x):
        return pd.NaT

    if isinstance(x, pd.Timestamp):
        return x.to_period("M").to_timestamp()

    sx = str(x).strip()

    # Formato tipo 1975M01.
    if "M" in sx and len(sx) >= 7:
        try:
            ano, mes = sx.split("M")
            return pd.Timestamp(int(ano), int(mes), 1)
        except Exception:
            pass

    return pd.to_datetime(sx, errors="coerce")


def carregar_choque_oferta():
    """
    Lê o arquivo de Oil Supply News Shock.
    Usa a aba Monthly.
    """
    print("=" * 100)
    print("1) LEITURA DA SÉRIE EXTERNA DE CHOQUE DE OFERTA")
    print("=" * 100)

    arquivo_choque_usado = localizar_arquivo("oilSupplyNewsShocks_2025M06.xlsx", ARQUIVO_CHOQUE_OFERTA)
    print(f"Arquivo de choque usado: {arquivo_choque_usado}")
    choque = pd.read_excel(arquivo_choque_usado, sheet_name=ABA_CHOQUE)
    choque.columns = [str(c).strip() for c in choque.columns]

    col_data = encontrar_coluna(choque, ["Date", "Data", "date"], obrigatoria=True, nome_logico="data_choque")
    col_news = encontrar_coluna(
        choque,
        ["Oil supply news shock", "oil supply news shock", "Oil Supply News Shock"],
        obrigatoria=True,
        nome_logico="oil_supply_news_shock"
    )
    col_surprise = encontrar_coluna(
        choque,
        ["Oil supply surprise series", "oil supply surprise series", "Oil Supply Surprise Series"],
        obrigatoria=False,
        nome_logico="oil_supply_surprise"
    )

    choque["Data"] = choque[col_data].apply(parse_data_mensal_choque)
    choque["oil_supply_news_shock"] = pd.to_numeric(choque[col_news], errors="coerce")

    if col_surprise is not None:
        choque["oil_supply_surprise_series"] = pd.to_numeric(choque[col_surprise], errors="coerce")

    choque = choque.dropna(subset=["Data"]).sort_values("Data")
    choque = choque[["Data", "oil_supply_news_shock"] + (["oil_supply_surprise_series"] if col_surprise else [])]

    if PADRONIZAR_CHOQUE_EXTERNO:
        choque["oil_supply_news_shock_std"] = padronizar(choque["oil_supply_news_shock"])
    else:
        choque["oil_supply_news_shock_std"] = choque["oil_supply_news_shock"]

    print("Arquivo de choque carregado.")
    print(f"Período: {choque['Data'].min().date()} até {choque['Data'].max().date()}")
    print(f"Observações: {len(choque)}")
    print(f"Coluna usada como instrumento/choque: {col_news}")

    return choque


def carregar_preparar_base_ipca():
    """
    Lê IPCA.xlsx e cria as variáveis transformadas.
    """
    print("\n" + "=" * 100)
    print("2) LEITURA E PREPARAÇÃO DA BASE PRINCIPAL")
    print("=" * 100)

    arquivo_ipca_usado = localizar_arquivo("IPCA.xlsx", ARQUIVO_IPCA)
    print(f"Arquivo IPCA usado: {arquivo_ipca_usado}")
    df = pd.read_excel(arquivo_ipca_usado, sheet_name=ABA_IPCA)
    df.columns = [str(c).strip() for c in df.columns]

    col_data = encontrar_coluna(df, CONFIG_COLUNAS["data"], obrigatoria=True, nome_logico="data")
    df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
    df = df.dropna(subset=[col_data]).sort_values(col_data).reset_index(drop=True)
    df = df.rename(columns={col_data: "Data"})

    df["Data"] = df["Data"].dt.to_period("M").dt.to_timestamp()
    df["mes"] = df["Data"].dt.month
    df["ano"] = df["Data"].dt.year

    # Corte da política de preços da Petrobras: setembro de 2016.
    # A dummy vale 0 antes do corte e 1 a partir de 2016-09.
    df["regime_petrobras_pos_set2016"] = (
        df["Data"] >= pd.to_datetime(DATA_CORTE_POLITICA_PETROBRAS)
    ).astype(float)

    mapa = {}
    for nome_logico, candidatos in CONFIG_COLUNAS.items():
        if nome_logico == "data":
            continue
        mapa[nome_logico] = encontrar_coluna(df, candidatos, obrigatoria=False, nome_logico=nome_logico)

    print("\nColunas identificadas na base principal:")
    for k, v in mapa.items():
        print(f"- {k}: {v}")

    vars_t = {}

    if mapa["petroleo"]:
        df["dln_petroleo_usd"] = 100 * safe_log_diff(df[mapa["petroleo"]])
        vars_t["petroleo_usd"] = "dln_petroleo_usd"

    if mapa["cambio"]:
        df["dln_cambio"] = 100 * safe_log_diff(df[mapa["cambio"]])
        vars_t["cambio"] = "dln_cambio"

    if mapa["petroleo"] and mapa["cambio"]:
        petroleo = pd.to_numeric(df[mapa["petroleo"]], errors="coerce")
        cambio = pd.to_numeric(df[mapa["cambio"]], errors="coerce")
        df["petroleo_brl_nivel"] = petroleo * cambio
        df["dln_petroleo_brl"] = 100 * safe_log_diff(df["petroleo_brl_nivel"])

        if PADRONIZAR_PETROLEO_ENDOGENO:
            df["dln_petroleo_brl_std"] = padronizar(df["dln_petroleo_brl"])
            vars_t["petroleo_brl"] = "dln_petroleo_brl_std"
        else:
            vars_t["petroleo_brl"] = "dln_petroleo_brl"

    if mapa["gasolina_refinaria"]:
        df["dln_gasolina_refinaria"] = 100 * safe_log_diff(df[mapa["gasolina_refinaria"]])
        vars_t["gasolina_refinaria"] = "dln_gasolina_refinaria"

    if mapa["gasolina"]:
        df["dln_gasolina"] = 100 * safe_log_diff(df[mapa["gasolina"]])
        vars_t["gasolina"] = "dln_gasolina"

    if mapa["etanol"]:
        df["dln_etanol"] = 100 * safe_log_diff(df[mapa["etanol"]])
        vars_t["etanol"] = "dln_etanol"

    if mapa["diesel"]:
        df["dln_diesel"] = 100 * safe_log_diff(df[mapa["diesel"]])
        vars_t["diesel"] = "dln_diesel"

    if mapa["atividade"]:
        df["dln_atividade"] = 100 * safe_log_diff(df[mapa["atividade"]])
        vars_t["atividade"] = "dln_atividade"

    if mapa["ipca_geral"]:
        df["ipca_geral_mensal"] = diff_se_precisa(df[mapa["ipca_geral"]])
        vars_t["ipca_geral"] = "ipca_geral_mensal"

    if mapa["ipca_transporte"]:
        df["ipca_transporte_mensal"] = diff_se_precisa(df[mapa["ipca_transporte"]])
        vars_t["ipca_transporte"] = "ipca_transporte_mensal"

    if mapa["selic"]:
        df["selic_controle"] = pd.to_numeric(df[mapa["selic"]], errors="coerce")
        vars_t["selic"] = "selic_controle"

    if mapa["expectativa"]:
        df["expectativa_controle"] = pd.to_numeric(df[mapa["expectativa"]], errors="coerce")
        vars_t["expectativa"] = "expectativa_controle"

    if mapa["stringency"]:
        df["stringency_controle"] = pd.to_numeric(df[mapa["stringency"]], errors="coerce")
        vars_t["stringency"] = "stringency_controle"

    vars_t["regime_petrobras"] = "regime_petrobras_pos_set2016"

    dummy_cols = []
    if USAR_DUMMIES_MENSAIS:
        dummies = pd.get_dummies(df["mes"], prefix="mes", drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        dummy_cols = list(dummies.columns)

    print("\nVariáveis transformadas criadas:")
    for k, v in vars_t.items():
        print(f"- {k}: {v}")

    return df, vars_t, dummy_cols


def juntar_bases():
    choque = carregar_choque_oferta()
    df, vars_t, dummy_cols = carregar_preparar_base_ipca()

    base = pd.merge(df, choque, on="Data", how="left")
    base = base[(base["Data"] >= pd.to_datetime(DATA_INICIO)) & (base["Data"] <= pd.to_datetime(DATA_FIM))].copy()
    base = base.sort_values("Data").reset_index(drop=True)

    vars_t["oil_supply_news_shock"] = "oil_supply_news_shock_std"

    base.to_excel(OUTPUT_DIR / "base_modelo9_com_choque_oferta.xlsx", index=False)

    print("\n" + "=" * 100)
    print("3) BASE FINAL")
    print("=" * 100)
    print(f"Período final: {base['Data'].min().date()} até {base['Data'].max().date()}")
    print(f"Observações finais: {len(base)}")
    print(f"Observações com Oil supply news shock: {base['oil_supply_news_shock'].notna().sum()}")
    print(f"linearmodels disponível: {TEM_LINEARMODELS}")

    return base, vars_t, dummy_cols


def controles_macro(vars_t, dummy_cols):
    controles = []
    for k in ["cambio", "atividade", "selic", "expectativa", "stringency", "regime_petrobras"]:
        if k in vars_t:
            controles.append(vars_t[k])
    controles = controles + dummy_cols
    return controles


def montar_y_h(temp, y, h, acumulada):
    if acumulada:
        cols_futuras = []
        for j in range(0, h + 1):
            col_fut = f"{y}_lead{j}"
            temp[col_fut] = temp[y].shift(-j)
            cols_futuras.append(col_fut)
        temp[f"y_h{h}"] = temp[cols_futuras].sum(axis=1, min_count=h + 1)
    else:
        temp[f"y_h{h}"] = temp[y].shift(-h)

    return temp


def ajustar_ols_hac(Y, X, h):
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(Y, X)

    if USAR_HAC:
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": max(1, h)})

    return model.fit(cov_type="HC1")


# =============================================================================
# 4. MODELO 9A - LP DIRETA COM CHOQUE EXTERNO
# =============================================================================

def local_projection_direta(
    df,
    y,
    shock,
    controls=None,
    h_max=24,
    acumulada=True,
    nome_modelo="modelo9A",
    subpasta="A_lp_direta_oil_supply_news"
):
    controls = controls or []
    base = df.copy()

    regressores_fixos = []

    regressores_fixos += criar_lags(base, y, LAGS_Y, prefix=y)
    regressores_fixos += criar_lags(base, shock, LAGS_SHOCK, prefix=shock)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            regressores_fixos += criar_lags(base, c, LAGS_CONTROLS, prefix=c)

    resultados = []

    for h in range(0, h_max + 1):
        temp = base.copy()
        temp = montar_y_h(temp, y, h, acumulada)

        X_cols = [shock] + regressores_fixos
        X_cols = [c for c in X_cols if c in temp.columns]

        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(MIN_OBS, len(X_cols) + 10):
            resultados.append({
                "h": h, "coef": np.nan, "se": np.nan, "t": np.nan,
                "pvalor": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "nobs": len(temp_reg)
            })
            continue

        Y = temp_reg[f"y_h{h}"]
        X = temp_reg[X_cols]

        try:
            res = ajustar_ols_hac(Y, X, h)
            coef = res.params.get(shock, np.nan)
            se = res.bse.get(shock, np.nan)
            t = res.tvalues.get(shock, np.nan)
            pvalor = res.pvalues.get(shock, np.nan)

            resultados.append({
                "h": h,
                "coef": coef,
                "se": se,
                "t": t,
                "pvalor": pvalor,
                "ci_low": coef - Z_CRIT * se,
                "ci_high": coef + Z_CRIT * se,
                "nobs": int(res.nobs)
            })

        except Exception as e:
            print(f"Erro LP direta em {nome_modelo}, h={h}: {e}")
            resultados.append({
                "h": h, "coef": np.nan, "se": np.nan, "t": np.nan,
                "pvalor": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "nobs": len(temp_reg)
            })

    tab = pd.DataFrame(resultados)

    pasta = OUTPUT_DIR / subpasta
    pasta.mkdir(parents=True, exist_ok=True)

    sufixo = "acumulada" if acumulada else "pontual"
    tab.to_csv(pasta / f"lp_direta_{sufixo}_{nome_modelo}.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tab["h"], tab["coef"], marker="o", label="Resposta estimada")
    ax.fill_between(tab["h"], tab["ci_low"], tab["ci_high"], alpha=0.2, label=f"IC {int(CONF * 100)}%")
    ax.axhline(0, linewidth=1)
    ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"Modelo 9A - LP direta - {nome_modelo}")
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("Resposta acumulada")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(pasta / f"lp_direta_{sufixo}_{nome_modelo}.png", dpi=300)
    plt.close()

    return tab


# =============================================================================
# 5. MODELO 9B - LP-IV
# =============================================================================

def primeiro_estagio(df_reg, endog, instrument, exog_cols):
    """
    Estima primeiro estágio para diagnóstico.
    endog = pi * instrument + controles.
    Retorna resultado OLS e F-stat do instrumento.
    """
    Y = df_reg[endog]
    X = sm.add_constant(df_reg[[instrument] + exog_cols], has_constant="add")
    res = sm.OLS(Y, X).fit(cov_type="HC1")

    try:
        ftest = res.f_test(f"{instrument} = 0")
        f_stat = float(ftest.fvalue)
        f_pvalue = float(ftest.pvalue)
    except Exception:
        f_stat = np.nan
        f_pvalue = np.nan

    return res, f_stat, f_pvalue


def lp_iv_por_horizonte(
    df,
    y,
    endog,
    instrument,
    controls=None,
    h_max=24,
    acumulada=True,
    nome_modelo="modelo9B",
    subpasta="B_lpiv_oil_supply_news"
):
    """
    LP-IV:
    y_h = beta * endog + controles, usando instrument para endog.

    endog: dln_petroleo_brl_std
    instrument: oil_supply_news_shock_std
    """

    controls = controls or []
    base = df.copy()

    regressores_fixos = []

    regressores_fixos += criar_lags(base, y, LAGS_Y, prefix=y)
    regressores_fixos += criar_lags(base, endog, LAGS_SHOCK, prefix=endog)
    regressores_fixos += criar_lags(base, instrument, LAGS_SHOCK, prefix=instrument)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            regressores_fixos += criar_lags(base, c, LAGS_CONTROLS, prefix=c)

    resultados = []
    primeiros_estagios = []

    for h in range(0, h_max + 1):
        temp = base.copy()
        temp = montar_y_h(temp, y, h, acumulada)

        exog_cols = [c for c in regressores_fixos if c in temp.columns]

        cols_necessarias = [f"y_h{h}", endog, instrument] + exog_cols
        temp_reg = temp[cols_necessarias].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(MIN_OBS, len(cols_necessarias) + 10):
            resultados.append({
                "h": h,
                "coef_iv": np.nan,
                "se_iv": np.nan,
                "t_iv": np.nan,
                "pvalor_iv": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "nobs": len(temp_reg),
                "first_stage_f": np.nan,
                "first_stage_pvalue": np.nan,
                "metodo": "insuficiente"
            })
            continue

        # Primeiro estágio, sempre calculado para diagnóstico.
        try:
            fs_res, fs_f, fs_p = primeiro_estagio(temp_reg, endog, instrument, exog_cols)
        except Exception as e:
            print(f"Erro no primeiro estágio de {nome_modelo}, h={h}: {e}")
            fs_res, fs_f, fs_p = None, np.nan, np.nan

        primeiros_estagios.append({
            "h": h,
            "nobs": len(temp_reg),
            "first_stage_f": fs_f,
            "first_stage_pvalue": fs_p,
            "coef_instrumento": fs_res.params.get(instrument, np.nan) if fs_res is not None else np.nan,
            "pvalor_instrumento": fs_res.pvalues.get(instrument, np.nan) if fs_res is not None else np.nan,
            "r2_first_stage": fs_res.rsquared if fs_res is not None else np.nan
        })

        Y = temp_reg[f"y_h{h}"]

        # Preferência: linearmodels.iv.IV2SLS.
        if TEM_LINEARMODELS:
            try:
                exog = sm.add_constant(temp_reg[exog_cols], has_constant="add") if exog_cols else pd.DataFrame({"const": 1.0}, index=temp_reg.index)
                endog_df = temp_reg[[endog]]
                instr_df = temp_reg[[instrument]]

                modelo = IV2SLS(Y, exog, endog_df, instr_df)
                res_iv = modelo.fit(cov_type="kernel", kernel="bartlett", bandwidth=max(1, h))

                coef = res_iv.params.get(endog, np.nan)
                se = res_iv.std_errors.get(endog, np.nan)
                t = res_iv.tstats.get(endog, np.nan)
                pvalor = res_iv.pvalues.get(endog, np.nan)

                resultados.append({
                    "h": h,
                    "coef_iv": coef,
                    "se_iv": se,
                    "t_iv": t,
                    "pvalor_iv": pvalor,
                    "ci_low": coef - Z_CRIT * se,
                    "ci_high": coef + Z_CRIT * se,
                    "nobs": int(res_iv.nobs),
                    "first_stage_f": fs_f,
                    "first_stage_pvalue": fs_p,
                    "metodo": "IV2SLS_linearmodels"
                })
                continue

            except Exception as e:
                print(f"Erro IV2SLS linearmodels em {nome_modelo}, h={h}: {e}")
                # Cai para 2SLS manual.

        # Fallback: 2SLS manual.
        try:
            if fs_res is None:
                raise ValueError("Primeiro estágio indisponível.")

            X_fs = sm.add_constant(temp_reg[[instrument] + exog_cols], has_constant="add")
            temp_reg = temp_reg.copy()
            temp_reg[f"{endog}_hat"] = fs_res.predict(X_fs)

            X_second_cols = [f"{endog}_hat"] + exog_cols
            X2 = temp_reg[X_second_cols]
            res2 = ajustar_ols_hac(Y, X2, h)

            coef = res2.params.get(f"{endog}_hat", np.nan)
            se = res2.bse.get(f"{endog}_hat", np.nan)
            t = res2.tvalues.get(f"{endog}_hat", np.nan)
            pvalor = res2.pvalues.get(f"{endog}_hat", np.nan)

            resultados.append({
                "h": h,
                "coef_iv": coef,
                "se_iv": se,
                "t_iv": t,
                "pvalor_iv": pvalor,
                "ci_low": coef - Z_CRIT * se,
                "ci_high": coef + Z_CRIT * se,
                "nobs": int(res2.nobs),
                "first_stage_f": fs_f,
                "first_stage_pvalue": fs_p,
                "metodo": "2SLS_manual_aproximado"
            })

        except Exception as e:
            print(f"Erro fallback 2SLS em {nome_modelo}, h={h}: {e}")
            resultados.append({
                "h": h,
                "coef_iv": np.nan,
                "se_iv": np.nan,
                "t_iv": np.nan,
                "pvalor_iv": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "nobs": len(temp_reg),
                "first_stage_f": fs_f,
                "first_stage_pvalue": fs_p,
                "metodo": "erro"
            })

    tab = pd.DataFrame(resultados)
    fs_tab = pd.DataFrame(primeiros_estagios)

    pasta = OUTPUT_DIR / subpasta
    pasta.mkdir(parents=True, exist_ok=True)

    sufixo = "acumulada" if acumulada else "pontual"
    tab.to_csv(pasta / f"lpiv_{sufixo}_{nome_modelo}.csv", index=False, encoding="utf-8-sig")
    fs_tab.to_csv(pasta / f"primeiro_estagio_{nome_modelo}.csv", index=False, encoding="utf-8-sig")

    # Gráfico IV.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tab["h"], tab["coef_iv"], marker="o", label="Resposta IV estimada")
    ax.fill_between(tab["h"], tab["ci_low"], tab["ci_high"], alpha=0.2, label=f"IC {int(CONF * 100)}%")
    ax.axhline(0, linewidth=1)
    ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"Modelo 9B - LP-IV - {nome_modelo}")
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("Resposta acumulada")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(pasta / f"lpiv_{sufixo}_{nome_modelo}.png", dpi=300)
    plt.close()

    # Gráfico F-stat primeiro estágio.
    if not fs_tab.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(fs_tab["h"], fs_tab["first_stage_f"], marker="o")
        ax.axhline(10, linestyle="--", linewidth=1, label="Regra prática F = 10")
        ax.set_title(f"Primeiro estágio - {nome_modelo}")
        ax.set_xlabel("Horizonte h, em meses")
        ax.set_ylabel("F-stat do instrumento")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(pasta / f"primeiro_estagio_fstat_{nome_modelo}.png", dpi=300)
        plt.close()

    return tab, fs_tab


# =============================================================================
# 6. RODAR MODELOS
# =============================================================================

def estimar_modelo9A(df, vars_t, dummy_cols):
    print("\n" + "=" * 100)
    print("4) MODELO 9A - LP DIRETA COM OIL SUPPLY NEWS SHOCK")
    print("=" * 100)

    controles = controles_macro(vars_t, dummy_cols)
    shock = vars_t["oil_supply_news_shock"]

    alvos = []
    for k in ["gasolina_refinaria", "gasolina", "diesel", "etanol", "ipca_geral", "ipca_transporte"]:
        if k in vars_t:
            alvos.append(k)

    for alvo in alvos:
        y = vars_t[alvo]
        nome = f"oil_supply_news_para_{alvo}"
        print(f"Estimando Modelo 9A: {nome}")

        local_projection_direta(
            df=df,
            y=y,
            shock=shock,
            controls=controles,
            h_max=H_MAX,
            acumulada=True,
            nome_modelo=nome,
            subpasta="A_lp_direta_oil_supply_news"
        )


def estimar_modelo9B(df, vars_t, dummy_cols):
    print("\n" + "=" * 100)
    print("5) MODELO 9B - LP-IV COM OIL SUPPLY NEWS SHOCK COMO INSTRUMENTO")
    print("=" * 100)

    if "petroleo_brl" not in vars_t:
        print("Não há dln_petroleo_brl na base. Modelo 9B não será estimado.")
        return

    controles = controles_macro(vars_t, dummy_cols)

    endog = vars_t["petroleo_brl"]
    instrument = vars_t["oil_supply_news_shock"]

    alvos = []
    for k in ["gasolina_refinaria", "gasolina", "diesel", "etanol", "ipca_geral", "ipca_transporte"]:
        if k in vars_t:
            alvos.append(k)

    for alvo in alvos:
        y = vars_t[alvo]
        nome = f"petroleo_brl_iv_oil_supply_news_para_{alvo}"
        print(f"Estimando Modelo 9B: {nome}")

        lp_iv_por_horizonte(
            df=df,
            y=y,
            endog=endog,
            instrument=instrument,
            controls=controles,
            h_max=H_MAX,
            acumulada=True,
            nome_modelo=nome,
            subpasta="B_lpiv_oil_supply_news"
        )


def criar_resumos_modelo9():
    print("\n" + "=" * 100)
    print("6) CRIANDO RESUMOS DO MODELO 9")
    print("=" * 100)

    arquivos = list(OUTPUT_DIR.rglob("*.csv"))
    resumo = []
    resumo_fs = []

    for arq in arquivos:
        try:
            tab = pd.read_csv(arq)
        except Exception:
            continue

        nome_arq = str(arq)

        if "primeiro_estagio" in arq.name:
            for h_ref in [0, 3, 6, 12]:
                if "h" in tab.columns and h_ref in tab["h"].values:
                    row = tab.loc[tab["h"] == h_ref].iloc[0].to_dict()
                    row["arquivo"] = nome_arq
                    row["h_ref"] = h_ref
                    row["instrumento_forte_regra_10"] = (
                        pd.notna(row.get("first_stage_f")) and row.get("first_stage_f") >= 10
                    )
                    resumo_fs.append(row)
            continue

        for h_ref in [0, 3, 6, 12]:
            if "h" not in tab.columns or h_ref not in tab["h"].values:
                continue

            row = tab.loc[tab["h"] == h_ref].iloc[0].to_dict()
            row["arquivo"] = nome_arq
            row["h_ref"] = h_ref

            if "coef" in row:
                row["tipo"] = "LP_direta"
                row["significativo_10pct"] = (
                    pd.notna(row.get("ci_low")) and pd.notna(row.get("ci_high")) and
                    ((row.get("ci_low") > 0) or (row.get("ci_high") < 0))
                )
            elif "coef_iv" in row:
                row["tipo"] = "LP_IV"
                row["significativo_10pct"] = (
                    pd.notna(row.get("ci_low")) and pd.notna(row.get("ci_high")) and
                    ((row.get("ci_low") > 0) or (row.get("ci_high") < 0))
                )

            resumo.append(row)

    if resumo:
        resumo_df = pd.DataFrame(resumo)
        resumo_df.to_csv(OUTPUT_DIR / "resumo_modelo9_resultados_horizontes.csv", index=False, encoding="utf-8-sig")
        resumo_df.to_excel(OUTPUT_DIR / "resumo_modelo9_resultados_horizontes.xlsx", index=False)

    if resumo_fs:
        fs_df = pd.DataFrame(resumo_fs)
        fs_df.to_csv(OUTPUT_DIR / "resumo_modelo9_primeiro_estagio.csv", index=False, encoding="utf-8-sig")
        fs_df.to_excel(OUTPUT_DIR / "resumo_modelo9_primeiro_estagio.xlsx", index=False)

    texto = []
    texto.append("RELATÓRIO DO MODELO 9\n")
    texto.append("=====================\n\n")
    texto.append("O Modelo 9 usa a série Oil supply news shock como fonte externa de identificação.\n\n")
    texto.append("Modelo 9A:\n")
    texto.append("- Estima Local Projections diretas usando o choque externo de oferta de petróleo.\n")
    texto.append("- Interpretação: resposta dos combustíveis e da inflação a uma surpresa externa de oferta.\n\n")
    texto.append("Modelo 9B:\n")
    texto.append("- Usa Oil supply news shock como instrumento para dln_petroleo_brl.\n")
    texto.append("- Primeiro estágio: dln_petroleo_brl em função do instrumento e controles.\n")
    texto.append("- Segundo estágio: respostas acumuladas usando petróleo em reais instrumentado.\n\n")
    texto.append("Como avaliar o Modelo 9B:\n")
    texto.append("- Primeiro olhe o arquivo resumo_modelo9_primeiro_estagio.xlsx.\n")
    texto.append("- Se o F-stat do primeiro estágio for maior que 10, o instrumento é mais defensável.\n")
    texto.append("- Se o F-stat for menor que 10, trate o LP-IV com cautela e use o Modelo 9A como evidência externa.\n\n")
    texto.append("Corte Petrobras:\n")
    texto.append(f"- O regime Petrobras foi definido com corte em {DATA_CORTE_POLITICA_PETROBRAS}, isto é, setembro de 2016.\n")
    texto.append("- A dummy regime_petrobras_pos_set2016 vale 0 antes do corte e 1 a partir de 2016-09.\n\n")
    texto.append("Observação:\n")
    texto.append(f"- linearmodels disponível nesta execução: {TEM_LINEARMODELS}\n")
    texto.append("- Se não estiver disponível, o script usa 2SLS manual aproximado.\n")

    (OUTPUT_DIR / "RELATORIO_MODELO9.txt").write_text("".join(texto), encoding="utf-8")


# =============================================================================
# 7. EXECUÇÃO
# =============================================================================

def main():
    df, vars_t, dummy_cols = juntar_bases()

    if "oil_supply_news_shock" not in vars_t:
        raise ValueError("A série Oil supply news shock não foi carregada corretamente.")

    estimar_modelo9A(df, vars_t, dummy_cols)
    estimar_modelo9B(df, vars_t, dummy_cols)
    criar_resumos_modelo9()

    print("\n" + "=" * 100)
    print("MODELO 9 FINALIZADO")
    print("=" * 100)
    print(f"Arquivos salvos em: {OUTPUT_DIR.resolve()}")

    print("\nO que olhar primeiro:")
    print("1. resumo_modelo9_primeiro_estagio.xlsx")
    print("2. resumo_modelo9_resultados_horizontes.xlsx")
    print("3. gráficos da pasta A_lp_direta_oil_supply_news")
    print("4. gráficos da pasta B_lpiv_oil_supply_news")


if __name__ == "__main__":
    main()
