# -*- coding: utf-8 -*-
"""
petroleo_lp_modelo4_regimes_brl.py

MODELO 4 - EXTENSÃO DO MODELO PRINCIPAL DE LOCAL PROJECTIONS

Objetivo do Modelo 4
--------------------
Este script complementa o seu modelo principal de Local Projections.

Ele adiciona duas robustezes importantes para deixar o artigo/TCC mais forte:

1. Brent em reais:
   - Cria a variável Petróleo_BRL = Preço do barril em dólar * câmbio.
   - Estima Local Projections usando tanto o petróleo em dólar quanto o petróleo em reais.
   - Isso permite testar se o choque relevante para a inflação brasileira é mais bem capturado
     pelo preço internacional convertido para moeda doméstica.

2. Teste formal de regimes Petrobras:
   - Cria uma dummy de regime pós-2015.
   - Estima o efeito separado do choque no regime pré-2015 e no regime pós-2015.
   - Aplica teste de Wald para avaliar se o repasse pré-2015 é estatisticamente diferente
     do repasse pós-2015.

Este Modelo 4 não substitui:
- Modelo 3: Local Projections principal por blocos.
- Modelo 10: VAR de robustez com IRF, FEVD, IRF acumulada e Cholesky.

Ele serve como camada adicional de robustez econométrica.
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


# =============================================================================
# 1. CONFIGURAÇÕES GERAIS
# =============================================================================

ARQUIVO = r"C:\Users\pedro\OneDrive\Documentos\TCC\IPCA.xlsx"
ABA = 0

OUTPUT_DIR = Path("output_petroleo_lp_modelo4_regimes_brl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H_MAX = 24
H_PRINCIPAL = 12

LAGS_Y = 3
LAGS_SHOCK = 3
LAGS_CONTROLS = 3

CONF = 0.90
Z_CRIT = 1.645

USAR_HAC = True
PADRONIZAR_CHOQUE = True
USAR_DUMMIES_MENSAIS = True

DATA_CORTE_REGIME = "2015-01-01"

# Estimativas só são feitas quando houver observações suficientes.
MIN_OBS = 60


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

    # Busca ignorando alguns separadores.
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
    out = np.log(s.where(s > 0)).diff()
    return out


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


def criar_lags(df, var, n_lags, prefix=None):
    if prefix is None:
        prefix = var

    lag_cols = []
    for L in range(1, n_lags + 1):
        col = f"{prefix}_lag{L}"
        df[col] = df[var].shift(L)
        lag_cols.append(col)

    return lag_cols


def preparar_base():
    print("=" * 100)
    print("1) LEITURA E PREPARAÇÃO DA BASE")
    print("=" * 100)

    df = pd.read_excel(ARQUIVO, sheet_name=ABA)
    df.columns = [str(c).strip() for c in df.columns]

    col_data = encontrar_coluna(df, CONFIG_COLUNAS["data"], obrigatoria=True, nome_logico="data")
    df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
    df = df.dropna(subset=[col_data]).sort_values(col_data).reset_index(drop=True)
    df = df.rename(columns={col_data: "Data"})

    df["mes"] = df["Data"].dt.month
    df["ano"] = df["Data"].dt.year
    df["D_pos2015"] = (df["Data"] >= pd.to_datetime(DATA_CORTE_REGIME)).astype(int)

    mapa = {}
    for nome_logico, candidatos in CONFIG_COLUNAS.items():
        if nome_logico == "data":
            continue
        mapa[nome_logico] = encontrar_coluna(df, candidatos, obrigatoria=False, nome_logico=nome_logico)

    print("\nColunas identificadas:")
    for k, v in mapa.items():
        print(f"- {k}: {v}")

    transformadas = {}

    # Petróleo em dólar.
    if mapa["petroleo"]:
        df["dln_petroleo_usd"] = 100 * safe_log_diff(df[mapa["petroleo"]])
        transformadas["petroleo_usd"] = "dln_petroleo_usd"

    # Câmbio.
    if mapa["cambio"]:
        df["dln_cambio"] = 100 * safe_log_diff(df[mapa["cambio"]])
        transformadas["cambio"] = "dln_cambio"

    # Petróleo em reais = barril em dólar * câmbio.
    if mapa["petroleo"] and mapa["cambio"]:
        petroleo = pd.to_numeric(df[mapa["petroleo"]], errors="coerce")
        cambio = pd.to_numeric(df[mapa["cambio"]], errors="coerce")
        df["petroleo_brl_nivel"] = petroleo * cambio
        df["dln_petroleo_brl"] = 100 * safe_log_diff(df["petroleo_brl_nivel"])
        transformadas["petroleo_brl"] = "dln_petroleo_brl"

    # Combustíveis.
    if mapa["gasolina_refinaria"]:
        df["dln_gasolina_refinaria"] = 100 * safe_log_diff(df[mapa["gasolina_refinaria"]])
        transformadas["gasolina_refinaria"] = "dln_gasolina_refinaria"

    if mapa["gasolina"]:
        df["dln_gasolina"] = 100 * safe_log_diff(df[mapa["gasolina"]])
        transformadas["gasolina"] = "dln_gasolina"

    if mapa["etanol"]:
        df["dln_etanol"] = 100 * safe_log_diff(df[mapa["etanol"]])
        transformadas["etanol"] = "dln_etanol"

    if mapa["diesel"]:
        df["dln_diesel"] = 100 * safe_log_diff(df[mapa["diesel"]])
        transformadas["diesel"] = "dln_diesel"

    # Controles.
    if mapa["atividade"]:
        df["dln_atividade"] = 100 * safe_log_diff(df[mapa["atividade"]])
        transformadas["atividade"] = "dln_atividade"

    if mapa["ipca_geral"]:
        df["ipca_geral_mensal"] = diff_se_precisa(df[mapa["ipca_geral"]])
        transformadas["ipca_geral"] = "ipca_geral_mensal"

    if mapa["ipca_transporte"]:
        df["ipca_transporte_mensal"] = diff_se_precisa(df[mapa["ipca_transporte"]])
        transformadas["ipca_transporte"] = "ipca_transporte_mensal"

    if mapa["selic"]:
        df["selic_controle"] = pd.to_numeric(df[mapa["selic"]], errors="coerce")
        transformadas["selic"] = "selic_controle"

    if mapa["expectativa"]:
        df["expectativa_controle"] = pd.to_numeric(df[mapa["expectativa"]], errors="coerce")
        transformadas["expectativa"] = "expectativa_controle"

    if mapa["stringency"]:
        df["stringency_controle"] = pd.to_numeric(df[mapa["stringency"]], errors="coerce")
        transformadas["stringency"] = "stringency_controle"

    # Dummies mensais.
    dummy_cols = []
    if USAR_DUMMIES_MENSAIS:
        dummies = pd.get_dummies(df["mes"], prefix="mes", drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        dummy_cols = list(dummies.columns)

    print("\nVariáveis transformadas criadas:")
    for k, v in transformadas.items():
        print(f"- {k}: {v}")

    # Salva uma cópia da base transformada para conferência.
    df.to_excel(OUTPUT_DIR / "base_transformada_modelo4.xlsx", index=False)

    return df, transformadas, dummy_cols


def ajustar_ols_hac(Y, X, h):
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(Y, X)

    if USAR_HAC:
        maxlags = max(1, h)
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    return model.fit(cov_type="HC1")


def local_projection_simples(
    df,
    y,
    shock,
    controls=None,
    h_max=24,
    acumulada=True,
    nome_modelo="modelo",
    subpasta="lp_simples"
):
    """
    LP tradicional:
    y acumulado ou pontual = alpha + beta * choque + controles + erro.
    """

    controls = controls or []
    base = df.copy()

    shock_usado = shock
    if PADRONIZAR_CHOQUE:
        sd = base[shock].std(skipna=True)
        if pd.notna(sd) and sd > 0:
            shock_usado = f"{shock}_std"
            base[shock_usado] = base[shock] / sd

    regressores_fixos = []

    regressores_fixos += criar_lags(base, y, LAGS_Y, prefix=y)
    regressores_fixos += criar_lags(base, shock_usado, LAGS_SHOCK, prefix=shock_usado)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            regressores_fixos += criar_lags(base, c, LAGS_CONTROLS, prefix=c)

    resultados = []

    for h in range(0, h_max + 1):
        temp = base.copy()

        if acumulada:
            cols_futuras = []
            for j in range(0, h + 1):
                col_fut = f"{y}_lead{j}"
                temp[col_fut] = temp[y].shift(-j)
                cols_futuras.append(col_fut)
            temp[f"y_h{h}"] = temp[cols_futuras].sum(axis=1, min_count=h + 1)
        else:
            temp[f"y_h{h}"] = temp[y].shift(-h)

        X_cols = [shock_usado] + regressores_fixos
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

            coef = res.params.get(shock_usado, np.nan)
            se = res.bse.get(shock_usado, np.nan)
            t = res.tvalues.get(shock_usado, np.nan)
            pvalor = res.pvalues.get(shock_usado, np.nan)

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
            print(f"Erro em {nome_modelo}, h={h}: {e}")
            resultados.append({
                "h": h, "coef": np.nan, "se": np.nan, "t": np.nan,
                "pvalor": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "nobs": len(temp_reg)
            })

    tab = pd.DataFrame(resultados)

    pasta = OUTPUT_DIR / subpasta
    pasta.mkdir(parents=True, exist_ok=True)

    sufixo = "acumulada" if acumulada else "pontual"
    tab.to_csv(pasta / f"lp_{sufixo}_{nome_modelo}.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tab["h"], tab["coef"], marker="o", label="Resposta estimada")
    ax.fill_between(tab["h"], tab["ci_low"], tab["ci_high"], alpha=0.2, label=f"IC {int(CONF * 100)}%")
    ax.axhline(0, linewidth=1)
    ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"LP {sufixo} - {nome_modelo}")
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("Resposta estimada")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(pasta / f"lp_{sufixo}_{nome_modelo}.png", dpi=300)
    plt.close()

    return tab


def local_projection_regime_wald(
    df,
    y,
    shock,
    controls=None,
    h_max=24,
    acumulada=True,
    nome_modelo="modelo_regime",
    subpasta="lp_regime_wald",
    regime_col="D_pos2015"
):
    """
    LP com interação de regime:
    y = alpha + beta_pre * shock_pre + beta_pos * shock_pos + controles + erro

    shock_pre = shock * (1 - D_pos2015)
    shock_pos = shock * D_pos2015

    Faz teste de Wald:
    beta_pre = beta_pos

    A saída permite saber se o repasse mudou estatisticamente entre os regimes.
    """

    controls = controls or []
    base = df.copy()

    shock_usado = shock
    if PADRONIZAR_CHOQUE:
        sd = base[shock].std(skipna=True)
        if pd.notna(sd) and sd > 0:
            shock_usado = f"{shock}_std"
            base[shock_usado] = base[shock] / sd

    base["shock_pre2015"] = base[shock_usado] * (1 - base[regime_col])
    base["shock_pos2015"] = base[shock_usado] * base[regime_col]

    regressores_fixos = []

    regressores_fixos += criar_lags(base, y, LAGS_Y, prefix=y)
    regressores_fixos += criar_lags(base, shock_usado, LAGS_SHOCK, prefix=shock_usado)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            regressores_fixos += criar_lags(base, c, LAGS_CONTROLS, prefix=c)

    resultados = []

    for h in range(0, h_max + 1):
        temp = base.copy()

        if acumulada:
            cols_futuras = []
            for j in range(0, h + 1):
                col_fut = f"{y}_lead{j}"
                temp[col_fut] = temp[y].shift(-j)
                cols_futuras.append(col_fut)
            temp[f"y_h{h}"] = temp[cols_futuras].sum(axis=1, min_count=h + 1)
        else:
            temp[f"y_h{h}"] = temp[y].shift(-h)

        X_cols = ["shock_pre2015", "shock_pos2015"] + regressores_fixos
        X_cols = [c for c in X_cols if c in temp.columns]

        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(MIN_OBS, len(X_cols) + 10):
            resultados.append({
                "h": h,
                "coef_pre2015": np.nan,
                "se_pre2015": np.nan,
                "pvalor_pre2015": np.nan,
                "ci_low_pre2015": np.nan,
                "ci_high_pre2015": np.nan,
                "coef_pos2015": np.nan,
                "se_pos2015": np.nan,
                "pvalor_pos2015": np.nan,
                "ci_low_pos2015": np.nan,
                "ci_high_pos2015": np.nan,
                "diff_pos_menos_pre": np.nan,
                "pvalor_wald_diff": np.nan,
                "nobs": len(temp_reg)
            })
            continue

        Y = temp_reg[f"y_h{h}"]
        X = temp_reg[X_cols]

        try:
            res = ajustar_ols_hac(Y, X, h)

            b_pre = res.params.get("shock_pre2015", np.nan)
            se_pre = res.bse.get("shock_pre2015", np.nan)
            p_pre = res.pvalues.get("shock_pre2015", np.nan)

            b_pos = res.params.get("shock_pos2015", np.nan)
            se_pos = res.bse.get("shock_pos2015", np.nan)
            p_pos = res.pvalues.get("shock_pos2015", np.nan)

            try:
                wald = res.wald_test("shock_pre2015 = shock_pos2015", scalar=True)
                p_wald = float(wald.pvalue)
            except Exception:
                p_wald = np.nan

            resultados.append({
                "h": h,
                "coef_pre2015": b_pre,
                "se_pre2015": se_pre,
                "pvalor_pre2015": p_pre,
                "ci_low_pre2015": b_pre - Z_CRIT * se_pre,
                "ci_high_pre2015": b_pre + Z_CRIT * se_pre,
                "coef_pos2015": b_pos,
                "se_pos2015": se_pos,
                "pvalor_pos2015": p_pos,
                "ci_low_pos2015": b_pos - Z_CRIT * se_pos,
                "ci_high_pos2015": b_pos + Z_CRIT * se_pos,
                "diff_pos_menos_pre": b_pos - b_pre,
                "pvalor_wald_diff": p_wald,
                "nobs": int(res.nobs)
            })

        except Exception as e:
            print(f"Erro em {nome_modelo}, h={h}: {e}")
            resultados.append({
                "h": h,
                "coef_pre2015": np.nan,
                "se_pre2015": np.nan,
                "pvalor_pre2015": np.nan,
                "ci_low_pre2015": np.nan,
                "ci_high_pre2015": np.nan,
                "coef_pos2015": np.nan,
                "se_pos2015": np.nan,
                "pvalor_pos2015": np.nan,
                "ci_low_pos2015": np.nan,
                "ci_high_pos2015": np.nan,
                "diff_pos_menos_pre": np.nan,
                "pvalor_wald_diff": np.nan,
                "nobs": len(temp_reg)
            })

    tab = pd.DataFrame(resultados)

    pasta = OUTPUT_DIR / subpasta
    pasta.mkdir(parents=True, exist_ok=True)

    sufixo = "acumulada" if acumulada else "pontual"
    tab.to_csv(pasta / f"lp_regime_{sufixo}_{nome_modelo}.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(tab["h"], tab["coef_pre2015"], marker="o", label="Pré-2015")
    ax.fill_between(tab["h"], tab["ci_low_pre2015"], tab["ci_high_pre2015"], alpha=0.2)

    ax.plot(tab["h"], tab["coef_pos2015"], marker="s", label="Pós-2015")
    ax.fill_between(tab["h"], tab["ci_low_pos2015"], tab["ci_high_pos2015"], alpha=0.2)

    ax.axhline(0, linewidth=1)
    ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"LP por regime - {nome_modelo}")
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("Resposta estimada")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(pasta / f"lp_regime_{sufixo}_{nome_modelo}.png", dpi=300)
    plt.close()

    # Gráfico do p-valor do teste de diferença entre regimes.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tab["h"], tab["pvalor_wald_diff"], marker="o")
    ax.axhline(0.10, linestyle="--", linewidth=1, label="10%")
    ax.axhline(0.05, linestyle="--", linewidth=1, label="5%")
    ax.set_ylim(0, 1)
    ax.set_title(f"Teste de diferença entre regimes - {nome_modelo}")
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("p-valor do teste de Wald")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(pasta / f"wald_pvalor_{nome_modelo}.png", dpi=300)
    plt.close()

    return tab


# =============================================================================
# 4. ESTIMAÇÕES DO MODELO 4
# =============================================================================

def controles_macro(vars_t, dummy_cols):
    controles = []

    for k in ["cambio", "atividade", "selic", "expectativa", "stringency"]:
        if k in vars_t:
            controles.append(vars_t[k])

    controles = controles + dummy_cols
    return controles


def estimar_brent_usd_vs_brl(df, vars_t, dummy_cols):
    """
    Compara petróleo em dólar e petróleo em reais.
    """

    print("\n" + "=" * 100)
    print("2) ROBUSTEZ: PETRÓLEO EM DÓLAR VS PETRÓLEO EM REAIS")
    print("=" * 100)

    controles = controles_macro(vars_t, dummy_cols)

    choques_petroleo = []
    if "petroleo_usd" in vars_t:
        choques_petroleo.append(("petroleo_usd", vars_t["petroleo_usd"]))
    if "petroleo_brl" in vars_t:
        choques_petroleo.append(("petroleo_brl", vars_t["petroleo_brl"]))

    combustiveis = ["gasolina_refinaria", "gasolina", "etanol", "diesel"]
    inflacoes = ["ipca_geral", "ipca_transporte"]

    # Petróleo USD/BRL -> combustíveis.
    for nome_shock, shock in choques_petroleo:
        for comb in combustiveis:
            if comb not in vars_t:
                continue

            nome = f"{nome_shock}_para_{comb}"
            print(f"Estimando: {nome}")

            local_projection_simples(
                df=df,
                y=vars_t[comb],
                shock=shock,
                controls=controles,
                h_max=H_MAX,
                acumulada=True,
                nome_modelo=nome,
                subpasta="A_usd_vs_brl_petroleo_para_combustiveis"
            )

    # Petróleo USD/BRL -> IPCA.
    for nome_shock, shock in choques_petroleo:
        for infl in inflacoes:
            if infl not in vars_t:
                continue

            nome = f"{nome_shock}_para_{infl}"
            print(f"Estimando: {nome}")

            local_projection_simples(
                df=df,
                y=vars_t[infl],
                shock=shock,
                controls=controles,
                h_max=H_MAX,
                acumulada=True,
                nome_modelo=nome,
                subpasta="B_usd_vs_brl_petroleo_para_ipca"
            )


def estimar_regimes_com_wald(df, vars_t, dummy_cols):
    """
    Estima interação com regime e teste formal de diferença entre regimes.
    """

    print("\n" + "=" * 100)
    print("3) REGIMES PETROBRAS COM TESTE FORMAL DE WALD")
    print("=" * 100)

    controles = controles_macro(vars_t, dummy_cols)

    # Para evitar excesso de modelos, foco nos canais principais.
    modelos = []

    # Petróleo em reais é prioridade para regimes.
    shock_petroleo = None
    if "petroleo_brl" in vars_t:
        shock_petroleo = "petroleo_brl"
    elif "petroleo_usd" in vars_t:
        shock_petroleo = "petroleo_usd"

    if shock_petroleo is not None:
        if "gasolina_refinaria" in vars_t:
            modelos.append(("gasolina_refinaria", shock_petroleo, f"{shock_petroleo}_para_gasolina_refinaria"))
        if "gasolina" in vars_t:
            modelos.append(("gasolina", shock_petroleo, f"{shock_petroleo}_para_gasolina"))
        if "diesel" in vars_t:
            modelos.append(("diesel", shock_petroleo, f"{shock_petroleo}_para_diesel"))
        if "ipca_transporte" in vars_t:
            modelos.append(("ipca_transporte", shock_petroleo, f"{shock_petroleo}_para_ipca_transporte"))
        if "ipca_geral" in vars_t:
            modelos.append(("ipca_geral", shock_petroleo, f"{shock_petroleo}_para_ipca_geral"))

    # Combustíveis para inflação, especialmente gasolina e refinaria.
    for comb in ["gasolina_refinaria", "gasolina", "diesel", "etanol"]:
        if comb not in vars_t:
            continue

        if "ipca_transporte" in vars_t:
            modelos.append(("ipca_transporte", comb, f"{comb}_para_ipca_transporte"))
        if "ipca_geral" in vars_t:
            modelos.append(("ipca_geral", comb, f"{comb}_para_ipca_geral"))

    for y_key, shock_key, nome in modelos:
        y = vars_t[y_key]
        shock = vars_t[shock_key]

        controles_modelo = controles.copy()

        # Se o choque for combustível, controla por petróleo em reais ou dólar.
        if shock_key not in ["petroleo_brl", "petroleo_usd"]:
            if "petroleo_brl" in vars_t:
                controles_modelo.append(vars_t["petroleo_brl"])
            elif "petroleo_usd" in vars_t:
                controles_modelo.append(vars_t["petroleo_usd"])

        print(f"Estimando regime/Wald: {nome}")

        local_projection_regime_wald(
            df=df,
            y=y,
            shock=shock,
            controls=controles_modelo,
            h_max=H_MAX,
            acumulada=True,
            nome_modelo=nome,
            subpasta="C_regimes_wald"
        )


def criar_tabelas_resumo():
    print("\n" + "=" * 100)
    print("4) CRIANDO TABELAS-RESUMO")
    print("=" * 100)

    arquivos = list(OUTPUT_DIR.rglob("*.csv"))

    resumo_simples = []
    resumo_regime = []

    for arq in arquivos:
        try:
            tab = pd.read_csv(arq)
        except Exception:
            continue

        # Resultados simples.
        if "coef" in tab.columns:
            for h_ref in [0, 3, 6, 12, 18, 24]:
                if h_ref in tab["h"].values:
                    row = tab.loc[tab["h"] == h_ref].iloc[0].to_dict()
                    row["arquivo"] = str(arq)
                    row["h_ref"] = h_ref
                    row["significativo_10pct"] = (
                        pd.notna(row.get("ci_low")) and pd.notna(row.get("ci_high")) and
                        ((row.get("ci_low") > 0) or (row.get("ci_high") < 0))
                    )
                    resumo_simples.append(row)

        # Resultados com regime.
        if "coef_pre2015" in tab.columns:
            for h_ref in [0, 3, 6, 12, 18, 24]:
                if h_ref in tab["h"].values:
                    row = tab.loc[tab["h"] == h_ref].iloc[0].to_dict()
                    row["arquivo"] = str(arq)
                    row["h_ref"] = h_ref
                    row["pre_significativo_10pct"] = (
                        pd.notna(row.get("ci_low_pre2015")) and pd.notna(row.get("ci_high_pre2015")) and
                        ((row.get("ci_low_pre2015") > 0) or (row.get("ci_high_pre2015") < 0))
                    )
                    row["pos_significativo_10pct"] = (
                        pd.notna(row.get("ci_low_pos2015")) and pd.notna(row.get("ci_high_pos2015")) and
                        ((row.get("ci_low_pos2015") > 0) or (row.get("ci_high_pos2015") < 0))
                    )
                    row["diferenca_regimes_10pct"] = (
                        pd.notna(row.get("pvalor_wald_diff")) and row.get("pvalor_wald_diff") < 0.10
                    )
                    resumo_regime.append(row)

    if resumo_simples:
        df_resumo = pd.DataFrame(resumo_simples)
        df_resumo.to_csv(
            OUTPUT_DIR / "resumo_modelo4_usd_vs_brl_horizontes.csv",
            index=False,
            encoding="utf-8-sig"
        )
        df_resumo.to_excel(
            OUTPUT_DIR / "resumo_modelo4_usd_vs_brl_horizontes.xlsx",
            index=False
        )

    if resumo_regime:
        df_regime = pd.DataFrame(resumo_regime)
        df_regime.to_csv(
            OUTPUT_DIR / "resumo_modelo4_regimes_wald_horizontes.csv",
            index=False,
            encoding="utf-8-sig"
        )
        df_regime.to_excel(
            OUTPUT_DIR / "resumo_modelo4_regimes_wald_horizontes.xlsx",
            index=False
        )

    # Relatório textual simples.
    texto = []
    texto.append("RELATÓRIO DO MODELO 4\n")
    texto.append("=====================\n\n")
    texto.append("O Modelo 4 adiciona duas robustezes ao modelo principal de Local Projections.\n\n")
    texto.append("1. Brent em reais:\n")
    texto.append("   Foi criada a variável petroleo_brl_nivel = Preco_Barril * Cambio.\n")
    texto.append("   Em seguida, foi criada dln_petroleo_brl = 100 * Δlog(petroleo_brl_nivel).\n")
    texto.append("   Isso permite comparar choques no petróleo em dólar com choques no petróleo em reais.\n\n")
    texto.append("2. Regimes Petrobras com teste formal:\n")
    texto.append("   Foi criada uma dummy D_pos2015, igual a 1 a partir de 2015-01-01.\n")
    texto.append("   O choque foi separado entre shock_pre2015 e shock_pos2015.\n")
    texto.append("   O teste de Wald verifica H0: beta_pre2015 = beta_pos2015.\n\n")
    texto.append("Como interpretar:\n")
    texto.append("- Se o Brent em reais gerar respostas mais fortes, o câmbio é parte importante do canal.\n")
    texto.append("- Se o p-valor do Wald for menor que 0,10, há evidência de diferença entre regimes.\n")
    texto.append("- Se o IPCA Transportes responder mais que o IPCA Geral, o canal dos combustíveis é setorialmente mais forte.\n")
    texto.append("- Se gasolina de refinaria responder antes da gasolina ao consumidor, há evidência do canal Petrobras/refinaria.\n")

    (OUTPUT_DIR / "RELATORIO_MODELO4.txt").write_text("".join(texto), encoding="utf-8")


# =============================================================================
# 5. EXECUÇÃO
# =============================================================================

def main():
    df, vars_t, dummy_cols = preparar_base()

    if "petroleo_usd" not in vars_t:
        raise ValueError("Não encontrei a variável de petróleo. Verifique o nome da coluna Preco_Barril/Brent.")

    if "petroleo_brl" not in vars_t:
        print("Atenção: não foi possível criar petróleo em reais. Verifique se existem colunas de petróleo e câmbio.")

    estimar_brent_usd_vs_brl(df, vars_t, dummy_cols)
    estimar_regimes_com_wald(df, vars_t, dummy_cols)
    criar_tabelas_resumo()

    print("\n" + "=" * 100)
    print("MODELO 4 FINALIZADO")
    print("=" * 100)
    print(f"Arquivos salvos em: {OUTPUT_DIR.resolve()}")
    print("\nUse principalmente os horizontes 3, 6 e 12 meses.")
    print("Os horizontes 18 e 24 meses devem ser tratados como robustez, pois tendem a ter maior incerteza.")


if __name__ == "__main__":
    main()
