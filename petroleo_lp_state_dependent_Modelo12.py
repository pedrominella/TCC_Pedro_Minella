# -*- coding: utf-8 -*-
"""
petroleo_lp_state_dependent_Modelo12.py
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ARQUIVO_IPCA = r"IPCA.xlsx"
ABA_IPCA = 0
OUTPUT_DIR = Path("output_petroleo_lp_state_dependent_Modelo12")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_INICIO = "2003-01-01"
DATA_CORTE_POLITICA_PETROBRAS = "2016-09-01"

H_MAX = 24
LAGS_Y = 3
LAGS_SHOCK = 3
LAGS_CONTROLS = 3
CONF = 0.90
Z_CRIT = 1.645

def safe_log_diff(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s.where(s > 0)).diff()

def diff_se_precisa(s):
    s = pd.to_numeric(s, errors="coerce")
    med = s.dropna().median()
    if pd.notna(med) and med > 20:
        return 100 * safe_log_diff(s)
    return s

def criar_lags(df, var, n_lags, prefix=None):
    if prefix is None: prefix = var
    lag_cols = []
    for L in range(1, n_lags + 1):
        col = f"{prefix}_lag{L}"
        df[col] = df[var].shift(L)
        lag_cols.append(col)
    return lag_cols

def carregar_preparar_base_ipca():
    df = pd.read_excel(ARQUIVO_IPCA, sheet_name=ABA_IPCA)
    df.columns = [str(c).strip() for c in df.columns]
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)
    df["Data"] = df["Data"].dt.to_period("M").dt.to_timestamp()
    df["mes"] = df["Data"].dt.month

    # Regime Petrobras
    df["regime_petrobras"] = (df["Data"] >= pd.to_datetime(DATA_CORTE_POLITICA_PETROBRAS)).astype(float)
    df["regime_petrobras_lag1"] = df["regime_petrobras"].shift(1).fillna(0) # I_{t-1}

    df["dln_petroleo"] = 100 * safe_log_diff(df["Preco_Barril"])
    df["dln_cambio"] = 100 * safe_log_diff(df["Cambio"])
    df["dln_gasolina"] = 100 * safe_log_diff(df["Gasolina_nivel"])
    df["dln_diesel"] = 100 * safe_log_diff(df["Oleo_diesel_nivel"])
    df["dln_atividade"] = 100 * safe_log_diff(df["Atividade"])
    df["ipca_geral_mensal"] = diff_se_precisa(df["IPCA_Geral_nivel"])
    df["ipca_transporte_mensal"] = diff_se_precisa(df["IPCA_Trans_nivel"])
    df["selic_controle"] = pd.to_numeric(df["Selic"], errors="coerce")
    df["expectativa_controle"] = pd.to_numeric(df["Expectativa_inflacao"], errors="coerce")

    dummies = pd.get_dummies(df["mes"], prefix="mes", drop_first=True, dtype=float)
    df = pd.concat([df, dummies], axis=1)
    
    base = df[(df["Data"] >= pd.to_datetime(DATA_INICIO))].copy()
    base = base.sort_values("Data").reset_index(drop=True)
    
    return base, list(dummies.columns)

def montar_y_h(temp, y, h):
    cols_futuras = []
    for j in range(0, h + 1):
        col_fut = f"{y}_lead{j}"
        temp[col_fut] = temp[y].shift(-j)
        cols_futuras.append(col_fut)
    temp[f"y_h{h}"] = temp[cols_futuras].sum(axis=1, min_count=h + 1)
    return temp

def local_projection_state_dependent(df, y, shock, state_var, controls, dummy_cols, h_max=24, nome_modelo="SD_LP"):
    base = df.copy()
    
    # I_{t-1}
    I = base[state_var]
    # (1 - I_{t-1})
    I_comp = 1 - I
    
    base["shock_state1"] = base[shock] * I         # Regime pós-2016
    base["shock_state0"] = base[shock] * I_comp    # Regime pré-2016
    
    regressores_base = []
    regressores_base += criar_lags(base, y, LAGS_Y, prefix=y)
    regressores_base += criar_lags(base, shock, LAGS_SHOCK, prefix=shock)
    
    for c in controls:
        if c in base.columns:
            regressores_base.append(c)
            regressores_base += criar_lags(base, c, LAGS_CONTROLS, prefix=c)
            
    # Cria os regressores interagidos
    X_state1 = ["shock_state1"] + [c + "_state1" for c in regressores_base]
    X_state0 = ["shock_state0"] + [c + "_state0" for c in regressores_base]
    
    for c in regressores_base:
        base[c + "_state1"] = base[c] * I
        base[c + "_state0"] = base[c] * I_comp

    resultados_s1 = []
    resultados_s0 = []

    for h in range(0, h_max + 1):
        temp = base.copy()
        temp = montar_y_h(temp, y, h)
        
        X_cols = X_state1 + X_state0 + dummy_cols
        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(temp_reg) < len(X_cols) + 10:
            continue
            
        Y = temp_reg[f"y_h{h}"]
        X = sm.add_constant(temp_reg[X_cols], has_constant="add")
        
        try:
            res = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, h)})
            
            # Regime Pos-2016
            coef1 = res.params.get("shock_state1", np.nan)
            se1 = res.bse.get("shock_state1", np.nan)
            resultados_s1.append({
                "h": h, "coef": coef1, "ci_low": coef1 - Z_CRIT * se1, "ci_high": coef1 + Z_CRIT * se1
            })
            
            # Regime Pre-2016
            coef0 = res.params.get("shock_state0", np.nan)
            se0 = res.bse.get("shock_state0", np.nan)
            resultados_s0.append({
                "h": h, "coef": coef0, "ci_low": coef0 - Z_CRIT * se0, "ci_high": coef0 + Z_CRIT * se0
            })
            
        except Exception as e:
            print(f"Erro no h={h}: {e}")
            
    tab1 = pd.DataFrame(resultados_s1)
    tab0 = pd.DataFrame(resultados_s0)
    
    if tab1.empty or tab0.empty: return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # State 0 (Pre-2016)
    axes[0].plot(tab0["h"], tab0["coef"], marker="o", label="Pre-2016", color="blue")
    axes[0].fill_between(tab0["h"], tab0["ci_low"], tab0["ci_high"], alpha=0.2, color="blue")
    axes[0].axhline(0, linewidth=1, color="black")
    axes[0].set_title(f"Pré-Setembro/2016: {y}")
    axes[0].grid(True, alpha=0.3)
    
    # State 1 (Pos-2016)
    axes[1].plot(tab1["h"], tab1["coef"], marker="o", label="Pós-2016", color="red")
    axes[1].fill_between(tab1["h"], tab1["ci_low"], tab1["ci_high"], alpha=0.2, color="red")
    axes[1].axhline(0, linewidth=1, color="black")
    axes[1].set_title(f"Pós-Setembro/2016: {y}")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{nome_modelo}_{y}.png", dpi=300)
    plt.close()
    
    tab1.to_csv(OUTPUT_DIR / f"{nome_modelo}_{y}_pos2016.csv", index=False)
    tab0.to_csv(OUTPUT_DIR / f"{nome_modelo}_{y}_pre2016.csv", index=False)

def main():
    base, dummy_cols = carregar_preparar_base_ipca()
    controls = ["dln_cambio", "dln_atividade", "selic_controle", "expectativa_controle"]
    
    targets = ["dln_gasolina", "dln_diesel", "ipca_geral_mensal", "ipca_transporte_mensal"]
    for t in targets:
        print(f"Rodando State-Dependent LP para {t}...")
        local_projection_state_dependent(
            base, y=t, shock="dln_petroleo", state_var="regime_petrobras_lag1", 
            controls=controls, dummy_cols=dummy_cols, h_max=H_MAX, nome_modelo="SD_LP"
        )
        
if __name__ == "__main__":
    main()
