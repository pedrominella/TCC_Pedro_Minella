# -*- coding: utf-8 -*-
"""
comparar_brent_reais_dolar.py

Objetivo:
---------
Este script realiza uma análise comparativa do impacto de choques no preço do petróleo
mensurado em duas moedas distintas no modelo de Projeções Locais (LP-OLS):
1. **Brent em Reais (BRL)**: choque em dln_petroleo_brl_std (Brent * Câmbio).
2. **Brent em Dólares (USD)**: choque em dln_petroleo_usd_std (Brent em USD).

Ambas as estimações controlam diretamente pelo Índice Kilian de Atividade Econômica
Global (IGREA) e outros controles macroeconômicos clássicos (IBC-Br, Selic, Câmbio,
Expectativas e Dummies Sazonais).

Os gráficos gerados são 100% LIMPOS de estatística-F do primeiro estágio ou curvas
LP-IV, focando puramente no contraste entre as duas moedas no LP-OLS.
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# Configurações de caminhos
BASE_EXCEL = Path("output_petroleo_lp_modelo10_kilian") / "base_modelo10_kilian.xlsx"
OUTPUT_DIR = Path("output_petroleo_lp_modelo10_kilian") / "comparativo_brent_usd_brl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parâmetros de projeção
H_MAX = 12
LAGS_Y = 3
LAGS_SHOCK = 3
LAGS_CONTROLS = 3
CONF = 0.90
Z_CRIT = 1.645
USAR_HAC = True
MIN_OBS = 60

# Mapeamento de variáveis
VARS_T = {
    "petroleo_brl": "dln_petroleo_brl_std",
    "petroleo_usd": "dln_petroleo_usd_std",  # Criado e padronizado dinamicamente
    "kilian": "kilian_std",
    "cambio": "dln_cambio",
    "atividade": "dln_atividade",
    "selic": "selic_controle",
    "expectativa": "expectativa_controle",
    "regime_petrobras": "regime_petrobras_pos_set2016",
    
    # Variáveis alvo
    "gasolina_refinaria": "dln_gasolina_refinaria",
    "gasolina": "dln_gasolina",
    "diesel": "dln_diesel",
    "etanol": "dln_etanol",
    "ipca_geral": "ipca_geral_mensal",
    "ipca_transporte": "ipca_transporte_mensal"
}

ALVOS = {
    "gasolina_refinaria": "Preço Gasolina na Refinaria (dlog %)",
    "gasolina": "Preço Gasolina ao Consumidor (dlog %)",
    "diesel": "Preço Diesel ao Consumidor (dlog %)",
    "etanol": "Preço Etanol ao Consumidor (dlog %)",
    "ipca_geral": "IPCA Geral Mensal (%)",
    "ipca_transporte": "IPCA Transportes Mensal (%)"
}

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

def montar_y_h(temp, y, h):
    cols_futuras = []
    for j in range(0, h + 1):
        col_fut = f"{y}_lead{j}"
        temp[col_fut] = temp[y].shift(-j)
        cols_futuras.append(col_fut)
    temp[f"y_h{h}"] = temp[cols_futuras].sum(axis=1, min_count=h + 1)
    return temp

def ajustar_ols_hac(Y, X, h):
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(Y, X)
    if USAR_HAC:
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": max(1, h)})
    return model.fit(cov_type="HC1")

def estimar_lp_ols(base, y_name, shock, kilian_index, controls, h_max=12):
    resultados = []
    regressores_fixos = []
    
    # Defasagens da variável dependente e do choque
    regressores_fixos += criar_lags(base, y_name, LAGS_Y, prefix=y_name)
    regressores_fixos += criar_lags(base, shock, LAGS_SHOCK, prefix=shock)
    
    # Índice Kilian (em nível e defasado) como controle
    regressores_fixos.append(kilian_index)
    regressores_fixos += criar_lags(base, kilian_index, LAGS_CONTROLS, prefix=kilian_index)
    
    # Outros controles macroeconômicos
    for c in controls:
        if c in base.columns and c != kilian_index and c != shock:
            regressores_fixos.append(c)
            regressores_fixos += criar_lags(base, c, LAGS_CONTROLS, prefix=c)
            
    for h in range(0, h_max + 1):
        temp = base.copy()
        temp = montar_y_h(temp, y_name, h)
        
        X_cols = [shock] + regressores_fixos
        X_cols = [c for c in X_cols if c in temp.columns]
        
        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(temp_reg) < max(MIN_OBS, len(X_cols) + 10):
            resultados.append({
                "h": h, "coef": np.nan, "se": np.nan, "t": np.nan, "pvalor": np.nan,
                "ci_low": np.nan, "ci_high": np.nan
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
                "h": h, "coef": coef, "se": se, "t": t, "pvalor": pvalor,
                "ci_low": coef - Z_CRIT * se, "ci_high": coef + Z_CRIT * se
            })
        except Exception:
            resultados.append({
                "h": h, "coef": np.nan, "se": np.nan, "t": np.nan, "pvalor": np.nan,
                "ci_low": np.nan, "ci_high": np.nan
            })
            
    return pd.DataFrame(resultados)

def main():
    print("=" * 100)
    print("INICIANDO ANALISE COMPARATIVA: BRENT EM REAIS (BRL) VS BRENT EM DOLARES (USD)")
    print("=" * 100)
    
    if not BASE_EXCEL.exists():
        raise FileNotFoundError(f"Não encontrei a base preparada em: {BASE_EXCEL}")
        
    base = pd.read_excel(BASE_EXCEL)
    
    # Criar e padronizar o choque de petróleo em USD
    if "dln_petroleo_usd" in base.columns:
        base["dln_petroleo_usd_std"] = padronizar(base["dln_petroleo_usd"])
        print("Sucesso: Choque do Brent em USD padronizado.")
    else:
        raise KeyError("Coluna dln_petroleo_usd nao encontrada na base excel.")
        
    # Identificar colunas de dummies sazonais
    dummy_cols = [c for c in base.columns if c.startswith("mes_")]
    
    # Controles principais
    controles = []
    for k in ["cambio", "atividade", "selic", "expectativa", "regime_petrobras"]:
        if k in VARS_T and VARS_T[k] in base.columns:
            controles.append(VARS_T[k])
    controles = controles + dummy_cols
    
    kilian_index = VARS_T["kilian"]
    
    registros_comparacao = []
    
    for alvo_chave, alvo_label in ALVOS.items():
        if alvo_chave not in VARS_T:
            continue
            
        y_name = VARS_T[alvo_chave]
        if y_name not in base.columns:
            continue
            
        print(f"\nEstimando modelos comparativos para: {alvo_label}...")
        
        # 1. Estimar OLS com Brent em Reais (BRL)
        df_brl = estimar_lp_ols(base, y_name, VARS_T["petroleo_brl"], kilian_index, controles, h_max=H_MAX)
        
        # 2. Estimar OLS com Brent em Dólares (USD)
        df_usd = estimar_lp_ols(base, y_name, VARS_T["petroleo_usd"], kilian_index, controles, h_max=H_MAX)
        
        # Salvar tabelas individuais em CSV
        df_brl.to_csv(OUTPUT_DIR / f"tabela_LP_OLS_BRL_{alvo_chave}.csv", index=False)
        df_usd.to_csv(OUTPUT_DIR / f"tabela_LP_OLS_USD_{alvo_chave}.csv", index=False)
        
        # Guardar valores de 12 meses para tabela síntese
        coef_brl_12 = df_brl.loc[df_brl["h"] == 12, "coef"].values[0]
        p_brl_12 = df_brl.loc[df_brl["h"] == 12, "pvalor"].values[0]
        coef_usd_12 = df_usd.loc[df_usd["h"] == 12, "coef"].values[0]
        p_usd_12 = df_usd.loc[df_usd["h"] == 12, "pvalor"].values[0]
        
        registros_comparacao.append({
            "Variavel Alvo": alvo_label,
            "BRL Coef (12m)": coef_brl_12,
            "BRL P-Value (12m)": p_brl_12,
            "BRL Signif (12m)": "Sim" if p_brl_12 < 0.10 else "Nao",
            "USD Coef (12m)": coef_usd_12,
            "USD P-Value (12m)": p_usd_12,
            "USD Signif (12m)": "Sim" if p_usd_12 < 0.10 else "Nao",
        })
        
        # Gerar gráfico comparativo super limpo
        fig, ax = plt.subplots(figsize=(9.5, 6))
        
        # Curva 1: Brent BRL (Royal Blue)
        ax.plot(df_brl["h"], df_brl["coef"], marker="o", color="#1a73e8", linewidth=2.5, label="Choque do Brent em Reais (BRL)")
        ax.fill_between(df_brl["h"], df_brl["ci_low"], df_brl["ci_high"], color="#1a73e8", alpha=0.12)
        
        # Curva 2: Brent USD (Deep Red / Crimson)
        ax.plot(df_usd["h"], df_usd["coef"], marker="^", color="#d93025", linewidth=2.2, linestyle="--", label="Choque do Brent em Dólares (USD)")
        ax.fill_between(df_usd["h"], df_usd["ci_low"], df_usd["ci_high"], color="#d93025", alpha=0.08)
        
        # Linhas de referência
        ax.axhline(0, color="#202124", linewidth=1.0, linestyle="-")
        
        # Estética Premium
        ax.set_title(f"Impacto Comparativo: Petróleo em BRL vs. USD $\\rightarrow$ {alvo_label}\n(Modelos LP-OLS Limpos - Controle Kilian)", fontsize=11.5, fontweight="bold", pad=15)
        ax.set_xlabel("Horizonte de Projeção (h, em meses)", fontsize=10)
        ax.set_ylabel("Impacto Acumulado (%)", fontsize=10)
        ax.set_xticks(range(0, H_MAX + 1))
        ax.grid(True, linestyle=":", alpha=0.5, color="#dadce0")
        ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor="#dadce0")
        
        # Remover bordas desnecessárias (spines)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        nome_img = f"comparativo_usd_brl_{alvo_chave}.png"
        plt.savefig(OUTPUT_DIR / nome_img, dpi=300)
        plt.close()
        
        print(f"Grafico comparativo limpo salvo em: {OUTPUT_DIR / nome_img}")
        
    # Salvar tabela de comparacao
    df_sintese = pd.DataFrame(registros_comparacao)
    df_sintese.to_csv(OUTPUT_DIR / "tabela_sintese_brl_usd.csv", index=False)
    
    print("\n" + "=" * 100)
    print(" TABELA SÍNTESE - RESULTADO COMPARATIVO NO HORIZONTE DE 12 MESES")
    print("=" * 100)
    print(df_sintese.to_string(index=False))
    print("=" * 100)
    print(f"Resultados e graficos salvos com sucesso na pasta: {OUTPUT_DIR.resolve()}")
    print("=" * 100)

if __name__ == "__main__":
    main()
