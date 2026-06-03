# -*- coding: utf-8 -*-
"""
gerar_graficos_ols_limpos.py

Objetivo:
---------
Este script lê a base de dados já preparada pelo Modelo 10 (base_modelo10_kilian.xlsx)
e gera gráficos de Impulso-Resposta (IRF) 100% LIMPOS e exclusivos para o modelo
**LP-OLS com o Índice Kilian como Controle**.

Diferente dos gráficos anteriores:
- Remove totalmente a curva LP-IV (azul).
- Remove totalmente qualquer menção à Estatística-F do primeiro estágio.
- Mostra apenas a curva OLS com sua respectiva banda de confiança de 90% (HAC).
- Ideal para inclusão direta no texto do TCC.
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# Configurações
BASE_EXCEL = Path("output_petroleo_lp_modelo10_kilian") / "base_modelo10_kilian.xlsx"
OUTPUT_DIR = Path("output_petroleo_lp_modelo10_kilian") / "graficos_ols_limpos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H_MAX = 12
LAGS_Y = 3
LAGS_SHOCK = 3
LAGS_CONTROLS = 3
CONF = 0.90
Z_CRIT = 1.645
USAR_HAC = True
MIN_OBS = 60

# Variáveis mapeadas
VARS_T = {
    "petroleo_brl": "dln_petroleo_brl_std",
    "kilian": "kilian_std",
    "cambio": "dln_cambio",
    "atividade": "dln_atividade",
    "selic": "selic_controle",
    "expectativa": "expectativa_controle",
    "regime_petrobras": "regime_petrobras_pos_set2016",
    
    # Alvos
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

def estimar_lp_ols_controle(base, y_name, shock, kilian_index, controls, h_max=12):
    resultados = []
    regressores_fixos = []
    
    # Lags do Y e do Shock
    regressores_fixos += criar_lags(base, y_name, LAGS_Y, prefix=y_name)
    regressores_fixos += criar_lags(base, shock, LAGS_SHOCK, prefix=shock)
    
    # Índice Kilian como controle em nível e lags
    regressores_fixos.append(kilian_index)
    regressores_fixos += criar_lags(base, kilian_index, LAGS_CONTROLS, prefix=kilian_index)
    
    # Outros controles
    for c in controls:
        if c in base.columns and c != kilian_index:
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
    print("GERANDO GRÁFICOS OLS LIMPOS (CONTROLE DO ÍNDICE KILIAN)")
    print("=" * 100)
    
    if not BASE_EXCEL.exists():
        raise FileNotFoundError(f"Não encontrei a base preparada em: {BASE_EXCEL}")
        
    base = pd.read_excel(BASE_EXCEL)
    
    # Identificar colunas de dummies mensais
    dummy_cols = [c for c in base.columns if c.startswith("mes_")]
    
    # Controles principais
    controles = []
    for k in ["cambio", "atividade", "selic", "expectativa", "regime_petrobras"]:
        if k in VARS_T and VARS_T[k] in base.columns:
            controles.append(VARS_T[k])
    controles = controles + dummy_cols
    
    shock = VARS_T["petroleo_brl"]
    kilian_index = VARS_T["kilian"]
    
    for alvo_chave, alvo_label in ALVOS.items():
        if alvo_chave not in VARS_T:
            continue
            
        y_name = VARS_T[alvo_chave]
        if y_name not in base.columns:
            print(f"Aviso: {y_name} não encontrado na base. Pulando.")
            continue
            
        print(f"Estimando e desenhando OLS limpo para: {alvo_label}...")
        
        df_ols = estimar_lp_ols_controle(base, y_name, shock, kilian_index, controles, h_max=H_MAX)
        
        # Salvar tabela limpa em CSV
        df_ols.to_csv(OUTPUT_DIR / f"tabela_LP_OLS_limpa_{alvo_chave}.csv", index=False)
        
        # Gerar gráfico 100% LIMPO
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        # Curva OLS (cor premium: Deep Orange/Rust ou Navy Blue)
        # Vamos usar um azul royal profissional para a curva e preenchimento cinza claro para a faixa
        ax.plot(df_ols["h"], df_ols["coef"], marker="o", color="#1a73e8", linewidth=2.5, label="Impacto do Petróleo em R$ (Local Projections OLS)")
        ax.fill_between(df_ols["h"], df_ols["ci_low"], df_ols["ci_high"], color="#1a73e8", alpha=0.15, label="Intervalo de Confiança de 90% (HAC)")
        
        # Linhas de referência
        ax.axhline(0, color="#202124", linewidth=1.0, linestyle="-")
        
        # Formatação Premium
        ax.set_title(f"Resposta Acumulada: Petróleo em Reais $\\rightarrow$ {alvo_label}\n(Modelo LP-OLS - Controlando pelo Índice Kilian)", fontsize=11.5, fontweight="bold", pad=15)
        ax.set_xlabel("Horizonte de Projeção (h, em meses)", fontsize=10)
        ax.set_ylabel("Impacto Acumulado (%)", fontsize=10)
        ax.set_xticks(range(0, H_MAX + 1))
        ax.grid(True, linestyle=":", alpha=0.5, color="#dadce0")
        ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#dadce0")
        
        # Remover bordas desnecessárias (spines)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        nome_img = f"LP_OLS_Kilian_limpo_{alvo_chave}.png"
        plt.savefig(OUTPUT_DIR / nome_img, dpi=300)
        plt.close()
        
        print(f"Gráfico limpo salvo em: {OUTPUT_DIR / nome_img}")
        
    print("\n" + "=" * 100)
    print("TODOS OS GRÁFICOS OLS LIMPOS FORAM GERADOS COM SUCESSO!")
    print(f"Salvos na pasta: {OUTPUT_DIR.resolve()}")
    print("=" * 100)

if __name__ == "__main__":
    main()
