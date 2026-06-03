# -*- coding: utf-8 -*-
"""
gerar_graficos_ols_dupla_faixa.py

Objetivo:
---------
Este script realiza a estimação do modelo LP-OLS controlando pelo Índice Kilian (IGREA)
e gera gráficos com DUPLA FAIXA DE CONFIANÇA (90% e 95%) para demonstrar de forma
100% transparente o comportamento dos coeficientes e das bandas de significância.
Remove qualquer resquício de teste F do primeiro estágio ou curvas LP-IV.
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
OUTPUT_DIR = Path("output_petroleo_lp_modelo10_kilian") / "graficos_ols_dupla_faixa"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parâmetros de projeção
H_MAX = 12
LAGS_Y = 3
LAGS_SHOCK = 3
LAGS_CONTROLS = 3
USAR_HAC = True
MIN_OBS = 60

# Mapeamento de variáveis
VARS_T = {
    "petroleo_brl": "dln_petroleo_brl_std",
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
    "gasolina_refinaria": "Gasolina na Refinaria (dlog %)",
    "gasolina": "Gasolina ao Consumidor (dlog %)",
    "diesel": "Diesel ao Consumidor (dlog %)",
    "etanol": "Etanol ao Consumidor (dlog %)",
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
    
    # Índice Kilian como controle principal (em nível)
    regressores_fixos.append(kilian_index)
    regressores_fixos += criar_lags(base, kilian_index, LAGS_CONTROLS, prefix=kilian_index)
    
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
                "ci_low_90": np.nan, "ci_high_90": np.nan,
                "ci_low_95": np.nan, "ci_high_95": np.nan
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
                "ci_low_90": coef - 1.645 * se,
                "ci_high_90": coef + 1.645 * se,
                "ci_low_95": coef - 1.96 * se,
                "ci_high_95": coef + 1.96 * se
            })
        except Exception:
            resultados.append({
                "h": h, "coef": np.nan, "se": np.nan, "t": np.nan, "pvalor": np.nan,
                "ci_low_90": np.nan, "ci_high_90": np.nan,
                "ci_low_95": np.nan, "ci_high_95": np.nan
            })
            
    return pd.DataFrame(resultados)

def main():
    print("=" * 100)
    print("GERANDO GRÁFICOS OLS COM DUPLA FAIXA DE CONFIANÇA (TRANSPARÊNCIA MÁXIMA)")
    print("=" * 100)
    
    if not BASE_EXCEL.exists():
        raise FileNotFoundError(f"Não encontrei a base em: {BASE_EXCEL}")
        
    base = pd.read_excel(BASE_EXCEL)
    
    dummy_cols = [c for c in base.columns if c.startswith("mes_")]
    
    controles = []
    for k in ["cambio", "atividade", "selic", "expectativa", "regime_petrobras"]:
        if k in VARS_T and VARS_T[k] in base.columns:
            controles.append(VARS_T[k])
    controles = controles + dummy_cols
    
    shock = VARS_T["petroleo_brl"]
    kilian_index = VARS_T["kilian"]
    
    for alvo_chave, alvo_label in ALVOS.items():
        y_name = VARS_T[alvo_chave]
        if y_name not in base.columns:
            continue
            
        print(f"Processando: {alvo_label}...")
        df_ols = estimar_lp_ols_controle(base, y_name, shock, kilian_index, controles, h_max=H_MAX)
        
        # Salvar tabela com dupla faixa em CSV
        df_ols.to_csv(OUTPUT_DIR / f"tabela_LP_OLS_dupla_faixa_{alvo_chave}.csv", index=False)
        
        # Gerar gráfico premium com dupla faixa
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # 1. Faixa externa: 95% Intervalo de Confiança (Cor azul royal bem clara / translúcida)
        ax.fill_between(df_ols["h"], df_ols["ci_low_95"], df_ols["ci_high_95"], 
                        color="#1a73e8", alpha=0.08, label="Intervalo de Confiança de 95% (Z = 1.96)")
        
        # 2. Faixa interna: 90% Intervalo de Confiança (Cor azul royal um pouco mais escura)
        ax.fill_between(df_ols["h"], df_ols["ci_low_90"], df_ols["ci_high_90"], 
                        color="#1a73e8", alpha=0.18, label="Intervalo de Confiança de 90% (Z = 1.645)")
        
        # 3. Linha do Coeficiente OLS
        ax.plot(df_ols["h"], df_ols["coef"], marker="o", color="#1557b0", linewidth=2.5, 
                label="Impacto Acumulado do Brent (LP-OLS)")
        
        # Linha horizontal no zero
        ax.axhline(0, color="#202124", linewidth=1.2, linestyle="-")
        
        # Estética do gráfico
        ax.set_title(f"Impacto do Preço do Petróleo (Reais) $\\rightarrow$ {alvo_label}\n(Modelo LP-OLS Limpo com Controle do Índice Kilian)", 
                     fontsize=11.5, fontweight="bold", pad=15)
        ax.set_xlabel("Horizonte de Projeção (h, em meses)", fontsize=10)
        ax.set_ylabel("Impacto Acumulado (%)", fontsize=10)
        ax.set_xticks(range(0, H_MAX + 1))
        ax.grid(True, linestyle=":", alpha=0.5, color="#dadce0")
        ax.legend(loc="upper left", fontsize=9.5, frameon=True, facecolor="white", edgecolor="#dadce0")
        
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
            
        plt.tight_layout()
        nome_img = f"LP_OLS_Kilian_dupla_faixa_{alvo_chave}.png"
        plt.savefig(OUTPUT_DIR / nome_img, dpi=300)
        plt.close()
        print(f"Gráfico com dupla faixa salvo em: {OUTPUT_DIR / nome_img}")
        
    print("\n" + "=" * 100)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"Tabelas e gráficos salvos em: {OUTPUT_DIR.resolve()}")
    print("=" * 100)

if __name__ == "__main__":
    main()
