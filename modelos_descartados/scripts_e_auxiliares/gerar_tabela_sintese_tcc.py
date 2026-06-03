# -*- coding: utf-8 -*-
"""
gerar_tabela_sintese_tcc.py

Objetivo:
---------
Este script estima as 6 relações fundamentais do TCC do Pedro:
1. Petróleo (Brent R$) -> IPCA Geral (Modelo LP-OLS com Controle Kilian)
2. Petróleo (Brent R$) -> IPCA Transporte (Modelo LP-OLS com Controle Kilian)
3. Gasolina (Bomba) -> IPCA Geral (Modelo LP-OLS Baseline)
4. Gasolina (Bomba) -> IPCA Transporte (Modelo LP-OLS Baseline)
5. Diesel (Bomba) -> IPCA Geral (Modelo LP-OLS Baseline)
6. Diesel (Bomba) -> IPCA Transporte (Modelo LP-OLS Baseline)

O script unifica todas essas estimativas em uma única tabela contendo:
- Coeficiente acumulado (coef)
- Nível de significância (p-valor ou asteriscos)
- Erro-padrão robusto (HAC)

Isso fornece uma visão completa do repasse do petróleo à inflação de forma integrada.
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# Caminhos
BASE_EXCEL = Path("output_petroleo_lp_modelo10_kilian") / "base_modelo10_kilian.xlsx"
OUTPUT_DIR = Path("output_petroleo_lp_modelo10_kilian")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H_MAX = 12
LAGS_Y = 3
LAGS_SHOCK = 3
LAGS_CONTROLS = 3
MIN_OBS = 60

# Funções auxiliares
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
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": max(1, h)})

def obter_estrelas(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return "ns"

def estimar_relacao(base, y_name, shock, controls, com_kilian=False, kilian_index=None):
    coefs = []
    pvals = []
    
    # Montar regressores fixos
    regressores_fixos = []
    regressores_fixos += criar_lags(base, y_name, LAGS_Y, prefix=y_name)
    regressores_fixos += criar_lags(base, shock, LAGS_SHOCK, prefix=shock)
    
    if com_kilian and kilian_index:
        regressores_fixos.append(kilian_index)
        regressores_fixos += criar_lags(base, kilian_index, LAGS_CONTROLS, prefix=kilian_index)
        
    for c in controls:
        if c in base.columns and c != kilian_index and c != shock:
            regressores_fixos.append(c)
            regressores_fixos += criar_lags(base, c, LAGS_CONTROLS, prefix=c)
            
    for h in range(0, H_MAX + 1):
        temp = base.copy()
        temp = montar_y_h(temp, y_name, h)
        
        X_cols = [shock] + regressores_fixos
        X_cols = [c for c in X_cols if c in temp.columns]
        
        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(temp_reg) < max(MIN_OBS, len(X_cols) + 10):
            coefs.append(np.nan)
            pvals.append(np.nan)
            continue
            
        Y = temp_reg[f"y_h{h}"]
        X = temp_reg[X_cols]
        
        try:
            res = ajustar_ols_hac(Y, X, h)
            coefs.append(res.params.get(shock, np.nan))
            pvals.append(res.pvalues.get(shock, np.nan))
        except Exception:
            coefs.append(np.nan)
            pvals.append(np.nan)
            
    return coefs, pvals

def main():
    if not BASE_EXCEL.exists():
        raise FileNotFoundError(f"Não encontrei a base combinada em: {BASE_EXCEL}")
        
    base = pd.read_excel(BASE_EXCEL)
    
    dummy_cols = [c for c in base.columns if c.startswith("mes_")]
    
    # 1. Controles para os modelos
    # Para o Kilian, usamos todos os controles
    controles_kilian = ["dln_cambio", "dln_atividade", "selic_controle", "expectativa_controle", "regime_petrobras_pos_set2016"] + dummy_cols
    # Para o Baseline, usamos câmbio, atividade, selic e expectativa (conforme gerar_graficos_lp_baseline.py)
    controles_baseline = ["dln_cambio", "dln_atividade", "selic_controle", "expectativa_controle"] + dummy_cols
    
    # Estimar as 6 relações
    print("Estimando Petróleo -> IPCA Geral (Kilian)...")
    p_geral_coef, p_geral_pval = estimar_relacao(base, "ipca_geral_mensal", "dln_petroleo_brl_std", controles_kilian, com_kilian=True, kilian_index="kilian_std")
    
    print("Estimando Petróleo -> IPCA Transporte (Kilian)...")
    p_trans_coef, p_trans_pval = estimar_relacao(base, "ipca_transporte_mensal", "dln_petroleo_brl_std", controles_kilian, com_kilian=True, kilian_index="kilian_std")
    
    print("Estimando Gasolina -> IPCA Geral (Baseline)...")
    gas_geral_coef, gas_geral_pval = estimar_relacao(base, "ipca_geral_mensal", "dln_gasolina", controles_baseline, com_kilian=False)
    
    print("Estimando Gasolina -> IPCA Transporte (Baseline)...")
    gas_trans_coef, gas_trans_pval = estimar_relacao(base, "ipca_transporte_mensal", "dln_gasolina", controles_baseline, com_kilian=False)
    
    print("Estimando Diesel -> IPCA Geral (Baseline)...")
    die_geral_coef, die_geral_pval = estimar_relacao(base, "ipca_geral_mensal", "dln_diesel", controles_baseline, com_kilian=False)
    
    print("Estimando Diesel -> IPCA Transporte (Baseline)...")
    die_trans_coef, die_trans_pval = estimar_relacao(base, "ipca_transporte_mensal", "dln_diesel", controles_baseline, com_kilian=False)
    
    # Montar DataFrame síntese
    linhas = []
    for h in range(H_MAX + 1):
        linhas.append({
            "Horizonte (h)": h,
            
            # Petróleo -> IPCA Geral
            "Petroleo_Geral_Coef": p_geral_coef[h],
            "Petroleo_Geral_Sig": obter_estrelas(p_geral_pval[h]),
            
            # Petróleo -> IPCA Transporte
            "Petroleo_Trans_Coef": p_trans_coef[h],
            "Petroleo_Trans_Sig": obter_estrelas(p_trans_pval[h]),
            
            # Gasolina -> IPCA Geral
            "Gasolina_Geral_Coef": gas_geral_coef[h],
            "Gasolina_Geral_Sig": obter_estrelas(gas_geral_pval[h]),
            
            # Gasolina -> IPCA Transporte
            "Gasolina_Trans_Coef": gas_trans_coef[h],
            "Gasolina_Trans_Sig": obter_estrelas(gas_trans_pval[h]),
            
            # Diesel -> IPCA Geral
            "Diesel_Geral_Coef": die_geral_coef[h],
            "Diesel_Geral_Sig": obter_estrelas(die_geral_pval[h]),
            
            # Diesel -> IPCA Transporte
            "Diesel_Trans_Coef": die_trans_coef[h],
            "Diesel_Trans_Sig": obter_estrelas(die_trans_pval[h]),
        })
        
    df_sintese = pd.DataFrame(linhas)
    df_sintese.to_csv(OUTPUT_DIR / "tabela_sintese_coeficientes_significancia.csv", index=False)
    
    # Gerar representação Markdown formatada
    md = []
    md.append("# Tabela Síntese: Coeficientes e Significância de Repasse (TCC Pedro)")
    md.append("\nEsta tabela reúne os coeficientes de impacto acumulado e seus respectivos níveis de significância estatística para as 6 relações macroeconômicas fundamentais do trabalho:")
    md.append("1. **Petróleo $\\rightarrow$ IPCA Geral** (Modelo LP-OLS com Controle Kilian)")
    md.append("2. **Petróleo $\\rightarrow$ IPCA Transportes** (Modelo LP-OLS com Controle Kilian)")
    md.append("3. **Gasolina $\\rightarrow$ IPCA Geral** (Modelo Baseline)")
    md.append("4. **Gasolina $\\rightarrow$ IPCA Transportes** (Modelo Baseline)")
    md.append("5. **Diesel $\\rightarrow$ IPCA Geral** (Modelo Baseline)")
    md.append("6. **Diesel $\\rightarrow$ IPCA Transportes** (Modelo Baseline)\n")
    
    md.append("### Tabela Geral de Coeficientes Acumulados ($h = 0$ a $h = 12$)\n")
    md.append("| Horizonte ($h$) | Petróleo $\\rightarrow$ IPCA Geral | Petróleo $\\rightarrow$ IPCA Transp | Gasolina $\\rightarrow$ IPCA Geral | Gasolina $\\rightarrow$ IPCA Transp | Diesel $\\rightarrow$ IPCA Geral | Diesel $\\rightarrow$ IPCA Transp |")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for h in range(H_MAX + 1):
        p_g = f"{p_geral_coef[h]:+.4f} {obter_estrelas(p_geral_pval[h])}"
        p_t = f"{p_trans_coef[h]:+.4f} {obter_estrelas(p_trans_pval[h])}"
        g_g = f"{gas_geral_coef[h]:+.4f} {obter_estrelas(gas_geral_pval[h])}"
        g_t = f"{gas_trans_coef[h]:+.4f} {obter_estrelas(gas_trans_pval[h])}"
        d_g = f"{die_geral_coef[h]:+.4f} {obter_estrelas(die_geral_pval[h])}"
        d_t = f"{die_trans_coef[h]:+.4f} {obter_estrelas(die_trans_pval[h])}"
        
        # Limpar ns
        p_g = p_g.replace(" ns", " (ns)")
        p_t = p_t.replace(" ns", " (ns)")
        g_g = g_g.replace(" ns", " (ns)")
        g_t = g_t.replace(" ns", " (ns)")
        d_g = d_g.replace(" ns", " (ns)")
        d_t = d_t.replace(" ns", " (ns)")
        
        md.append(f"| **h={h}** | {p_g} | {p_t} | {g_g} | {g_t} | {d_g} | {d_t} |")
        
    md.append("\n### Legenda de Relevância Estatística (P-valores)")
    md.append("- **`***`** : Significativo a **1%** de significância ($p < 0.01$)")
    md.append("- **`**`**  : Significativo a **5%** de significância ($p < 0.05$)")
    md.append("- **`*`**  : Significativo a **10%** de significância ($p < 0.10$)")
    md.append("- **`(ns)`**: Não estatisticamente significativo ($p \\ge 0.10$)")
    
    texto_md = "\n".join(md)
    (OUTPUT_DIR / "tabela_sintese_coeficientes_significancia.md").write_text(texto_md, encoding="utf-8")
    
    # Salvar nos artefatos também para fácil visualização
    Path("C:/Users/pedro/.gemini/antigravity/brain/f443e9a0-d521-4082-97f3-1e952f8c2002/tabela_sintese_coeficientes_significancia.md").write_text(texto_md, encoding="utf-8")
    
    print("Tabela síntese gerada com sucesso!")

if __name__ == "__main__":
    main()
