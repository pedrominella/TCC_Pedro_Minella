# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Load IPCA baseline data
df_base = pd.read_excel('IPCA.xlsx')
df_base['Data'] = pd.to_datetime(df_base['Data'])
df_base = df_base.sort_values('Data').set_index('Data')
df_base = df_base[df_base.index >= pd.Timestamp('2003-01-01')].copy()

controls = ['Cambio', 'Atividade', 'Selic', 'Expectativa_inflacao']
for c in controls:
    s = pd.to_numeric(df_base[c], errors='coerce')
    s = s.where(s > 0)
    df_base['LN_'+c] = np.log(s)

df_base['dln_cambio'] = df_base['LN_Cambio'].diff()
df_base['dln_atividade'] = df_base['LN_Atividade'].diff()
df_base['selic'] = df_base['LN_Selic']
df_base['expectativa'] = df_base['LN_Expectativa_inflacao'].diff()

df_base['ipca_geral'] = df_base['Var_IPCA_Geral']
df_base['ipca_transporte'] = df_base['Var_IPCA_Trans']
df_base['Var_Gasolina'] = pd.to_numeric(df_base['Var_Gasolina'], errors='coerce')
df_base['Var_Oleo_diesel'] = pd.to_numeric(df_base['Var_Oleo_diesel'], errors='coerce')

def criar_lags(data, col, lags, prefix):
    cols = []
    for i in range(1, lags + 1):
        name = f'{prefix}_L{i}'
        data[name] = data[col].shift(i)
        cols.append(name)
    return cols

def montar_y_h(data, y_name, h):
    y_accum = data[y_name].rolling(window=h+1).sum().shift(-h)
    data[f'y_h{h}'] = y_accum
    return data

def estimar_baseline(df, y, shock, controls_cols):
    horizons = 12
    coefs = np.zeros(horizons+1)
    se = np.zeros(horizons+1)
    pvalues = np.zeros(horizons+1)
    
    reg_base = criar_lags(df, y, 3, y) + criar_lags(df, shock, 3, shock)
    for c in controls_cols:
        reg_base += [c] + criar_lags(df, c, 3, c)
        
    resultados = []
    for h in range(horizons+1):
        temp = df.copy()
        temp = montar_y_h(temp, y, h)
        X_cols = [shock] + reg_base
        for m in range(2, 13):
            temp[f'm_{m}'] = (temp.index.month == m).astype(float)
            X_cols.append(f'm_{m}')
            
        temp_reg = temp[[f'y_h{h}'] + X_cols].dropna()
        Y = temp_reg[f'y_h{h}']
        X = sm.add_constant(temp_reg[X_cols])
        mod = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(1,h)})
        
        coef = mod.params[shock]
        std_err = mod.bse[shock]
        pval = mod.pvalues[shock]
        
        resultados.append({
            'h': h,
            'coef': coef,
            'se': std_err,
            'pvalor': pval,
            'ci_low_90': coef - 1.645 * std_err,
            'ci_high_90': coef + 1.645 * std_err,
            'ci_low_95': coef - 1.96 * std_err,
            'ci_high_95': coef + 1.96 * std_err
        })
    return pd.DataFrame(resultados)

ctrls = ['dln_cambio', 'dln_atividade', 'selic', 'expectativa']
df_res_gas = estimar_baseline(df_base, 'ipca_transporte', 'Var_Gasolina', ctrls)
df_res_gas.to_csv('output_lp_baseline_graficos/tabela_baseline_gasolina_transporte.csv', index=False)

df_res_diesel = estimar_baseline(df_base, 'ipca_transporte', 'Var_Oleo_diesel', ctrls)
df_res_diesel.to_csv('output_lp_baseline_graficos/tabela_baseline_diesel_transporte.csv', index=False)

print("Tabelas baseline geradas com sucesso!")
