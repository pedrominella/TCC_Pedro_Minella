import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

outdir = Path("output_lp_baseline_graficos")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_excel('IPCA.xlsx')
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values('Data').set_index('Data')
df = df[df.index >= pd.Timestamp('2003-01-01')].copy()

controls = ['Cambio', 'Atividade', 'Selic', 'Expectativa_inflacao']
for c in controls:
    s = pd.to_numeric(df[c], errors='coerce')
    s = s.where(s > 0)
    df['LN_'+c] = np.log(s)

df['dln_cambio'] = df['LN_Cambio'].diff()
df['dln_atividade'] = df['LN_Atividade'].diff()
df['selic'] = df['LN_Selic']
df['expectativa'] = df['LN_Expectativa_inflacao'].diff()

df['ipca_geral'] = df['Var_IPCA_Geral']
df['ipca_transporte'] = df['Var_IPCA_Trans']
df['Var_Gasolina'] = pd.to_numeric(df['Var_Gasolina'], errors='coerce')
df['Var_Oleo_diesel'] = pd.to_numeric(df['Var_Oleo_diesel'], errors='coerce')

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

def plot_lp(df, y, shock, controls_cols, title, filename):
    horizons = 12
    coefs = np.zeros(horizons+1)
    se = np.zeros(horizons+1)
    
    reg_base = criar_lags(df, y, 3, y) + criar_lags(df, shock, 3, shock)
    for c in controls_cols:
        reg_base += [c] + criar_lags(df, c, 3, c)
        
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
        coefs[h] = mod.params[shock]
        se[h] = mod.bse[shock]
        
    ci_up = coefs + 1.96 * se
    ci_dn = coefs - 1.96 * se
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(horizons+1), coefs, 'b-', linewidth=2, label='Efeito Acumulado (LP)')
    ax.fill_between(range(horizons+1), ci_dn, ci_up, color='b', alpha=0.2, label='95% IC')
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_title(title)
    ax.set_xlabel('Meses após o choque')
    ax.set_ylabel('Resposta Acumulada (%)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)

ctrls = ['dln_cambio', 'dln_atividade', 'selic', 'expectativa']

plot_lp(df, 'ipca_transporte', 'Var_Gasolina', ctrls, 'Efeito da Gasolina no IPCA Transporte (LP)', 'lp_gasolina_transporte.png')
plot_lp(df, 'ipca_geral', 'Var_Gasolina', ctrls, 'Efeito da Gasolina no IPCA Geral (LP)', 'lp_gasolina_geral.png')
plot_lp(df, 'ipca_transporte', 'Var_Oleo_diesel', ctrls, 'Efeito do Óleo Diesel no IPCA Transporte (LP)', 'lp_diesel_transporte.png')
plot_lp(df, 'ipca_geral', 'Var_Oleo_diesel', ctrls, 'Efeito do Óleo Diesel no IPCA Geral (LP)', 'lp_diesel_geral.png')

print("Gráficos LP gerados com sucesso.")
