import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
FILE_PATH = r"IPCA.xlsx"
STRINGENCY_FILE = r"Stringency_index.csv"
SHEET_NAME = "Sheet1"
DATA_INICIO = "2003-01-01"
MAXLAGS = 6 # BVAR consegue lidar com mais lags, mas 6 é seguro
HORIZONTE_IRF = 24
OUTDIR_BASE = Path("output_petroleo_bvar_minnesota_Modelo13")
SELIC_DAILY_FILE = r"STP-20260429165342557.csv"
ALPHA_RIDGE = 10.0 # Parâmetro de penalização (prior tightness)

SUBAMOSTRAS = [
    ("2003_2026", "2003-01-01", "2026-12-01"),
]

def preparar_pastas(outdir):
    (outdir / "graficos" / "05_irf").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "07_irf_acumulada").mkdir(parents=True, exist_ok=True)

def safe_log(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)
    return np.log(s)

def load_stringency_monthly(filepath, target_index):
    try:
        s = pd.read_csv(filepath, sep=";", low_memory=False)
        s = s.rename(columns={"#country": "CountryName", "#country+code": "CountryCode", "#date": "Date"})
        possible_cols = ["StringencyIndex_Average_ForDisplay", "StringencyIndex_Average", "StringencyIndex"]
        str_col = next((c for c in possible_cols if c in s.columns), None)
        s = s[s["CountryCode"].astype(str).str.upper() == "BRA"].copy()
        s["Date"] = pd.to_datetime(s["Date"].astype(str), format="%Y%m%d", errors="coerce")
        s[str_col] = pd.to_numeric(s[str_col], errors="coerce")
        s = s.dropna(subset=["Date"]).sort_values("Date")
        s = s[["Date", str_col]].rename(columns={str_col: "Stringency"})
        s["Data_mes"] = s["Date"].dt.to_period("M").dt.to_timestamp()
        s_m = s.groupby("Data_mes", as_index=True)["Stringency"].mean().to_frame()
        s_m = s_m.reindex(target_index)
        s_m["Stringency"] = s_m["Stringency"].fillna(0.0)
        return s_m
    except:
        return pd.DataFrame(0.0, index=target_index, columns=["Stringency"])

def load_selic_meta_mensal(filepath, target_index):
    try:
        candidatos = [Path(filepath)] + sorted(Path('.').glob('STP*.csv'))
        arquivo_selic = next((c for c in candidatos if c.exists()), None)
        if not arquivo_selic: return pd.DataFrame(np.nan, index=target_index, columns=["Selic"])
        s = pd.read_csv(arquivo_selic, sep=';', low_memory=False)
        s.columns = [str(c).strip() for c in s.columns]
        data_col = next((c for c in s.columns if c.lower() in ['data', 'date']), s.columns[0])
        valor_col = next((c for c in s.columns if c != data_col and ('selic' in c.lower() or '432' in c.lower())), [c for c in s.columns if c != data_col][0])
        s[data_col] = pd.to_datetime(s[data_col], dayfirst=True, errors='coerce')
        s[valor_col] = s[valor_col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        s[valor_col] = pd.to_numeric(s[valor_col], errors='coerce')
        s = s.dropna(subset=[data_col, valor_col]).sort_values(data_col)
        s['Data_mes'] = s[data_col].dt.to_period('M').dt.to_timestamp()
        selic_m = s.groupby('Data_mes', as_index=True)[valor_col].mean().to_frame('Selic')
        selic_m = selic_m.reindex(target_index)
        selic_m['Selic'] = selic_m['Selic'].interpolate(method='time').ffill().bfill()
        return selic_m
    except:
        return pd.DataFrame(np.nan, index=target_index, columns=["Selic"])

def carregar_preparar_base():
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    df["Data"] = pd.to_datetime(df["Data"])
    df = df.sort_values("Data").set_index("Data")
    df = df[df.index >= pd.Timestamp(DATA_INICIO)].copy()

    map_cols = {
        "ipca_geral_nivel": "IPCA_Geral_nivel" if "IPCA_Geral_nivel" in df.columns else "IPCA_Geral",
        "ipca_trans_nivel": "IPCA_Trans_nivel" if "IPCA_Trans_nivel" in df.columns else "Var_IPCA_Trans",
        "gasolina_nivel": "Gasolina_nivel" if "Gasolina_nivel" in df.columns else "Var_Gasolina",
        "oleo_diesel_nivel": "Oleo_diesel_nivel" if "Oleo_diesel_nivel" in df.columns else "Var_Oleo_diesel",
        "cambio": "Cambio",
        "preco_barril": "Preco_Barril",
        "atividade": "Atividade",
        "expectativa_inflacao": "Expectativa_inflacao",
    }
    work = pd.DataFrame(index=df.index)
    for k, v in map_cols.items():
        if k not in ["expectativa_inflacao", "atividade", "cambio", "preco_barril"]:
            # Só pra simplificar neste script, assumimos que já estão nivelados.
            pass
    work["IPCA_Geral_nivel"] = pd.to_numeric(df[map_cols["ipca_geral_nivel"]], errors="coerce")
    work["IPCA_Trans_nivel"] = pd.to_numeric(df[map_cols["ipca_trans_nivel"]], errors="coerce")
    work["Gasolina_nivel"] = pd.to_numeric(df[map_cols["gasolina_nivel"]], errors="coerce")
    work["Oleo_diesel_nivel"] = pd.to_numeric(df[map_cols["oleo_diesel_nivel"]], errors="coerce")
    work["Cambio"] = pd.to_numeric(df[map_cols["cambio"]], errors="coerce")
    work["Preco_Barril"] = pd.to_numeric(df[map_cols["preco_barril"]], errors="coerce")
    work["Atividade"] = pd.to_numeric(df[map_cols["atividade"]], errors="coerce")
    work["Selic"] = load_selic_meta_mensal(SELIC_DAILY_FILE, work.index)["Selic"]
    work["Expectativa_Inflacao"] = pd.to_numeric(df[map_cols["expectativa_inflacao"]], errors="coerce")
    
    base_vars = ["Preco_Barril", "Cambio", "Gasolina_nivel", "Oleo_diesel_nivel", "Atividade", "Expectativa_Inflacao", "Selic", "IPCA_Trans_nivel", "IPCA_Geral_nivel"]
    for c in base_vars:
        work[f"LN_{c}"] = safe_log(work[c])
        work[f"DLN_{c}"] = work[f"LN_{c}"].diff()
        
    return work

def fit_bvar_ridge(Y, p, alpha):
    T, K = Y.shape
    cols = Y.columns
    
    X_lags = []
    for lag in range(1, p+1):
        x_lag = Y.shift(lag)
        x_lag.columns = [f"{c}_L{lag}" for c in cols]
        # Minnesota scaling: divide by lag
        x_lag = x_lag / lag 
        X_lags.append(x_lag)
        
    X = pd.concat(X_lags, axis=1)
    df = pd.concat([Y, X], axis=1).dropna()
    Y_train = df[cols].values
    X_train = df[X.columns].values
    
    # Add intercept
    X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    
    # Ridge formula
    # beta = (X'X + alpha*I)^{-1} X'Y
    # Penalize all except intercept
    I_pen = np.eye(X_train_aug.shape[1])
    I_pen[0, 0] = 0 # do not penalize intercept
    
    beta = np.linalg.inv(X_train_aug.T @ X_train_aug + alpha * I_pen) @ X_train_aug.T @ Y_train
    coefs = beta[1:, :].T # shape (K, Kp)
    intercept = beta[0, :]
    
    idx = 0
    for lag in range(1, p+1):
        coefs[:, idx:idx+K] = coefs[:, idx:idx+K] / lag
        idx += K
        
    # Reconstruct original X without scaling to calculate residuals
    X_unscaled_lags = []
    for lag in range(1, p+1):
        X_unscaled_lags.append(Y.shift(lag))
    X_unscaled = pd.concat(X_unscaled_lags, axis=1).dropna().values
    
    # predict
    Y_pred = X_unscaled @ coefs.T + intercept
    resid = Y_train - Y_pred
    Sigma = np.cov(resid, rowvar=False)
    
    return coefs, Sigma

def compute_irfs(coefs, Sigma, K, p, horizons):
    A = np.zeros((K*p, K*p))
    A[:K, :] = coefs
    if p > 1:
        A[K:, :-K] = np.eye(K*(p-1))
        
    P = np.linalg.cholesky(Sigma)
    irfs = np.zeros((horizons+1, K, K))
    irfs[0] = P
    
    A_pow = np.eye(K*p)
    for h in range(1, horizons+1):
        A_pow = A @ A_pow
        irfs[h] = A_pow[:K, :K] @ P
        
    return irfs

def rodar_modelo(work, nome_sub, data_ini, data_fim):
    outdir = OUTDIR_BASE / nome_sub
    preparar_pastas(outdir)

    mask = (work.index >= pd.Timestamp(data_ini)) & (work.index <= pd.Timestamp(data_fim))
    work_sub = work.loc[mask].copy()

    comb_map = {"Gasolina": "Gasolina_nivel", "Oleo_diesel": "Oleo_diesel_nivel"}
    resp_map = {"IPCA_Brasil": "IPCA_Geral_nivel", "IPCA_Transporte": "IPCA_Trans_nivel"}

    for comb_name, comb_col in comb_map.items():
        for resp_name, resp_col in resp_map.items():
            nome = f"{comb_name}_{resp_name}"
            
            endog_cols_names = [
                "DLN_Preco_Barril", "DLN_Cambio", "DLN_Combustivel", "DLN_Atividade",
                "DLN_Expectativa_Inflacao", "LN_Selic", "DLN_IPCA_Resposta"
            ]
            
            model_df = pd.concat([
                work_sub["DLN_Preco_Barril"], work_sub["DLN_Cambio"], work_sub[f"DLN_{comb_col}"],
                work_sub["DLN_Atividade"], work_sub["DLN_Expectativa_Inflacao"],
                work_sub["LN_Selic"], work_sub[f"DLN_{resp_col}"]
            ], axis=1).dropna()

            endog = model_df.iloc[:, :7].copy()
            endog.columns = endog_cols_names
            
            p = MAXLAGS
            K = endog.shape[1]
            coefs, Sigma = fit_bvar_ridge(endog, p, alpha=ALPHA_RIDGE)
            irfs = compute_irfs(coefs, Sigma, K, p, HORIZONTE_IRF)
            
            # Plot
            resp_idx = endog_cols_names.index("DLN_IPCA_Resposta")
            horizontes = np.arange(HORIZONTE_IRF + 1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            for imp in ["DLN_Preco_Barril", "DLN_Cambio", "DLN_Combustivel"]:
                imp_idx = endog_cols_names.index(imp)
                acumulada = np.cumsum(irfs[:, resp_idx, imp_idx])
                ax.plot(horizontes, acumulada, marker='o', linewidth=1.5, label=imp)
                
            ax.axhline(0, color='black', lw=0.8)
            ax.set_title(f'BVAR (Ridge) {nome} - Resposta acumulada do IPCA')
            ax.set_xlabel('Horizonte mensal')
            ax.set_ylabel('Resposta acumulada')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / 'graficos' / '07_irf_acumulada' / f'bvar_acumulada_{nome}.png', dpi=180, bbox_inches='tight')
            plt.close(fig)

if __name__ == "__main__":
    OUTDIR_BASE.mkdir(parents=True, exist_ok=True)
    work = carregar_preparar_base()
    for nome_sub, data_ini, data_fim in SUBAMOSTRAS:
        rodar_modelo(work, nome_sub, data_ini, data_fim)
    print("BVAR Minnesota (Ridge approximation) concluído.")
