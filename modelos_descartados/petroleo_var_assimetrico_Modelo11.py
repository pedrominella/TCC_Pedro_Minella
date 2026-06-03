import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
import statsmodels.api as sm

# =========================
# CONFIG
# =========================
FILE_PATH = r"IPCA.xlsx"
STRINGENCY_FILE = r"Stringency_index.csv"
SHEET_NAME = "Sheet1"
DATA_INICIO = "2003-01-01"
MAXLAGS = 12
HORIZONTE_IRF = 24
OUTDIR_BASE = Path("output_petroleo_var_assimetrico_Modelo11")
SELIC_DAILY_FILE = r"STP-20260429165342557.csv"

SUBAMOSTRAS = [
    ("2003_2026", "2003-01-01", "2026-12-01"),
]

# =========================
# FUNÇÕES DE CARREGAMENTO
# =========================
def preparar_pastas(outdir):
    (outdir / "graficos" / "05_irf").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "07_irf_acumulada").mkdir(parents=True, exist_ok=True)
    (outdir / "tabelas").mkdir(parents=True, exist_ok=True)
    (outdir / "modelos").mkdir(parents=True, exist_ok=True)

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def candidate_col(df, preferred, fallback=None):
    if preferred in df.columns:
        return preferred
    if fallback is not None and fallback in df.columns:
        return fallback
    raise KeyError(f"Não encontrei '{preferred}' nem '{fallback}'.")

def safe_log(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)
    return np.log(s)

def build_index_from_var(series_pct, base=100.0):
    s = pd.to_numeric(series_pct, errors="coerce").fillna(0.0) / 100.0
    idx = (1.0 + s).cumprod() * base
    return idx

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

def drop_constant_or_duplicate_columns(df, tol=1e-12):
    if df is None or df.empty: return df
    out = df.copy()
    nunique = out.nunique(dropna=False)
    out = out[nunique[nunique > 1].index.tolist()]
    if out.empty: return out
    var_ok = out.var(numeric_only=True).fillna(0.0)
    out = out[var_ok[var_ok > tol].index.tolist()]
    if out.empty: return out
    out = out.loc[:, ~out.T.duplicated()]
    return out

def escolher_maxlags_seguro(nobs, nvars, nexog, maxlags_desejado):
    limite = int((nobs - 5) / max(1, (nvars + nexog)))
    return max(1, min(maxlags_desejado, limite))

def carregar_preparar_base():
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    df["Data"] = pd.to_datetime(df["Data"])
    df = df.sort_values("Data").set_index("Data")
    df = df[df.index >= pd.Timestamp(DATA_INICIO)].copy()

    map_cols = {
        "ipca_geral_nivel": candidate_col(df, "IPCA_Geral_nivel", "IPCA_Geral"),
        "ipca_trans_nivel": candidate_col(df, "IPCA_Trans_nivel", "Var_IPCA_Trans"),
        "gasolinaA_nivel": candidate_col(df, "GasolinaABrasil_media_nivel", "GasolinaABrasil_media"),
        "gasolina_nivel": candidate_col(df, "Gasolina_nivel", "Var_Gasolina"),
        "etanol_nivel": candidate_col(df, "Etanol_nivel", "Var_Etanol"),
        "oleo_diesel_nivel": candidate_col(df, "Oleo_diesel_nivel", "Var_Oleo_diesel"),
        "cambio": "Cambio",
        "preco_barril": "Preco_Barril",
        "atividade": "Atividade",
        "expectativa_inflacao": "Expectativa_inflacao",
    }
    ensure_numeric(df, list(set(map_cols.values())))

    if map_cols["ipca_trans_nivel"] == "Var_IPCA_Trans":
        df["IPCA_Trans_nivel_fallback"] = build_index_from_var(df["Var_IPCA_Trans"])
        map_cols["ipca_trans_nivel"] = "IPCA_Trans_nivel_fallback"
    for map_key, new_name, old_name in [("gasolina_nivel", "Gasolina_nivel_fallback", "Var_Gasolina"), ("etanol_nivel", "Etanol_nivel_fallback", "Var_Etanol"), ("oleo_diesel_nivel", "Oleo_diesel_nivel_fallback", "Var_Oleo_diesel")]:
        if map_cols[map_key] == old_name:
            df[new_name] = build_index_from_var(df[old_name])
            map_cols[map_key] = new_name

    work = pd.DataFrame(index=df.index)
    work["IPCA_Geral_nivel"] = pd.to_numeric(df[map_cols["ipca_geral_nivel"]], errors="coerce")
    work["IPCA_Trans_nivel"] = pd.to_numeric(df[map_cols["ipca_trans_nivel"]], errors="coerce")
    work["GasolinaA_nivel"] = pd.to_numeric(df[map_cols["gasolinaA_nivel"]], errors="coerce")
    work["Gasolina_nivel"] = pd.to_numeric(df[map_cols["gasolina_nivel"]], errors="coerce")
    work["Etanol_nivel"] = pd.to_numeric(df[map_cols["etanol_nivel"]], errors="coerce")
    work["Oleo_diesel_nivel"] = pd.to_numeric(df[map_cols["oleo_diesel_nivel"]], errors="coerce")
    work["Cambio"] = pd.to_numeric(df[map_cols["cambio"]], errors="coerce")
    work["Preco_Barril"] = pd.to_numeric(df[map_cols["preco_barril"]], errors="coerce")
    work["Atividade"] = pd.to_numeric(df[map_cols["atividade"]], errors="coerce")
    work["Selic"] = load_selic_meta_mensal(SELIC_DAILY_FILE, work.index)["Selic"]
    work["Expectativa_Inflacao"] = pd.to_numeric(df[map_cols["expectativa_inflacao"]], errors="coerce")
    work["Stringency"] = load_stringency_monthly(STRINGENCY_FILE, work.index)["Stringency"]

    base_vars = ["Preco_Barril", "Cambio", "GasolinaA_nivel", "Gasolina_nivel", "Etanol_nivel", "Oleo_diesel_nivel", "Atividade", "Expectativa_Inflacao", "Selic", "IPCA_Trans_nivel", "IPCA_Geral_nivel"]
    for c in base_vars:
        work[f"LN_{c}"] = safe_log(work[c])
        work[f"DLN_{c}"] = work[f"LN_{c}"].diff()

    # CRIAR VARIÁVEIS ASSIMÉTRICAS
    work["DLN_Preco_Barril_Pos"] = work["DLN_Preco_Barril"].clip(lower=0)
    work["DLN_Preco_Barril_Neg"] = work["DLN_Preco_Barril"].clip(upper=0).abs() # Transforma queda em valor positivo pra facilitar interpretação da magnitude ou deixa negativo mesmo
    work["DLN_Preco_Barril_Neg_Real"] = work["DLN_Preco_Barril"].clip(upper=0) 

    return work

def irf_orth_array(res, horizonte):
    irf = res.irf(horizonte)
    return irf, irf.orth_irfs

def salvar_irf_acumulada(irf_array, endog_cols, nome, outdir):
    response = 'DLN_IPCA_Resposta'
    impulses = ['DLN_Preco_Barril_Pos', 'DLN_Preco_Barril_Neg_Real', 'DLN_Combustivel']
    if response not in endog_cols: return
    resp_idx = endog_cols.index(response)
    horizontes = np.arange(irf_array.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    for imp in impulses:
        if imp not in endog_cols: continue
        imp_idx = endog_cols.index(imp)
        acumulada = np.cumsum(irf_array[:, resp_idx, imp_idx])
        ax.plot(horizontes, acumulada, marker='o', linewidth=1.5, label=imp)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title(f'{nome} - resposta acumulada do IPCA')
    ax.set_xlabel('Horizonte mensal')
    ax.set_ylabel('Resposta acumulada')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'graficos' / '07_irf_acumulada' / f'irf_acumulada_{nome}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

def rodar_modelo(work, nome_sub, data_ini, data_fim):
    outdir = OUTDIR_BASE / nome_sub
    preparar_pastas(outdir)

    mask = (work.index >= pd.Timestamp(data_ini)) & (work.index <= pd.Timestamp(data_fim))
    work_sub = work.loc[mask].copy()

    exog = pd.get_dummies(work_sub.index.month, prefix="m", drop_first=True)
    exog.index = work_sub.index
    exog = pd.concat([exog, work_sub[["Stringency"]]], axis=1).astype(float)
    exog = drop_constant_or_duplicate_columns(exog)

    comb_map = {"Gasolina": "Gasolina_nivel", "Oleo_diesel": "Oleo_diesel_nivel"}
    resp_map = {"IPCA_Brasil": "IPCA_Geral_nivel", "IPCA_Transporte": "IPCA_Trans_nivel"}

    sumarios = []

    for comb_name, comb_col in comb_map.items():
        for resp_name, resp_col in resp_map.items():
            nome = f"{comb_name}_{resp_name}"
            
            endog_cols_names = [
                "DLN_Preco_Barril_Pos",
                "DLN_Preco_Barril_Neg_Real",
                "DLN_Cambio",
                "DLN_Combustivel",
                "DLN_Atividade",
                "DLN_Expectativa_Inflacao",
                "LN_Selic",
                "DLN_IPCA_Resposta"
            ]
            
            model_df = pd.concat([
                work_sub["DLN_Preco_Barril_Pos"],
                work_sub["DLN_Preco_Barril_Neg_Real"],
                work_sub["DLN_Cambio"],
                work_sub[f"DLN_{comb_col}"],
                work_sub["DLN_Atividade"],
                work_sub["DLN_Expectativa_Inflacao"],
                work_sub["LN_Selic"],
                work_sub[f"DLN_{resp_col}"],
                exog
            ], axis=1).dropna()

            endog = model_df.iloc[:, :8].copy()
            endog.columns = endog_cols_names
            X = model_df.iloc[:, 8:].copy()
            X = drop_constant_or_duplicate_columns(X)

            try:
                maxlags_efetivo = escolher_maxlags_seguro(len(endog), endog.shape[1], X.shape[1] if X is not None else 0, MAXLAGS)
                varsel = VAR(endog, exog=X if not X.empty else None)
                sel = varsel.select_order(maxlags_efetivo)
                lag = max(1, int(sel.selected_orders.get("aic", 1)))
                res = varsel.fit(lag)
                
                with open(outdir / "modelos" / f"resumo_{nome}.txt", "w", encoding="utf-8") as f:
                    f.write(str(res.summary()))

                irf, irf_orth = irf_orth_array(res, HORIZONTE_IRF)
                for imp in ["DLN_Preco_Barril_Pos", "DLN_Preco_Barril_Neg_Real"]:
                    fig = irf.plot(impulse=imp, response="DLN_IPCA_Resposta", orth=True)
                    plt.tight_layout()
                    plt.savefig(outdir / "graficos" / "05_irf" / f"irf_{nome}_{imp}.png", dpi=180, bbox_inches="tight")
                    plt.close()
                salvar_irf_acumulada(irf_orth, list(endog.columns), nome, outdir)
                
                sumarios.append({"modelo": nome, "lag": lag, "nobs": res.nobs})
            except Exception as e:
                print(f"Erro em {nome}: {e}")

    pd.DataFrame(sumarios).to_excel(outdir / "tabelas" / "sumario.xlsx", index=False)

if __name__ == "__main__":
    OUTDIR_BASE.mkdir(parents=True, exist_ok=True)
    work = carregar_preparar_base()
    for nome_sub, data_ini, data_fim in SUBAMOSTRAS:
        rodar_modelo(work, nome_sub, data_ini, data_fim)
    print("VAR Assimétrico concluído.")
