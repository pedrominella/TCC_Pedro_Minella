# -*- coding: utf-8 -*-
"""
Versão ajustada do script VAR para usar, PRIORITARIAMENTE, as colunas em nível
com base 100 em dez/2002:

- IPCA_Geral_nivel
- IPCA_Trans_nivel
- GasolinaABrasil_media_nivel
- Gasolina_nivel
- Etanol_nivel
- Oleo_diesel_nivel

Se essas colunas não existirem no Excel, o script cai automaticamente para as
colunas antigas para não quebrar.
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
import statsmodels.api as sm

# =========================
# CONFIG
# =========================
FILE_PATH = r"IPCA.xlsx"
SHEET_NAME = "Sheet1"
DATA_INICIO = "2003-01-01"
MAXLAGS = 12
HORIZONTE_IRF = 24

OUTDIR = Path("output_tcc_var_nivel_eventos")
(OUTDIR / "graficos" / "01_nivel").mkdir(parents=True, exist_ok=True)
(OUTDIR / "graficos" / "02_ln").mkdir(parents=True, exist_ok=True)
(OUTDIR / "graficos" / "03_dln").mkdir(parents=True, exist_ok=True)
(OUTDIR / "graficos" / "04_residuos").mkdir(parents=True, exist_ok=True)
(OUTDIR / "graficos" / "05_irf").mkdir(parents=True, exist_ok=True)
(OUTDIR / "tabelas").mkdir(parents=True, exist_ok=True)
(OUTDIR / "modelos").mkdir(parents=True, exist_ok=True)


# =========================
# FUNÇÕES
# =========================
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
    raise KeyError(f"Não encontrei nem '{preferred}' nem fallback '{fallback}' no arquivo.")

def build_index_from_var(series_pct, base=100.0):
    s = pd.to_numeric(series_pct, errors="coerce").fillna(0.0) / 100.0
    idx = (1.0 + s).cumprod() * base
    return idx

def plot_series(series, title, outfile):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)

def adf_test(x):
    x = pd.Series(x).dropna()
    if len(x) < 20:
        return {"stat": np.nan, "pvalue": np.nan, "lags": np.nan, "nobs": len(x)}
    stat, pvalue, lags, nobs, *_ = adfuller(x, autolag="AIC")
    return {"stat": stat, "pvalue": pvalue, "lags": lags, "nobs": nobs}

def kpss_test(x):
    x = pd.Series(x).dropna()
    if len(x) < 20:
        return {"stat": np.nan, "pvalue": np.nan, "lags": np.nan, "nobs": len(x)}
    stat, pvalue, lags, _ = kpss(x, regression="c", nlags="auto")
    return {"stat": stat, "pvalue": pvalue, "lags": lags, "nobs": len(x)}

def create_month_dummies(df):
    d = pd.get_dummies(df.index.month, prefix="m", drop_first=True)
    d.index = df.index
    return d

def create_event_dummies(df):
    idx = df.index
    out = pd.DataFrame(index=idx)
    # Pulso do mês mais extremo da pandemia
    out["pulse_2020_04"] = (idx == pd.Timestamp("2020-04-01")).astype(int)
    # Janela curta do choque tributário / combustíveis de 2022
    out["temp_2022_07_2022_09"] = ((idx >= pd.Timestamp("2022-07-01")) & (idx <= pd.Timestamp("2022-09-01"))).astype(int)
    # Mudança de regime comercial da Petrobras a partir de maio de 2023
    out["regime_2023_05_on"] = (idx >= pd.Timestamp("2023-05-01")).astype(int)
    return out

def safe_log(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)
    return np.log(s)

def johansen_rank(level_data, det_order=0, k_ar_diff=1):
    try:
        joh = coint_johansen(level_data.dropna(), det_order=det_order, k_ar_diff=k_ar_diff)
        trace = joh.lr1
        crit5 = joh.cvt[:, 1]
        rank = int(sum(trace > crit5))
        return rank, joh
    except Exception:
        return np.nan, None


# =========================
# LEITURA E PREPARAÇÃO
# =========================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df["Data"] = pd.to_datetime(df["Data"])
df = df.sort_values("Data").set_index("Data")
df = df[df.index >= pd.Timestamp(DATA_INICIO)].copy()

# Mapeamento EXATO pedido por você
map_cols = {
    "ipca_geral_nivel": candidate_col(df, "IPCA_Geral_nivel", "IPCA_Brasil"),
    "ipca_trans_nivel": candidate_col(df, "IPCA_Trans_nivel", "Var_IPCA_Trans"),
    "gasolinaA_nivel": candidate_col(df, "GasolinaABrasil_media_nivel", "GasolinaABrasil_media"),
    "gasolina_nivel": candidate_col(df, "Gasolina_nivel", "Gasolina"),
    "etanol_nivel": candidate_col(df, "Etanol_nivel", "Etanol"),
    "oleo_diesel_nivel": candidate_col(df, "Oleo_diesel_nivel", "Oleo_diesel"),
    "cambio": "Cambio",
    "preco_barril": "Preco_Barril",
    "atividade": "Atividade",
}

ensure_numeric(df, list(set(map_cols.values())))

# Se IPCA_Trans_nivel não existir, reconstrói a partir da variação
if map_cols["ipca_trans_nivel"] == "Var_IPCA_Trans":
    df["IPCA_Trans_nivel_fallback"] = build_index_from_var(df["Var_IPCA_Trans"], base=100.0)
    map_cols["ipca_trans_nivel"] = "IPCA_Trans_nivel_fallback"

# Se colunas de combustíveis vierem em variação no arquivo antigo, reconstrói
for map_key, new_name, old_name in [
    ("gasolina_nivel", "Gasolina_nivel_fallback", "Gasolina"),
    ("etanol_nivel", "Etanol_nivel_fallback", "Etanol"),
    ("oleo_diesel_nivel", "Oleo_diesel_nivel_fallback", "Oleo_diesel"),
]:
    if map_cols[map_key] == old_name:
        # Heurística: se houver valores <= 0 frequentes ou magnitudes pequenas, assume variação %
        s = pd.to_numeric(df[old_name], errors="coerce")
        if (s.le(0).mean() > 0.05) or (s.abs().median() < 20):
            df[new_name] = build_index_from_var(df[old_name], base=100.0)
            map_cols[map_key] = new_name

# Renomear para nomes internos padronizados
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

# Logs e dln
for c in ["IPCA_Geral_nivel", "IPCA_Trans_nivel", "GasolinaA_nivel", "Gasolina_nivel",
          "Etanol_nivel", "Oleo_diesel_nivel", "Cambio", "Preco_Barril", "Atividade"]:
    work[f"LN_{c}"] = safe_log(work[c])
    work[f"DLN_{c}"] = work[f"LN_{c}"].diff()

# Gráficos
for c in ["IPCA_Geral_nivel", "IPCA_Trans_nivel", "GasolinaA_nivel", "Gasolina_nivel",
          "Etanol_nivel", "Oleo_diesel_nivel", "Cambio", "Preco_Barril", "Atividade"]:
    plot_series(work[c].dropna(), f"{c} - nível", OUTDIR / "graficos" / "01_nivel" / f"{c}_nivel.png")
    plot_series(work[f"LN_{c}"].dropna(), f"LN_{c}", OUTDIR / "graficos" / "02_ln" / f"{c}_ln.png")
    plot_series(work[f"DLN_{c}"].dropna(), f"DLN_{c}", OUTDIR / "graficos" / "03_dln" / f"{c}_dln.png")

# Testes ADF / KPSS
tests = []
for c in ["IPCA_Geral_nivel", "IPCA_Trans_nivel", "GasolinaA_nivel", "Gasolina_nivel",
          "Etanol_nivel", "Oleo_diesel_nivel", "Cambio", "Preco_Barril", "Atividade"]:
    for kind in ["level", "dln"]:
        s = work[c] if kind == "level" else work[f"DLN_{c}"]
        tests.append({
            "variavel": c,
            "forma": kind,
            **{f"adf_{k}": v for k, v in adf_test(s).items()},
            **{f"kpss_{k}": v for k, v in kpss_test(s).items()},
        })
pd.DataFrame(tests).to_excel(OUTDIR / "tabelas" / "testes_estacionariedade.xlsx", index=False)

# Exógenas
# Mantemos um bloco enxuto para evitar overfitting:
# - DLN_Atividade
# - 11 dummies mensais
# - 3 dummies de evento economicamente defensáveis
exog = pd.concat([
    work[["DLN_Atividade"]],
    create_month_dummies(work),
    create_event_dummies(work)
], axis=1).apply(pd.to_numeric, errors="coerce").astype(float)

# Salva as exógenas para conferência
exog.to_excel(OUTDIR / "tabelas" / "exogenas_utilizadas.xlsx")

# Modelos
comb_map = {
    "GasolinaA": "GasolinaA_nivel",
    "Gasolina": "Gasolina_nivel",
    "Etanol": "Etanol_nivel",
    "Oleo_diesel": "Oleo_diesel_nivel",
}
resp_map = {
    "IPCA_Brasil": "IPCA_Geral_nivel",
    "IPCA_Transporte": "IPCA_Trans_nivel",
}

sumarios = []
causalidade = []
residuos_diag = []
johansen_tbl = []

for comb_name, comb_col in comb_map.items():
    for resp_name, resp_col in resp_map.items():
        nome = f"{comb_name}_{resp_name}"

        model_df = pd.concat([
            work["DLN_Preco_Barril"],
            work["DLN_Cambio"],
            work[f"DLN_{comb_col}"],
            work[f"DLN_{resp_col}"],
            exog
        ], axis=1).dropna()

        endog = model_df.iloc[:, :4].copy()
        endog.columns = ["DLN_Preco_Barril", "DLN_Cambio", "DLN_Combustivel", "DLN_IPCA_Resposta"]
        X = model_df.iloc[:, 4:].copy()

        if len(endog) < 50:
            continue

        varsel = VAR(endog, exog=X)
        sel = varsel.select_order(MAXLAGS)
        lag = sel.selected_orders.get("aic", None)
        if lag is None or lag < 1:
            lag = sel.selected_orders.get("bic", 1)
        lag = max(1, int(lag))

        res = varsel.fit(lag)
        with open(OUTDIR / "modelos" / f"resumo_{nome}.txt", "w", encoding="utf-8") as f:
            f.write(str(res.summary()))

        sumarios.append({
            "modelo": nome,
            "nobs": res.nobs,
            "lag": lag,
            "aic": res.aic,
            "bic": res.bic,
            "hqic": res.hqic,
            "estavel": bool(np.all(np.abs(res.roots) > 1)),
        })

        # Johansen em log-nível
        level_data = pd.concat([
            work["LN_Preco_Barril"],
            work["LN_Cambio"],
            work[f"LN_{comb_col}"],
            work[f"LN_{resp_col}"],
        ], axis=1).dropna()
        level_data.columns = ["LN_Preco_Barril", "LN_Cambio", "LN_Combustivel", "LN_IPCA_Resposta"]

        rank, joh = johansen_rank(level_data, det_order=0, k_ar_diff=max(lag - 1, 1))
        johansen_tbl.append({"modelo": nome, "rank_trace_5pct": rank})
        if joh is not None:
            pd.DataFrame({
                "trace_stat": joh.lr1,
                "crit_90": joh.cvt[:, 0],
                "crit_95": joh.cvt[:, 1],
                "crit_99": joh.cvt[:, 2],
            }).to_excel(OUTDIR / "tabelas" / f"johansen_{nome}.xlsx", index=False)

        # Causalidade de Granger
        for cause in ["DLN_Preco_Barril", "DLN_Cambio", "DLN_Combustivel"]:
            try:
                test = res.test_causality("DLN_IPCA_Resposta", [cause], kind="f")
                causalidade.append({
                    "modelo": nome,
                    "causa": cause,
                    "F": float(test.test_statistic),
                    "pvalue": float(test.pvalue),
                    "gl": str(test.df),
                })
            except Exception:
                pass

        # Diagnósticos dos resíduos
        try:
            white = res.test_whiteness(nlags=max(12, lag + 2))
            white_stat = float(getattr(white, "test_statistic", np.nan))
            white_p = float(getattr(white, "pvalue", np.nan))
        except Exception:
            white_stat, white_p = np.nan, np.nan

        try:
            norm = res.test_normality()
            norm_stat = float(getattr(norm, "test_statistic", np.nan))
            norm_p = float(getattr(norm, "pvalue", np.nan))
        except Exception:
            norm_stat, norm_p = np.nan, np.nan

        residuos_diag.append({
            "modelo": nome,
            "whiteness_stat": white_stat,
            "whiteness_pvalue": white_p,
            "normality_stat": norm_stat,
            "normality_pvalue": norm_p,
        })

        resid = pd.DataFrame(res.resid, index=endog.index, columns=endog.columns)
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        for ax, col in zip(axes, resid.columns):
            ax.plot(resid.index, resid[col])
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title(f"{nome} - resíduo {col}")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTDIR / "graficos" / "04_residuos" / f"residuos_{nome}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # BG por equação
        for dep in endog.columns:
            y = endog[dep]
            Xeq = sm.add_constant(pd.concat([endog.shift(i) for i in range(1, lag + 1)], axis=1).join(X))
            aux = pd.concat([y, Xeq], axis=1).dropna()
            y_aux = aux.iloc[:, 0]
            X_aux = aux.iloc[:, 1:]
            try:
                ols = sm.OLS(y_aux, X_aux).fit()
                bg = acorr_breusch_godfrey(ols, nlags=max(1, min(12, lag + 2)))
                residuos_diag.append({
                    "modelo": nome,
                    "equacao": dep,
                    "bg_lm_stat": float(bg[0]),
                    "bg_lm_pvalue": float(bg[1]),
                    "bg_f_stat": float(bg[2]),
                    "bg_f_pvalue": float(bg[3]),
                })
            except Exception:
                pass

        # IRFs
        try:
            irf = res.irf(HORIZONTE_IRF)
            impulses = ["DLN_Preco_Barril", "DLN_Cambio", "DLN_Combustivel"]
            responses = ["DLN_IPCA_Resposta"]
            fig = irf.plot(impulse=impulses[0], response=responses[0])
            plt.tight_layout()
            plt.savefig(OUTDIR / "graficos" / "05_irf" / f"irf_{nome}_{impulses[0]}.png", dpi=180, bbox_inches="tight")
            plt.close()

            for imp in impulses[1:]:
                fig = irf.plot(impulse=imp, response=responses[0])
                plt.tight_layout()
                plt.savefig(OUTDIR / "graficos" / "05_irf" / f"irf_{nome}_{imp}.png", dpi=180, bbox_inches="tight")
                plt.close()
        except Exception:
            pass

pd.DataFrame(sumarios).to_excel(OUTDIR / "tabelas" / "sumario_modelos.xlsx", index=False)
pd.DataFrame(causalidade).to_excel(OUTDIR / "tabelas" / "causalidade_granger.xlsx", index=False)
pd.DataFrame(residuos_diag).to_excel(OUTDIR / "tabelas" / "diagnosticos_residuos.xlsx", index=False)
pd.DataFrame(johansen_tbl).to_excel(OUTDIR / "tabelas" / "johansen_ranks.xlsx", index=False)

print("Concluído.")
print("Mapeamento usado:")
for k, v in map_cols.items():
    print(f"  {k}: {v}")
print(f"Saída em: {OUTDIR.resolve()}")
