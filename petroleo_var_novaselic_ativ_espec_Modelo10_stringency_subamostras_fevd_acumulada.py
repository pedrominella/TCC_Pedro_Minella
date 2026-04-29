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
STRINGENCY_FILE = r"Stringency_index.csv"
SHEET_NAME = "Sheet1"
DATA_INICIO = "2003-01-01"
MAXLAGS = 12
HORIZONTE_IRF = 24
OUTDIR_BASE = Path("output_tcc_var_novaselic_ativ_espec_Modelo10_stringency_subamostras_fevd_acumulada")
SELIC_DAILY_FILE = r"STP-20260429165342557.csv"

SUBAMOSTRAS = [
    ("2003_2014", "2003-01-01", "2014-12-01"),
    ("2015_2026", "2015-01-01", "2026-12-01"),
]

# =========================
# FUNÇÕES
# =========================
def preparar_pastas(outdir):
    (outdir / "graficos" / "01_nivel").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "02_ln").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "03_dln").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "04_residuos").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "05_irf").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "06_fevd").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "07_irf_acumulada").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "08_cholesky_robustez").mkdir(parents=True, exist_ok=True)
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
    raise KeyError(f"Não encontrei nem '{preferred}' nem fallback '{fallback}' no arquivo.")


def build_index_from_var(series_pct, base=100.0):
    s = pd.to_numeric(series_pct, errors="coerce").fillna(0.0) / 100.0
    idx = (1.0 + s).cumprod() * base
    return idx


def plot_series(series, title, outfile):
    if series.dropna().empty:
        return
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
    return d.astype(float)


def drop_constant_or_duplicate_columns(df, tol=1e-12):
    if df is None or df.empty:
        return df

    out = df.copy()

    nunique = out.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    out = out[keep]

    if out.empty:
        return out

    var_ok = out.var(numeric_only=True).fillna(0.0)
    keep = var_ok[var_ok > tol].index.tolist()
    out = out[keep]

    if out.empty:
        return out

    out = out.loc[:, ~out.T.duplicated()]
    return out


def escolher_maxlags_seguro(nobs, nvars, nexog, maxlags_desejado):
    limite = int((nobs - 5) / max(1, (nvars + nexog)))
    limite = max(1, min(maxlags_desejado, limite))
    return limite


def load_stringency_monthly(filepath, target_index):
    s = pd.read_csv(filepath, sep=";", low_memory=False)
    s = s.rename(columns={"#country": "CountryName", "#country+code": "CountryCode", "#date": "Date"})
    if "CountryCode" not in s.columns:
        raise KeyError("Não encontrei a coluna CountryCode no arquivo de stringency.")
    if "Date" not in s.columns:
        raise KeyError("Não encontrei a coluna Date no arquivo de stringency.")

    possible_cols = [
        "StringencyIndex_Average_ForDisplay",
        "StringencyIndex_Average",
        "StringencyIndex",
    ]
    str_col = next((c for c in possible_cols if c in s.columns), None)
    if str_col is None:
        raise KeyError("Não encontrei a coluna do índice de stringency no CSV.")

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



def encontrar_arquivo_selic(filepath):
    candidatos = [Path(filepath)]
    candidatos += sorted(Path('.').glob('STP*.csv'))
    candidatos += sorted(Path(__file__).resolve().parent.glob('STP*.csv'))
    vistos = set()
    for c in candidatos:
        c = Path(c)
        key = str(c.resolve()) if c.exists() else str(c)
        if key in vistos:
            continue
        vistos.add(key)
        if c.exists():
            return c
    raise FileNotFoundError(
        "Não encontrei o CSV diário da Selic. Coloque o arquivo STP-20260429165342557.csv "
        "na mesma pasta do script ou ajuste SELIC_DAILY_FILE."
    )


def load_selic_meta_mensal(filepath, target_index):
    """Lê a meta Selic diária e transforma em média mensal, alinhada à base principal."""
    arquivo_selic = encontrar_arquivo_selic(filepath)
    s = pd.read_csv(arquivo_selic, sep=';', low_memory=False)
    s.columns = [str(c).strip() for c in s.columns]

    data_col = next((c for c in s.columns if c.lower() in ['data', 'date']), None)
    if data_col is None:
        data_col = s.columns[0]

    valor_col = next((c for c in s.columns if c != data_col and ('selic' in c.lower() or '432' in c.lower())), None)
    if valor_col is None:
        valor_col = [c for c in s.columns if c != data_col][0]

    s[data_col] = pd.to_datetime(s[data_col], dayfirst=True, errors='coerce')
    s[valor_col] = (
        s[valor_col].astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    s[valor_col] = pd.to_numeric(s[valor_col], errors='coerce')
    s = s.dropna(subset=[data_col, valor_col]).sort_values(data_col)
    s['Data_mes'] = s[data_col].dt.to_period('M').dt.to_timestamp()

    selic_m = s.groupby('Data_mes', as_index=True)[valor_col].mean().to_frame('Selic')
    selic_m = selic_m.reindex(target_index)
    selic_m['Selic'] = selic_m['Selic'].interpolate(method='time').ffill().bfill()
    return selic_m, str(arquivo_selic), valor_col


def irf_orth_array(res, horizonte):
    irf = res.irf(horizonte)
    return irf, irf.orth_irfs


def salvar_irf_acumulada(irf_array, endog_cols, nome, outdir):
    response = 'DLN_IPCA_Resposta'
    impulses = ['DLN_Preco_Barril', 'DLN_Combustivel']
    if response not in endog_cols:
        return pd.DataFrame()
    resp_idx = endog_cols.index(response)
    horizontes = np.arange(irf_array.shape[0])
    tabela = pd.DataFrame({'horizonte': horizontes})
    fig, ax = plt.subplots(figsize=(10, 5))
    for imp in impulses:
        if imp not in endog_cols:
            continue
        imp_idx = endog_cols.index(imp)
        acumulada = np.cumsum(irf_array[:, resp_idx, imp_idx])
        tabela[f'acumulada_{imp}_para_IPCA'] = acumulada
        ax.plot(horizontes, acumulada, marker='o', linewidth=1.5, label=imp)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title(f'{nome} - resposta acumulada do IPCA a petróleo e combustível')
    ax.set_xlabel('Horizonte mensal')
    ax.set_ylabel('Resposta acumulada')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'graficos' / '07_irf_acumulada' / f'irf_acumulada_{nome}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    tabela.to_excel(outdir / 'tabelas' / f'irf_acumulada_{nome}.xlsx', index=False)
    return tabela


def salvar_fevd(res, endog_cols, nome, outdir, horizonte):
    try:
        fevd = res.fevd(horizonte)
        decomp = fevd.decomp
        registros = []
        for i_resp, resp in enumerate(endog_cols):
            for h in range(decomp.shape[1]):
                for i_imp, imp in enumerate(endog_cols):
                    registros.append({
                        'modelo': nome,
                        'resposta': resp,
                        'horizonte': h + 1,
                        'impulso': imp,
                        'participacao': float(decomp[i_resp, h, i_imp]),
                    })
        fevd_df = pd.DataFrame(registros)
        fevd_df.to_excel(outdir / 'tabelas' / f'fevd_{nome}.xlsx', index=False)
        alvo = fevd_df[fevd_df['resposta'] == 'DLN_IPCA_Resposta'].copy()
        if not alvo.empty:
            pivot = alvo.pivot(index='horizonte', columns='impulso', values='participacao')
            ax = pivot.plot(kind='bar', stacked=True, figsize=(11, 5), width=0.85)
            ax.set_title(f'{nome} - FEVD da resposta do IPCA')
            ax.set_xlabel('Horizonte mensal')
            ax.set_ylabel('Participação na variância do erro')
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / 'graficos' / '06_fevd' / f'fevd_ipca_{nome}.png', dpi=180, bbox_inches='tight')
            plt.close()
        return fevd_df
    except Exception as e:
        print(f'Erro ao calcular FEVD de {nome}: {e}')
        return pd.DataFrame()


def testar_ordens_cholesky(endog, X, lag, nome, nome_sub, outdir, horizonte):
    ordens = {
        'ordem_base_petroleo_cambio_combustivel': [
            'DLN_Preco_Barril', 'DLN_Cambio', 'DLN_Combustivel', 'DLN_Atividade',
            'DLN_Expectativa_Inflacao', 'LN_Selic', 'DLN_IPCA_Resposta'
        ],
        'ordem_petroleo_combustivel_cambio': [
            'DLN_Preco_Barril', 'DLN_Combustivel', 'DLN_Cambio', 'DLN_Atividade',
            'DLN_Expectativa_Inflacao', 'LN_Selic', 'DLN_IPCA_Resposta'
        ],
        'ordem_cambio_petroleo_combustivel': [
            'DLN_Cambio', 'DLN_Preco_Barril', 'DLN_Combustivel', 'DLN_Atividade',
            'DLN_Expectativa_Inflacao', 'LN_Selic', 'DLN_IPCA_Resposta'
        ],
        'ordem_petroleo_cambio_selic_combustivel': [
            'DLN_Preco_Barril', 'DLN_Cambio', 'LN_Selic', 'DLN_Combustivel', 'DLN_Atividade',
            'DLN_Expectativa_Inflacao', 'DLN_IPCA_Resposta'
        ],
        'ordem_petroleo_cambio_combustivel_ipca_antes_selic': [
            'DLN_Preco_Barril', 'DLN_Cambio', 'DLN_Combustivel', 'DLN_Atividade',
            'DLN_Expectativa_Inflacao', 'DLN_IPCA_Resposta', 'LN_Selic'
        ],
    }
    registros = []
    series_plot = {'DLN_Preco_Barril': {}, 'DLN_Cambio': {}, 'DLN_Combustivel': {}}
    for ordem_nome, ordem_cols in ordens.items():
        try:
            endog_ord = endog[ordem_cols].copy()
            modelo = VAR(endog_ord, exog=X if X is not None and not X.empty else None)
            res_ord = modelo.fit(lag)
            _, arr = irf_orth_array(res_ord, horizonte)
            resp_idx = ordem_cols.index('DLN_IPCA_Resposta')
            for imp in ['DLN_Preco_Barril', 'DLN_Cambio', 'DLN_Combustivel']:
                imp_idx = ordem_cols.index(imp)
                acumulada = np.cumsum(arr[:, resp_idx, imp_idx])
                series_plot[imp][ordem_nome] = acumulada
                registros.append({
                    'subamostra': nome_sub,
                    'modelo': nome,
                    'ordem_cholesky': ordem_nome,
                    'impulso': imp,
                    'resposta': 'DLN_IPCA_Resposta',
                    'acumulada_h6': float(acumulada[min(6, len(acumulada)-1)]),
                    'acumulada_h12': float(acumulada[min(12, len(acumulada)-1)]),
                    'acumulada_h24': float(acumulada[min(24, len(acumulada)-1)]),
                    'sinal_h12': np.sign(float(acumulada[min(12, len(acumulada)-1)])),
                })
        except Exception as e:
            registros.append({'subamostra': nome_sub, 'modelo': nome, 'ordem_cholesky': ordem_nome, 'erro': str(e)})
    robustez = pd.DataFrame(registros)
    robustez.to_excel(outdir / 'tabelas' / f'robustez_cholesky_{nome}.xlsx', index=False)
    for imp, curvas in series_plot.items():
        if not curvas:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for ordem_nome, valores in curvas.items():
            ax.plot(np.arange(len(valores)), valores, linewidth=1.4, label=ordem_nome)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f'{nome} - robustez Cholesky: {imp} -> IPCA')
        ax.set_xlabel('Horizonte mensal')
        ax.set_ylabel('Resposta acumulada')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        plt.savefig(outdir / 'graficos' / '08_cholesky_robustez' / f'robustez_cholesky_{nome}_{imp}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    return robustez

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
        "selic": "CSV_diario_meta_selic_media_mensal",
        "expectativa_inflacao": "espectativa_inflacao",
    }

    ensure_numeric(df, list(set(map_cols.values())))

    if map_cols["ipca_trans_nivel"] == "Var_IPCA_Trans":
        df["IPCA_Trans_nivel_fallback"] = build_index_from_var(df["Var_IPCA_Trans"], base=100.0)
        map_cols["ipca_trans_nivel"] = "IPCA_Trans_nivel_fallback"

    for map_key, new_name, old_name in [
        ("gasolina_nivel", "Gasolina_nivel_fallback", "Var_Gasolina"),
        ("etanol_nivel", "Etanol_nivel_fallback", "Var_Etanol"),
        ("oleo_diesel_nivel", "Oleo_diesel_nivel_fallback", "Var_Oleo_diesel"),
    ]:
        if map_cols[map_key] == old_name:
            df[new_name] = build_index_from_var(df[old_name], base=100.0)
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
    selic_mensal, selic_arquivo_usado, selic_coluna_usada = load_selic_meta_mensal(SELIC_DAILY_FILE, work.index)
    work["Selic"] = pd.to_numeric(selic_mensal["Selic"], errors="coerce")
    map_cols["selic_arquivo_usado"] = selic_arquivo_usado
    map_cols["selic_coluna_usada"] = selic_coluna_usada
    work["Expectativa_Inflacao"] = pd.to_numeric(df[map_cols["expectativa_inflacao"]], errors="coerce")

    stringency = load_stringency_monthly(STRINGENCY_FILE, work.index)
    work["Stringency"] = pd.to_numeric(stringency["Stringency"], errors="coerce")

    # Ordem econômica usada para gráficos, testes e transformações.
    # Para identificação por Cholesky nas IRFs, a lógica é:
    # petróleo internacional -> câmbio -> combustíveis -> condições domésticas -> inflação.
    base_vars = [
        "Preco_Barril",
        "Cambio",
        "GasolinaA_nivel",
        "Gasolina_nivel",
        "Etanol_nivel",
        "Oleo_diesel_nivel",
        "Atividade",
        "Expectativa_Inflacao",
        "Selic",
        "IPCA_Trans_nivel",
        "IPCA_Geral_nivel",
    ]

    for c in base_vars:
        work[f"LN_{c}"] = safe_log(work[c])
        work[f"DLN_{c}"] = work[f"LN_{c}"].diff()

    return work, map_cols, base_vars


def rodar_subamostra(work_full, base_vars, map_cols, nome_sub, data_ini, data_fim):
    outdir = OUTDIR_BASE / nome_sub
    preparar_pastas(outdir)

    mask = (work_full.index >= pd.Timestamp(data_ini)) & (work_full.index <= pd.Timestamp(data_fim))
    work = work_full.loc[mask].copy()

    if work.empty:
        print(f"Subamostra {nome_sub} vazia. Pulando.")
        return

    for c in base_vars:
        plot_series(work[c].dropna(), f"{c} - nível ({nome_sub})", outdir / "graficos" / "01_nivel" / f"{c}_nivel.png")
        plot_series(work[f"LN_{c}"].dropna(), f"LN_{c} ({nome_sub})", outdir / "graficos" / "02_ln" / f"{c}_ln.png")
        plot_series(work[f"DLN_{c}"].dropna(), f"DLN_{c} ({nome_sub})", outdir / "graficos" / "03_dln" / f"{c}_dln.png")

    plot_series(work["Stringency"].dropna(), f"Stringency - nível mensal ({nome_sub})", outdir / "graficos" / "01_nivel" / "Stringency_nivel.png")

    tests = []
    for c in base_vars:
        formas_teste = ["level", "ln"] if c == "Selic" else ["level", "dln"]
        for kind in formas_teste:
            s = work[c] if kind == "level" else (work[f"LN_{c}"] if kind == "ln" else work[f"DLN_{c}"])
            tests.append({
                "subamostra": nome_sub,
                "variavel": c,
                "forma": kind,
                **{f"adf_{k}": v for k, v in adf_test(s).items()},
                **{f"kpss_{k}": v for k, v in kpss_test(s).items()},
            })
    pd.DataFrame(tests).to_excel(outdir / "tabelas" / "testes_estacionariedade.xlsx", index=False)
    work[["Stringency"]].to_excel(outdir / "tabelas" / "stringency_mensal.xlsx")

    exog = pd.concat([
        create_month_dummies(work),
        work[["Stringency"]],
    ], axis=1).apply(pd.to_numeric, errors="coerce").astype(float)
    exog = drop_constant_or_duplicate_columns(exog)

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
    fevd_todos = []
    robustez_cholesky_todos = []

    for comb_name, comb_col in comb_map.items():
        for resp_name, resp_col in resp_map.items():
            nome = f"{comb_name}_{resp_name}"

            model_df = pd.concat([
                work["DLN_Preco_Barril"],
                work["DLN_Cambio"],
                work[f"DLN_{comb_col}"],
                work["DLN_Atividade"],
                work["DLN_Expectativa_Inflacao"],
                work["LN_Selic"],
                work[f"DLN_{resp_col}"],
                exog,
            ], axis=1).dropna()

            endog = model_df.iloc[:, :7].copy()
            endog.columns = [
                "DLN_Preco_Barril",
                "DLN_Cambio",
                "DLN_Combustivel",
                "DLN_Atividade",
                "DLN_Expectativa_Inflacao",
                "LN_Selic",
                "DLN_IPCA_Resposta",
            ]
            X = model_df.iloc[:, 7:].copy()
            X = drop_constant_or_duplicate_columns(X)

            if len(endog) < 50:
                print(f"Modelo {nome_sub} - {nome} com poucas observações ({len(endog)}). Pulando.")
                continue

            try:
                nvars = endog.shape[1]
                nexog = 0 if X is None else X.shape[1]
                maxlags_efetivo = escolher_maxlags_seguro(len(endog), nvars, nexog, MAXLAGS)

                varsel = VAR(endog, exog=X if nexog > 0 else None)
                sel = varsel.select_order(maxlags_efetivo)
                lag = sel.selected_orders.get("aic", None)
                if lag is None or lag < 1:
                    lag = sel.selected_orders.get("bic", 1)
                lag = max(1, int(lag))

                res = varsel.fit(lag)
            except Exception as e:
                print(f"Erro no ajuste VAR de {nome_sub} - {nome}: {e}")
                continue

            with open(outdir / "modelos" / f"resumo_{nome}.txt", "w", encoding="utf-8") as f:
                f.write(str(res.summary()))

            sumarios.append({
                "subamostra": nome_sub,
                "modelo": nome,
                "nobs": res.nobs,
                "lag": lag,
                "aic": res.aic,
                "bic": res.bic,
                "hqic": res.hqic,
                "estavel": bool(np.all(np.abs(res.roots) > 1)),
            })

            level_data = pd.concat([
                work["LN_Preco_Barril"],
                work["LN_Cambio"],
                work[f"LN_{comb_col}"],
                work["LN_Atividade"],
                work["LN_Expectativa_Inflacao"],
                work["LN_Selic"],
                work[f"LN_{resp_col}"],
            ], axis=1).dropna()
            level_data.columns = [
                "LN_Preco_Barril",
                "LN_Cambio",
                "LN_Combustivel",
                "LN_Atividade",
                "LN_Expectativa_Inflacao",
                "LN_Selic",
                "LN_IPCA_Resposta",
            ]

            rank, joh = johansen_rank(level_data, det_order=0, k_ar_diff=max(lag - 1, 1))
            johansen_tbl.append({"subamostra": nome_sub, "modelo": nome, "rank_trace_5pct": rank})
            if joh is not None:
                pd.DataFrame({
                    "trace_stat": joh.lr1,
                    "crit_90": joh.cvt[:, 0],
                    "crit_95": joh.cvt[:, 1],
                    "crit_99": joh.cvt[:, 2],
                }).to_excel(outdir / "tabelas" / f"johansen_{nome}.xlsx", index=False)

            for cause in [
                "DLN_Preco_Barril",
                "DLN_Cambio",
                "DLN_Combustivel",
                "DLN_Atividade",
                "DLN_Expectativa_Inflacao",
                "LN_Selic",
            ]:
                try:
                    test = res.test_causality("DLN_IPCA_Resposta", [cause], kind="f")
                    causalidade.append({
                        "subamostra": nome_sub,
                        "modelo": nome,
                        "causa": cause,
                        "F": float(test.test_statistic),
                        "pvalue": float(test.pvalue),
                        "gl": str(test.df),
                    })
                except Exception:
                    pass

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
                "subamostra": nome_sub,
                "modelo": nome,
                "whiteness_stat": white_stat,
                "whiteness_pvalue": white_p,
                "normality_stat": norm_stat,
                "normality_pvalue": norm_p,
            })

            resid = pd.DataFrame(res.resid, index=endog.index, columns=endog.columns)
            fig, axes = plt.subplots(len(resid.columns), 1, figsize=(10, 14), sharex=True)
            if len(resid.columns) == 1:
                axes = [axes]
            for ax, col in zip(axes, resid.columns):
                ax.plot(resid.index, resid[col])
                ax.axhline(0, color="black", lw=0.8)
                ax.set_title(f"{nome_sub} - {nome} - resíduo {col}")
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / "graficos" / "04_residuos" / f"residuos_{nome}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

            for dep in endog.columns:
                y = endog[dep]
                bloco_exog = X if X is not None and not X.empty else None
                Xeq = pd.concat([endog.shift(i) for i in range(1, lag + 1)], axis=1)
                if bloco_exog is not None:
                    Xeq = Xeq.join(bloco_exog)
                Xeq = drop_constant_or_duplicate_columns(Xeq)
                Xeq = sm.add_constant(Xeq, has_constant="add")
                aux = pd.concat([y, Xeq], axis=1).dropna()
                y_aux = aux.iloc[:, 0]
                X_aux = aux.iloc[:, 1:]
                try:
                    ols = sm.OLS(y_aux, X_aux).fit()
                    bg = acorr_breusch_godfrey(ols, nlags=max(1, min(12, lag + 2)))
                    residuos_diag.append({
                        "subamostra": nome_sub,
                        "modelo": nome,
                        "equacao": dep,
                        "bg_lm_stat": float(bg[0]),
                        "bg_lm_pvalue": float(bg[1]),
                        "bg_f_stat": float(bg[2]),
                        "bg_f_pvalue": float(bg[3]),
                    })
                except Exception:
                    pass

            try:
                irf, irf_orth = irf_orth_array(res, HORIZONTE_IRF)
                impulses = [
                    "DLN_Preco_Barril",
                    "DLN_Cambio",
                    "DLN_Combustivel",
                    "DLN_Atividade",
                    "DLN_Expectativa_Inflacao",
                    "LN_Selic",
                ]
                for imp in impulses:
                    fig = irf.plot(impulse=imp, response="DLN_IPCA_Resposta", orth=True)
                    plt.tight_layout()
                    plt.savefig(outdir / "graficos" / "05_irf" / f"irf_{nome}_{imp}.png", dpi=180, bbox_inches="tight")
                    plt.close()
                salvar_irf_acumulada(irf_orth, list(endog.columns), nome, outdir)
            except Exception as e:
                print(f"Erro ao gerar IRFs de {nome_sub} - {nome}: {e}")

            fevd_df = salvar_fevd(res, list(endog.columns), nome, outdir, HORIZONTE_IRF)
            if fevd_df is not None and not fevd_df.empty:
                fevd_df["subamostra"] = nome_sub
                fevd_todos.append(fevd_df)

            robustez_df = testar_ordens_cholesky(endog, X, lag, nome, nome_sub, outdir, HORIZONTE_IRF)
            if robustez_df is not None and not robustez_df.empty:
                robustez_cholesky_todos.append(robustez_df)

    pd.DataFrame(sumarios).to_excel(outdir / "tabelas" / "sumario_modelos.xlsx", index=False)
    pd.DataFrame(causalidade).to_excel(outdir / "tabelas" / "causalidade_granger.xlsx", index=False)
    pd.DataFrame(residuos_diag).to_excel(outdir / "tabelas" / "diagnosticos_residuos.xlsx", index=False)
    pd.DataFrame(johansen_tbl).to_excel(outdir / "tabelas" / "johansen_ranks.xlsx", index=False)
    if fevd_todos:
        pd.concat(fevd_todos, ignore_index=True).to_excel(outdir / "tabelas" / "fevd_todos_modelos.xlsx", index=False)
    if robustez_cholesky_todos:
        pd.concat(robustez_cholesky_todos, ignore_index=True).to_excel(outdir / "tabelas" / "robustez_cholesky_todos_modelos.xlsx", index=False)

    print(f"Concluído: {nome_sub}")
    print(f"Saída em: {outdir.resolve()}")


def main():
    OUTDIR_BASE.mkdir(parents=True, exist_ok=True)
    work, map_cols, base_vars = carregar_preparar_base()

    print("Mapeamento usado:")
    for k, v in map_cols.items():
        print(f"  {k}: {v}")

    for nome_sub, data_ini, data_fim in SUBAMOSTRAS:
        rodar_subamostra(work, base_vars, map_cols, nome_sub, data_ini, data_fim)

    print("Todas as subamostras foram processadas.")


if __name__ == "__main__":
    main()
