# -*- coding: utf-8 -*-
"""
petroleo_lp_modelo1.py

Local Projections para o TCC:
Choques no preço internacional do petróleo, combustíveis domésticos e inflação brasileira.

O script estima respostas dinâmicas por Local Projections para:
1) Petróleo -> combustíveis
2) Petróleo -> IPCA Geral/IPCA Transporte, controlando pelo canal dos combustíveis
3) Combustíveis -> IPCA Geral/IPCA Transporte, controlando por petróleo e câmbio

Principais escolhas:
- Selic diária do SGS 432 é convertida em média mensal.
- Selic mensal é transformada em índice nível base 100.
- Selic entra nos modelos em log: LN_Selic, sem primeira diferença.
- Meta de inflação entra como variável exógena após ser transformada em nível base 100.
- Erros-padrão HAC/Newey-West em cada horizonte.
- São geradas respostas mensais e acumuladas.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ======================================================
# CONFIGURAÇÕES
# ======================================================
FILE_PATH = r"IPCA.xlsx"
SHEET_NAME = "Sheet1"
STRINGENCY_FILE = r"Stringency_index.csv"
SELIC_DAILY_FILE = r"STP-20260429165342557.csv"

DATA_INICIO = "2003-01-01"
HORIZONTE_MAX = 24
LAGS_CONTROLES = 3
PADRONIZAR_CHOQUE = True  # True: resposta a choque de 1 desvio-padrão
INTERVALO_CONFIANCA = 0.95

OUTDIR_BASE = Path("output_tcc_local_projections_modelo1")

SUBAMOSTRAS = [
    ("2003_2014", "2003-01-01", "2014-12-01"),
    ("2015_2026", "2015-01-01", "2026-12-01"),
    ("2003_2026", "2003-01-01", "2026-12-01"),
]


# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================
def preparar_pastas(outdir: Path) -> None:
    (outdir / "tabelas").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "01_lp_mensal").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "02_lp_acumulada").mkdir(parents=True, exist_ok=True)
    (outdir / "graficos" / "03_series_controle").mkdir(parents=True, exist_ok=True)


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detectar_coluna_data(df: pd.DataFrame) -> str:
    candidatos_nome = [
        "Data", "DATA", "data", "Date", "date", "Periodo", "Período", "periodo", "Mês", "Mes", "mes"
    ]
    for c in candidatos_nome:
        if c in df.columns:
            return c

    # Se alguma coluna já é datetime, usa ela.
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c

    # Tentativa: procurar coluna que tenha muitas datas válidas.
    melhor_coluna = None
    melhor_validos = -1
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        validos = parsed.notna().sum()
        if validos > melhor_validos:
            melhor_validos = validos
            melhor_coluna = c

    if melhor_coluna is None or melhor_validos == 0:
        raise KeyError(
            "Não consegui detectar a coluna de data. Renomeie a coluna de datas para 'Data' no Excel."
        )
    return melhor_coluna


def to_numeric_br(s: pd.Series) -> pd.Series:
    """Converte número em formato brasileiro para float quando necessário."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def candidate_col(df: pd.DataFrame, candidates, required=True):
    if isinstance(candidates, str):
        candidates = [candidates]
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Não encontrei nenhuma destas colunas no arquivo: {candidates}")
    return None


def build_index_from_var(series_pct: pd.Series, base: float = 100.0) -> pd.Series:
    """
    Cria índice base 100 a partir de uma taxa percentual de variação mensal.
    Exemplo: 0,5 significa 0,5% no mês.
    """
    s = to_numeric_br(series_pct).fillna(0.0) / 100.0
    return (1.0 + s).cumprod() * base


def build_index_from_rate_level(series_rate: pd.Series, base: float = 100.0) -> pd.Series:
    """
    Cria uma série em base 100 a partir de uma variável em nível.
    Normaliza pelo primeiro valor válido.
    Usado para Selic meta mensal e meta de inflação.
    """
    s = to_numeric_br(series_rate)
    first_valid = s.dropna().iloc[0] if not s.dropna().empty else np.nan
    if pd.isna(first_valid) or first_valid == 0:
        return pd.Series(np.nan, index=series_rate.index)
    return (s / first_valid) * base


def safe_log(s: pd.Series) -> pd.Series:
    s = to_numeric_br(s)
    s = s.where(s > 0)
    return np.log(s)


def drop_constant_or_duplicate_columns(df: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.loc[:, ~out.columns.duplicated()]

    nunique = out.nunique(dropna=False)
    out = out[nunique[nunique > 1].index.tolist()]
    if out.empty:
        return out

    variancia = out.var(numeric_only=True).fillna(0.0)
    out = out[variancia[variancia > tol].index.tolist()]
    if out.empty:
        return out

    out = out.loc[:, ~out.T.duplicated()]
    return out


def create_month_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.get_dummies(index.month, prefix="m", drop_first=True)
    d.index = index
    return d.astype(float)


def plot_series(series: pd.Series, title: str, outfile: Path) -> None:
    if series.dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)


def encontrar_arquivo(filepath: str, padrao: str) -> Path | None:
    candidatos = [Path(filepath)]
    try:
        candidatos += sorted(Path(".").glob(padrao))
        candidatos += sorted(Path(__file__).resolve().parent.glob(padrao))
    except Exception:
        pass

    vistos = set()
    for c in candidatos:
        c = Path(c)
        key = str(c.resolve()) if c.exists() else str(c)
        if key in vistos:
            continue
        vistos.add(key)
        if c.exists():
            return c
    return None


def load_selic_meta_mensal(filepath: str, target_index: pd.DatetimeIndex):
    arquivo = encontrar_arquivo(filepath, "STP*.csv")
    if arquivo is None:
        raise FileNotFoundError(
            "Não encontrei o CSV diário da Selic. Coloque o arquivo STP-20260429165342557.csv "
            "na mesma pasta do script ou ajuste SELIC_DAILY_FILE."
        )

    s = pd.read_csv(arquivo, sep=";", low_memory=False)
    s = normalizar_colunas(s)

    data_col = next((c for c in s.columns if c.lower() in ["data", "date"]), None)
    if data_col is None:
        data_col = s.columns[0]

    valor_col = next(
        (c for c in s.columns if c != data_col and ("selic" in c.lower() or "432" in c.lower())),
        None,
    )
    if valor_col is None:
        valor_col = [c for c in s.columns if c != data_col][0]

    s[data_col] = pd.to_datetime(s[data_col], dayfirst=True, errors="coerce")
    s[valor_col] = to_numeric_br(s[valor_col])
    s = s.dropna(subset=[data_col, valor_col]).sort_values(data_col)
    s["Data_mes"] = s[data_col].dt.to_period("M").dt.to_timestamp()

    selic_m = s.groupby("Data_mes", as_index=True)[valor_col].mean().to_frame("Selic_Meta_Mensal")
    selic_m = selic_m.reindex(target_index)
    selic_m["Selic_Meta_Mensal"] = selic_m["Selic_Meta_Mensal"].interpolate(method="time").ffill().bfill()
    return selic_m, str(arquivo), valor_col


def load_stringency_monthly(filepath: str, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    arquivo = encontrar_arquivo(filepath, "Stringency*.csv")
    if arquivo is None:
        # Se não encontrar, cria zero para não quebrar o modelo.
        return pd.DataFrame({"Stringency": 0.0}, index=target_index)

    s = pd.read_csv(arquivo, sep=";", low_memory=False)
    s = normalizar_colunas(s)
    s = s.rename(columns={"#country": "CountryName", "#country+code": "CountryCode", "#date": "Date"})

    if "CountryCode" not in s.columns or "Date" not in s.columns:
        return pd.DataFrame({"Stringency": 0.0}, index=target_index)

    possible_cols = [
        "StringencyIndex_Average_ForDisplay",
        "StringencyIndex_Average",
        "StringencyIndex",
    ]
    str_col = next((c for c in possible_cols if c in s.columns), None)
    if str_col is None:
        return pd.DataFrame({"Stringency": 0.0}, index=target_index)

    s = s[s["CountryCode"].astype(str).str.upper() == "BRA"].copy()
    s["Date"] = pd.to_datetime(s["Date"].astype(str), format="%Y%m%d", errors="coerce")
    s[str_col] = to_numeric_br(s[str_col])
    s = s.dropna(subset=["Date"]).sort_values("Date")
    s["Data_mes"] = s["Date"].dt.to_period("M").dt.to_timestamp()
    s_m = s.groupby("Data_mes", as_index=True)[str_col].mean().to_frame("Stringency")
    s_m = s_m.reindex(target_index)
    s_m["Stringency"] = s_m["Stringency"].fillna(0.0)
    return s_m


# ======================================================
# PREPARAÇÃO DA BASE
# ======================================================
def carregar_preparar_base():
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    df = normalizar_colunas(df)

    data_col = detectar_coluna_data(df)
    df[data_col] = pd.to_datetime(df[data_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[data_col]).sort_values(data_col).set_index(data_col)
    df.index.name = "Data"
    df = df[df.index >= pd.Timestamp(DATA_INICIO)].copy()

    # Colunas da base principal. O código aceita nomes alternativos usados nas suas versões anteriores.
    col_ipca_geral = candidate_col(df, ["IPCA_Geral_nivel", "IPCA_Geral", "Var_IPCA_Brasil"])
    col_ipca_trans = candidate_col(df, ["IPCA_Trans_nivel", "Var_IPCA_Trans", "Var_IPCA_trans", "IPCA_Transporte"])
    col_gasolinaA = candidate_col(df, ["GasolinaABrasil_media_nivel", "GasolinaA_nivel", "GasolinaABrasil_media"])
    col_gasolina = candidate_col(df, ["Gasolina_nivel", "Var_Gasolina", "Gasolina"])
    col_etanol = candidate_col(df, ["Etanol_nivel", "Var_Etanol", "Etanol"])
    col_diesel = candidate_col(df, ["Oleo_diesel_nivel", "Var_Oleo_diesel", "Oleo_diesel", "Diesel"])
    col_cambio = candidate_col(df, ["Cambio", "Câmbio", "Taxa_Cambio"])
    col_petroleo = candidate_col(df, ["Preco_Barril", "Preço_Barril", "Brent", "Petroleo", "Petróleo"])
    col_atividade = candidate_col(df, ["Atividade", "IBC_BR", "IBC-Br", "IBC"])
    col_expectativa = candidate_col(df, ["Expectativa_Inflacao", "espectativa_inflacao", "Expectativa Focus", "Focus_Inflacao_12m"])
    col_meta = candidate_col(
        df,
        ["Meta_Inflacao", "meta_inflacao", "Meta Inflacao", "Meta de Inflacao", "Meta_Inflação", "Meta de Inflação", "Meta"],
        required=False,
    )

    # Monta base de trabalho.
    work = pd.DataFrame(index=df.index)

    # IPCA e combustíveis: se vierem como variação mensal, o código cria índice base 100.
    work["IPCA_Geral_nivel"] = to_numeric_br(df[col_ipca_geral])
    if col_ipca_geral == "Var_IPCA_Brasil":
        work["IPCA_Geral_nivel"] = build_index_from_var(df[col_ipca_geral])

    work["IPCA_Trans_nivel"] = to_numeric_br(df[col_ipca_trans])
    if col_ipca_trans.lower().startswith("var_"):
        work["IPCA_Trans_nivel"] = build_index_from_var(df[col_ipca_trans])

    work["GasolinaA_nivel"] = to_numeric_br(df[col_gasolinaA])
    # Gasolina A normalmente já vem como nível. Não transforma em índice automaticamente.

    work["Gasolina_nivel"] = to_numeric_br(df[col_gasolina])
    if col_gasolina.lower().startswith("var_"):
        work["Gasolina_nivel"] = build_index_from_var(df[col_gasolina])

    work["Etanol_nivel"] = to_numeric_br(df[col_etanol])
    if col_etanol.lower().startswith("var_"):
        work["Etanol_nivel"] = build_index_from_var(df[col_etanol])

    work["Oleo_diesel_nivel"] = to_numeric_br(df[col_diesel])
    if col_diesel.lower().startswith("var_"):
        work["Oleo_diesel_nivel"] = build_index_from_var(df[col_diesel])

    work["Cambio"] = to_numeric_br(df[col_cambio])
    work["Preco_Barril"] = to_numeric_br(df[col_petroleo])
    work["Atividade"] = to_numeric_br(df[col_atividade])
    work["Expectativa_Inflacao"] = to_numeric_br(df[col_expectativa])

    # Selic diária -> média mensal -> índice base 100.
    selic_m, selic_arquivo, selic_col = load_selic_meta_mensal(SELIC_DAILY_FILE, work.index)
    work["Selic_Meta_Mensal"] = selic_m["Selic_Meta_Mensal"]
    work["Selic"] = build_index_from_rate_level(work["Selic_Meta_Mensal"], base=100.0)

    # Meta de inflação: exógena em base 100.
    if col_meta is not None:
        work["Meta_Inflacao_Original"] = to_numeric_br(df[col_meta])
        work["Meta_Inflacao"] = build_index_from_rate_level(work["Meta_Inflacao_Original"], base=100.0)
    else:
        # Se não houver meta no arquivo, cria NaN. Depois ela será removida dos controles.
        work["Meta_Inflacao_Original"] = np.nan
        work["Meta_Inflacao"] = np.nan

    # Stringency mensal.
    stringency = load_stringency_monthly(STRINGENCY_FILE, work.index)
    work["Stringency"] = stringency["Stringency"]

    # Logs e diferenças logarítmicas.
    level_vars = [
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
        "Meta_Inflacao",
    ]

    for c in level_vars:
        work[f"LN_{c}"] = safe_log(work[c])
        work[f"DLN_{c}"] = work[f"LN_{c}"].diff()

    # Nomes curtos usados no modelo.
    work["DLN_IPCA_Geral"] = work["DLN_IPCA_Geral_nivel"]
    work["DLN_IPCA_Transporte"] = work["DLN_IPCA_Trans_nivel"]
    work["DLN_GasolinaA"] = work["DLN_GasolinaA_nivel"]
    work["DLN_Gasolina"] = work["DLN_Gasolina_nivel"]
    work["DLN_Etanol"] = work["DLN_Etanol_nivel"]
    work["DLN_Oleo_diesel"] = work["DLN_Oleo_diesel_nivel"]

    info = {
        "data_col": data_col,
        "selic_arquivo": selic_arquivo,
        "selic_coluna": selic_col,
        "col_meta": col_meta,
        "colunas_usadas": {
            "IPCA_Geral": col_ipca_geral,
            "IPCA_Transporte": col_ipca_trans,
            "GasolinaA": col_gasolinaA,
            "Gasolina": col_gasolina,
            "Etanol": col_etanol,
            "Oleo_diesel": col_diesel,
            "Cambio": col_cambio,
            "Preco_Barril": col_petroleo,
            "Atividade": col_atividade,
            "Expectativa_Inflacao": col_expectativa,
            "Meta_Inflacao": col_meta,
        },
    }
    return work, info


# ======================================================
# LOCAL PROJECTIONS
# ======================================================
def montar_lags(df: pd.DataFrame, vars_lag: list[str], lags: int) -> pd.DataFrame:
    blocos = []
    for var in vars_lag:
        if var not in df.columns:
            continue
        for L in range(1, lags + 1):
            blocos.append(df[var].shift(L).rename(f"{var}_L{L}"))
    if not blocos:
        return pd.DataFrame(index=df.index)
    return pd.concat(blocos, axis=1)


def preparar_choque(df: pd.DataFrame, shock_var: str) -> pd.Series:
    shock = df[shock_var].copy()
    if PADRONIZAR_CHOQUE:
        sd = shock.dropna().std()
        if sd and np.isfinite(sd) and sd > 0:
            shock = shock / sd
    return shock.rename("CHOQUE")


def estimar_lp(
    df: pd.DataFrame,
    y_var: str,
    shock_var: str,
    controles_contemporaneos: list[str],
    exog_cols: list[str],
    vars_lag: list[str],
    horizonte_max: int,
    acumulada: bool,
    min_obs: int = 45,
) -> pd.DataFrame:
    """
    Estima uma Local Projection para h = 0,...,H.

    Se acumulada=False:
        y_{t+h} = alpha_h + beta_h choque_t + controles + erro

    Se acumulada=True:
        sum_{j=0}^{h} y_{t+j} = alpha_h + beta_h choque_t + controles + erro
    """
    resultados = []
    zcrit = 1.96 if INTERVALO_CONFIANCA == 0.95 else 1.645

    shock = preparar_choque(df, shock_var)
    lags_df = montar_lags(df, vars_lag, LAGS_CONTROLES)

    controles_existentes = [c for c in controles_contemporaneos if c in df.columns and c != shock_var]
    exog_existentes = [c for c in exog_cols if c in df.columns]

    for h in range(0, horizonte_max + 1):
        if acumulada:
            y_h = sum(df[y_var].shift(-j) for j in range(0, h + 1)).rename("Y")
        else:
            y_h = df[y_var].shift(-h).rename("Y")

        X_parts = [shock]
        if controles_existentes:
            X_parts.append(df[controles_existentes])
        if exog_existentes:
            X_parts.append(df[exog_existentes])
        if not lags_df.empty:
            X_parts.append(lags_df)

        X = pd.concat(X_parts, axis=1)
        X = drop_constant_or_duplicate_columns(X)
        X = sm.add_constant(X, has_constant="add")

        reg = pd.concat([y_h, X], axis=1).dropna()
        if len(reg) < min_obs or "CHOQUE" not in reg.columns:
            resultados.append({
                "h": h,
                "beta": np.nan,
                "se": np.nan,
                "t": np.nan,
                "pvalue": np.nan,
                "ci_inf": np.nan,
                "ci_sup": np.nan,
                "nobs": len(reg),
                "r2": np.nan,
            })
            continue

        y = reg["Y"]
        Xreg = reg.drop(columns=["Y"])

        # HAC/Newey-West. Para horizonte maior, usa maxlags=h; para h=0, usa 1.
        hac_lags = max(1, h)
        try:
            modelo = sm.OLS(y, Xreg).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
            beta = float(modelo.params.get("CHOQUE", np.nan))
            se = float(modelo.bse.get("CHOQUE", np.nan))
            tstat = float(modelo.tvalues.get("CHOQUE", np.nan))
            pvalue = float(modelo.pvalues.get("CHOQUE", np.nan))
            r2 = float(modelo.rsquared)
        except Exception:
            beta, se, tstat, pvalue, r2 = np.nan, np.nan, np.nan, np.nan, np.nan

        resultados.append({
            "h": h,
            "beta": beta,
            "se": se,
            "t": tstat,
            "pvalue": pvalue,
            "ci_inf": beta - zcrit * se if pd.notna(beta) and pd.notna(se) else np.nan,
            "ci_sup": beta + zcrit * se if pd.notna(beta) and pd.notna(se) else np.nan,
            "nobs": len(reg),
            "r2": r2,
        })

    return pd.DataFrame(resultados)


def plot_lp(df_res: pd.DataFrame, title: str, outfile: Path) -> None:
    if df_res.empty or df_res["beta"].dropna().empty:
        return
    x = df_res["h"].to_numpy(dtype=float)
    beta = df_res["beta"].to_numpy(dtype=float)
    ci_inf = df_res["ci_inf"].to_numpy(dtype=float)
    ci_sup = df_res["ci_sup"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, beta, marker="o", linewidth=1.5)
    ax.fill_between(x, ci_inf, ci_sup, alpha=0.2)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("Resposta estimada")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)


def definir_modelos_lp():
    combustiveis = {
        "GasolinaA": "DLN_GasolinaA",
        "Gasolina": "DLN_Gasolina",
        "Etanol": "DLN_Etanol",
        "Oleo_diesel": "DLN_Oleo_diesel",
    }
    ipcas = {
        "IPCA_Geral": "DLN_IPCA_Geral",
        "IPCA_Transporte": "DLN_IPCA_Transporte",
    }

    modelos = []

    # 1) Petróleo -> combustíveis.
    for nome_comb, var_comb in combustiveis.items():
        modelos.append({
            "grupo": "petroleo_para_combustivel",
            "nome": f"Petroleo_para_{nome_comb}",
            "y_var": var_comb,
            "shock_var": "DLN_Preco_Barril",
            "controles": ["DLN_Cambio"],
        })

    # 2) Petróleo -> IPCA, controlando pelo combustível.
    for nome_comb, var_comb in combustiveis.items():
        for nome_ipca, var_ipca in ipcas.items():
            modelos.append({
                "grupo": "petroleo_para_ipca_controlando_combustivel",
                "nome": f"Petroleo_para_{nome_ipca}_controle_{nome_comb}",
                "y_var": var_ipca,
                "shock_var": "DLN_Preco_Barril",
                "controles": ["DLN_Cambio", var_comb],
            })

    # 3) Combustível -> IPCA, controlando por petróleo e câmbio.
    for nome_comb, var_comb in combustiveis.items():
        for nome_ipca, var_ipca in ipcas.items():
            modelos.append({
                "grupo": "combustivel_para_ipca",
                "nome": f"{nome_comb}_para_{nome_ipca}",
                "y_var": var_ipca,
                "shock_var": var_comb,
                "controles": ["DLN_Preco_Barril", "DLN_Cambio"],
            })

    return modelos


def rodar_subamostra(work_full: pd.DataFrame, nome_sub: str, data_ini: str, data_fim: str) -> None:
    outdir = OUTDIR_BASE / nome_sub
    preparar_pastas(outdir)

    mask = (work_full.index >= pd.Timestamp(data_ini)) & (work_full.index <= pd.Timestamp(data_fim))
    work = work_full.loc[mask].copy()

    if work.empty:
        print(f"Subamostra {nome_sub} vazia. Pulando.")
        return

    # Exógenas comuns: dummies mensais, stringency, meta de inflação base 100 e Selic em log.
    dummies = create_month_dummies(work.index)
    work = pd.concat([work, dummies], axis=1)

    exog_cols = list(dummies.columns)
    if work["Stringency"].notna().sum() > 0:
        exog_cols.append("Stringency")
    if work["Meta_Inflacao"].notna().sum() > 0:
        exog_cols.append("LN_Meta_Inflacao")

    # Selic entra em nível logarítmico, sem primeira diferença.
    if work["LN_Selic"].notna().sum() > 0:
        exog_cols.append("LN_Selic")

    # Controles defasados principais. Inclui as variáveis de transmissão e inflação.
    vars_lag = [
        "DLN_Preco_Barril",
        "DLN_Cambio",
        "DLN_GasolinaA",
        "DLN_Gasolina",
        "DLN_Etanol",
        "DLN_Oleo_diesel",
        "DLN_Atividade",
        "DLN_Expectativa_Inflacao",
        "DLN_IPCA_Geral",
        "DLN_IPCA_Transporte",
        "LN_Selic",
    ]

    # Séries de conferência.
    controles_export = [
        "Selic_Meta_Mensal",
        "Selic",
        "LN_Selic",
        "Meta_Inflacao_Original",
        "Meta_Inflacao",
        "LN_Meta_Inflacao",
        "Stringency",
    ]
    cols_export = [c for c in controles_export if c in work.columns]
    work[cols_export].to_excel(outdir / "tabelas" / "conferencia_selic_meta_stringency.xlsx")

    for c in cols_export:
        plot_series(work[c], f"{c} - {nome_sub}", outdir / "graficos" / "03_series_controle" / f"{c}.png")

    modelos = definir_modelos_lp()
    todos_resultados = []

    for spec in modelos:
        nome = spec["nome"]
        grupo = spec["grupo"]
        y_var = spec["y_var"]
        shock_var = spec["shock_var"]
        controles = spec["controles"] + ["DLN_Atividade", "DLN_Expectativa_Inflacao"]

        if y_var not in work.columns or shock_var not in work.columns:
            print(f"Modelo {nome_sub} - {nome}: variável ausente. Pulando.")
            continue

        # LP mensal.
        res_mensal = estimar_lp(
            df=work,
            y_var=y_var,
            shock_var=shock_var,
            controles_contemporaneos=controles,
            exog_cols=exog_cols,
            vars_lag=vars_lag,
            horizonte_max=HORIZONTE_MAX,
            acumulada=False,
        )
        res_mensal["subamostra"] = nome_sub
        res_mensal["grupo"] = grupo
        res_mensal["modelo"] = nome
        res_mensal["tipo_resposta"] = "mensal"
        res_mensal["y_var"] = y_var
        res_mensal["shock_var"] = shock_var

        res_mensal.to_excel(outdir / "tabelas" / f"lp_mensal_{nome}.xlsx", index=False)
        plot_lp(
            res_mensal,
            title=f"LP mensal - {nome_sub} - {nome}",
            outfile=outdir / "graficos" / "01_lp_mensal" / f"lp_mensal_{nome}.png",
        )

        # LP acumulada.
        res_acum = estimar_lp(
            df=work,
            y_var=y_var,
            shock_var=shock_var,
            controles_contemporaneos=controles,
            exog_cols=exog_cols,
            vars_lag=vars_lag,
            horizonte_max=HORIZONTE_MAX,
            acumulada=True,
        )
        res_acum["subamostra"] = nome_sub
        res_acum["grupo"] = grupo
        res_acum["modelo"] = nome
        res_acum["tipo_resposta"] = "acumulada"
        res_acum["y_var"] = y_var
        res_acum["shock_var"] = shock_var

        res_acum.to_excel(outdir / "tabelas" / f"lp_acumulada_{nome}.xlsx", index=False)
        plot_lp(
            res_acum,
            title=f"LP acumulada - {nome_sub} - {nome}",
            outfile=outdir / "graficos" / "02_lp_acumulada" / f"lp_acumulada_{nome}.png",
        )

        todos_resultados.append(res_mensal)
        todos_resultados.append(res_acum)

    if todos_resultados:
        painel = pd.concat(todos_resultados, axis=0, ignore_index=True)
        painel.to_excel(outdir / "tabelas" / "painel_resultados_local_projections.xlsx", index=False)

        # Tabela-resumo em horizontes centrais.
        resumo = painel[painel["h"].isin([0, 3, 6, 12, 18, 24])].copy()
        resumo = resumo[[
            "subamostra", "grupo", "modelo", "tipo_resposta", "h", "beta", "se", "pvalue",
            "ci_inf", "ci_sup", "nobs", "r2", "y_var", "shock_var"
        ]]
        resumo.to_excel(outdir / "tabelas" / "resumo_horizontes_principais.xlsx", index=False)

    print(f"Concluído: {nome_sub}")
    print(f"Saída: {outdir.resolve()}")


# ======================================================
# MAIN
# ======================================================
def main():
    OUTDIR_BASE.mkdir(parents=True, exist_ok=True)
    work, info = carregar_preparar_base()

    # Salva informações do mapeamento usado.
    info_path = OUTDIR_BASE / "mapeamento_colunas_usadas.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("MAPEAMENTO USADO NO MODELO LP\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Coluna de data detectada: {info['data_col']}\n")
        f.write(f"Arquivo Selic usado: {info['selic_arquivo']}\n")
        f.write(f"Coluna Selic diária usada: {info['selic_coluna']}\n")
        f.write(f"Coluna de meta de inflação usada: {info['col_meta']}\n\n")
        for k, v in info["colunas_usadas"].items():
            f.write(f"{k}: {v}\n")

        f.write("\nESPECIFICAÇÃO\n")
        f.write("=" * 60 + "\n")
        f.write(f"Horizonte máximo: {HORIZONTE_MAX}\n")
        f.write(f"Lags dos controles: {LAGS_CONTROLES}\n")
        f.write(f"Choque padronizado: {PADRONIZAR_CHOQUE}\n")
        f.write("Erros-padrão: HAC/Newey-West\n")
        f.write("Selic: média mensal da meta diária, transformada em base 100 e usada como LN_Selic.\n")
        f.write("Meta de inflação: transformada em base 100 e usada como LN_Meta_Inflacao exógena.\n")

    # Salva a base transformada para auditoria.
    work.to_excel(OUTDIR_BASE / "base_transformada_lp_modelo1.xlsx")

    print("Mapeamento usado:")
    for k, v in info["colunas_usadas"].items():
        print(f"  {k}: {v}")
    print(f"Arquivo Selic usado: {info['selic_arquivo']}")

    for nome_sub, data_ini, data_fim in SUBAMOSTRAS:
        rodar_subamostra(work, nome_sub, data_ini, data_fim)

    print("Todas as Local Projections foram processadas.")
    print(f"Resultados em: {OUTDIR_BASE.resolve()}")


if __name__ == "__main__":
    main()
