# -*- coding: utf-8 -*-
"""
petroleo_lp_modelo8.py

MODELO 8 - LOCAL PROJECTIONS DO TCC

Objetivo:
    Organizar a estratégia empírica em torno do canal:
    Brent em reais -> combustíveis -> IPCA.

Hierarquia do Modelo 8:
    Bloco A, principal: Brent em reais -> combustíveis.
    Bloco B, principal: combustíveis -> IPCA Geral e IPCA Transportes.
    Bloco C, complementar: Brent em reais -> IPCA Geral e IPCA Transportes.
    Bloco D, Petrobras: pré-outubro/2016 vs pós-outubro/2016.
    Bloco E, robustez: 6 defasagens.
    Bloco F, robustez de identificação: LP-IV com Oil Supply News Shock.

Observações:
    O choque principal é dln_petroleo_brl, isto é, Brent em reais.
    Oil Supply News Shock NÃO é o choque principal. Ele é usado apenas como instrumento no LP-IV.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

ARQUIVO_IPCA = "IPCA.xlsx"
ARQUIVO_SELIC = "STP-20260429165342557.csv"
ARQUIVO_STRINGENCY = "Stringency_index.csv"
ARQUIVO_OIL_NEWS = "oilSupplyNewsShocks_2025M06.xlsx"

DATA_INICIO = "2003-01-01"

H_MAX = 24
H_PRINCIPAL = 12
LAGS_PRINCIPAL = 3
HORIZONTES_RESUMO = [3, 6, 12, 24]

CONF = 0.90
Z = stats.norm.ppf(1 - (1 - CONF) / 2)
MIN_OBS = 60

OUTPUT_DIR = Path("output_petroleo_lp_modelo8")
OUTPUT_TABELAS = OUTPUT_DIR / "tabelas"
OUTPUT_GRAFICOS = OUTPUT_DIR / "graficos"
OUTPUT_RESUMOS = OUTPUT_DIR / "resumos"

for p in [OUTPUT_DIR, OUTPUT_TABELAS, OUTPUT_GRAFICOS, OUTPUT_RESUMOS]:
    p.mkdir(parents=True, exist_ok=True)

# Padrão do Modelo 5: identificação flexível de colunas
CONFIG_COLUNAS = {
    "data": ["Data", "data", "DATE", "Date"],

    "petroleo": ["Preco_Barril", "Petroleo", "Petróleo", "Brent", "DCOILBRENTEU", "preco_barril"],
    "cambio": ["Cambio", "cambio", "USDBRL", "Dolar", "Dólar", "Taxa_Cambio"],

    "gasolina_refinaria": [
        "GasolinaABrasil_media", "Var_GasolinaABrasil_media", "GasolinaABrasil_media_nivel",
        "GasolinaA", "GasolinaA_nivel", "Gasolina_A", "Gasolina_Refinaria", "Preco_Refinaria"
    ],
    "gasolina": ["Gasolina", "Var_Gasolina", "Gasolina_nivel", "Gasolina_consumidor", "Preco_Gasolina"],
    "etanol": ["Etanol", "Var_Etanol", "Etanol_nivel", "Preco_Etanol"],
    "diesel": ["Oleo_diesel", "Var_Oleo_diesel", "Oleo_diesel_nivel", "Diesel", "Preco_Diesel"],

    "ipca_geral": ["IPCA_Geral_nivel", "IPCA_Brasil", "Var_IPCA_Brasil", "Var_IPCA_Geral", "IPCA_Geral", "IPCA"],
    "ipca_transporte": ["IPCA_Trans_nivel", "Var_IPCA_trans", "Var_IPCA_Trans", "IPCA_Transporte", "IPCA_Trans"],

    "atividade": ["Atividade", "IBC_BR", "IBC_Br", "IBC-BR", "IBC"],
    "selic": ["Selic", "SELIC", "Meta_Selic", "selic", "Selic.1", "selic_diaria_media_mensal"],
    "expectativa": [
        "Expectativa_Inflacao", "Expectativa_Inflação", "Focus_IPCA_12m", "IPCA_Focus_12m",
        "Expectativa", "espectativa_inflacao", "Espectativa_Inflacao", "Espectativa_Inflação"
    ],
    "stringency": ["Stringency", "stringency", "Stringency_Index", "Oxford_Stringency", "stringency_externo"]
}

# =============================================================================
# 2. FUNÇÕES DE APOIO
# =============================================================================

def localizar_arquivo(nome):
    candidatos = [
        Path(nome),
        Path.cwd() / nome,
        Path(__file__).resolve().parent / nome,
        Path.home() / "OneDrive" / "Documentos" / "TCC" / nome,
        Path.home() / "OneDrive" / "Documentos" / "TCC_python" / nome,
        Path.home() / "Documents" / "TCC" / nome,
        Path.home() / "Documents" / "TCC_python" / nome,
    ]

    for c in candidatos:
        if c.exists():
            return str(c)

    raise FileNotFoundError(f"Arquivo não encontrado: {nome}")


def normalizar_nome(x):
    x = str(x).strip().lower()
    mapa = {
        "á": "a", "à": "a", "ã": "a", "â": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u",
        "ç": "c"
    }
    for a, b in mapa.items():
        x = x.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "", x)


def encontrar_coluna(df, candidatos, obrigatoria=False, nome_logico=""):
    cols = list(df.columns)

    for c in candidatos:
        if c in cols:
            return c

    lower = {str(c).lower(): c for c in cols}
    for c in candidatos:
        if str(c).lower() in lower:
            return lower[str(c).lower()]

    norm_cols = {normalizar_nome(c): c for c in cols}
    for c in candidatos:
        nc = normalizar_nome(c)
        if nc in norm_cols:
            return norm_cols[nc]

    for c in candidatos:
        nc = normalizar_nome(c)
        for k, v in norm_cols.items():
            if nc and (nc in k or k in nc):
                return v

    if obrigatoria:
        raise ValueError(
            f"Não encontrei coluna obrigatória para {nome_logico}. "
            f"Candidatos: {candidatos}. Colunas disponíveis: {cols}"
        )

    return None


def to_num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", ".", regex=False), errors="coerce")


def dlog100(s):
    s = to_num(s)
    s = s.where(s > 0)
    return 100 * np.log(s).diff()


def usar_variacao_ou_dlog(s):
    """
    Se a série parece nível/índice/preço, usa dlog*100.
    Se já parece variação mensal, preserva.
    """
    s = to_num(s)
    med = s.dropna().abs().median()
    if pd.notna(med) and med > 20:
        return dlog100(s)
    return s


def padronizar(s):
    s = to_num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd


def hac_maxlags(h):
    return max(3, h + 1)


def criar_lags(df, var, n_lags, prefix=None):
    prefix = prefix or var
    out = []
    for L in range(1, n_lags + 1):
        c = f"{prefix}_lag{L}"
        df[c] = df[var].shift(L)
        out.append(c)
    return out


def montar_y_h(temp, y, h, acumulada=True):
    if acumulada:
        cols = []
        for j in range(h + 1):
            c = f"{y}_lead{j}"
            temp[c] = temp[y].shift(-j)
            cols.append(c)
        temp[f"y_h{h}"] = temp[cols].sum(axis=1, min_count=h + 1)
    else:
        temp[f"y_h{h}"] = temp[y].shift(-h)
    return temp


class ResultadoHAC:
    def __init__(self, params, bse, tvalues, pvalues, cov, nobs, rsquared):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self.cov_params_matrix = cov
        self.nobs = nobs
        self.rsquared = rsquared

    def wald_test(self, R):
        R = np.asarray(R, dtype=float)
        beta = self.params.values.reshape(-1, 1)
        cov = self.cov_params_matrix.values
        diff = R @ beta
        middle = R @ cov @ R.T
        stat = float(diff.T @ np.linalg.pinv(middle) @ diff)
        pvalue = float(1 - stats.chi2.cdf(stat, R.shape[0]))

        class WaldResult:
            def __init__(self, statistic, pvalue):
                self.statistic = statistic
                self.pvalue = pvalue

        return WaldResult(stat, pvalue)


def ajustar_ols_hac(Y, X, h):
    X = sm.add_constant(X, has_constant="add")
    X = X.apply(pd.to_numeric, errors="coerce")
    Y = pd.to_numeric(Y, errors="coerce")

    dados = pd.concat([Y.rename("Y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    Yv = dados["Y"].to_numpy(dtype=float)
    Xdf = dados.drop(columns=["Y"])
    Xv = Xdf.to_numpy(dtype=float)

    nobs, k = Xv.shape
    if nobs <= k + 5:
        raise ValueError("Observações insuficientes para OLS HAC.")

    XtX = Xv.T @ Xv
    XtX_inv = np.linalg.pinv(XtX, rcond=1e-8)
    beta = XtX_inv @ (Xv.T @ Yv)

    resid = Yv - Xv @ beta
    xu = Xv * resid[:, None]

    S = xu.T @ xu
    L = min(hac_maxlags(h), nobs - 1)

    for ell in range(1, L + 1):
        peso = 1.0 - ell / (L + 1.0)
        Gamma = xu[ell:].T @ xu[:-ell]
        S += peso * (Gamma + Gamma.T)

    cov = XtX_inv @ S @ XtX_inv
    diag = np.diag(cov)
    diag = np.where(diag < 0, np.nan, diag)
    bse = np.sqrt(diag)

    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / bse
        pvals = 2 * (1 - stats.norm.cdf(np.abs(tvals)))

    ssr = float(np.sum(resid ** 2))
    tss = float(np.sum((Yv - Yv.mean()) ** 2))
    r2 = np.nan if tss == 0 else 1 - ssr / tss

    idx = Xdf.columns
    return ResultadoHAC(
        params=pd.Series(beta, index=idx),
        bse=pd.Series(bse, index=idx),
        tvalues=pd.Series(tvals, index=idx),
        pvalues=pd.Series(pvals, index=idx),
        cov=pd.DataFrame(cov, index=idx, columns=idx),
        nobs=int(nobs),
        rsquared=r2
    )


# =============================================================================
# 3. LEITURA DAS BASES
# =============================================================================

def carregar_selic_diaria():
    try:
        arquivo = localizar_arquivo(ARQUIVO_SELIC)
    except FileNotFoundError:
        print("[AVISO] Arquivo da Selic diária não encontrado. Usarei Selic da base principal, se houver.")
        return None

    df = pd.read_csv(arquivo, sep=";", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    col_data = encontrar_coluna(df, ["Data", "data", "Date"], True, "data_selic")
    col_valor = [c for c in df.columns if c != col_data][0]

    df[col_data] = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)
    df[col_valor] = to_num(df[col_valor])

    df = df.dropna(subset=[col_data]).copy()
    df["Data"] = df[col_data].dt.to_period("M").dt.to_timestamp()

    out = df.groupby("Data", as_index=False)[col_valor].mean()
    out = out.rename(columns={col_valor: "selic_diaria_media_mensal"})

    print("[OK] Selic diária importada e agregada para frequência mensal.")
    return out


def carregar_stringency():
    try:
        arquivo = localizar_arquivo(ARQUIVO_STRINGENCY)
    except FileNotFoundError:
        print("[AVISO] Stringency_index.csv não encontrado. Controle ficará zero.")
        return None

    df = pd.read_csv(arquivo, sep=";", encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "CountryCode" in df.columns:
        df = df[df["CountryCode"].astype(str).str.upper().eq("BRA")].copy()

    if "Jurisdiction" in df.columns:
        nat = df[df["Jurisdiction"].astype(str).str.upper().eq("NAT_TOTAL")].copy()
        if not nat.empty:
            df = nat

    col_data = encontrar_coluna(df, ["Date", "Data", "date"], True, "data_stringency")
    col_str = encontrar_coluna(
        df,
        ["StringencyIndex_Average", "StringencyIndex_Average_ForDisplay", "Stringency", "Stringency_Index"],
        True,
        "stringency"
    )

    data_parse = pd.to_datetime(df[col_data].astype(str), format="%Y%m%d", errors="coerce")
    if data_parse.isna().all():
        data_parse = pd.to_datetime(df[col_data], errors="coerce")

    df[col_data] = data_parse
    df[col_str] = to_num(df[col_str])
    df = df.dropna(subset=[col_data]).copy()

    df["Data"] = df[col_data].dt.to_period("M").dt.to_timestamp()
    out = df.groupby("Data", as_index=False)[col_str].mean()
    out = out.rename(columns={col_str: "stringency_externo"})

    print("[OK] Stringency Index importado.")
    return out


def parse_data_mensal_choque(x):
    if pd.isna(x):
        return pd.NaT

    if isinstance(x, pd.Timestamp):
        return x.to_period("M").to_timestamp()

    sx = str(x).strip()

    if "M" in sx and len(sx) >= 7:
        try:
            ano, mes = sx.split("M")
            return pd.Timestamp(int(ano), int(mes), 1)
        except Exception:
            pass

    return pd.to_datetime(sx, errors="coerce")


def carregar_choque_oferta():
    try:
        arquivo = localizar_arquivo(ARQUIVO_OIL_NEWS)
    except FileNotFoundError:
        print("[AVISO] Oil Supply News Shock não encontrado. LP-IV não será estimada.")
        return None

    xls = pd.ExcelFile(arquivo)
    aba = "Monthly" if "Monthly" in xls.sheet_names else xls.sheet_names[0]

    df = pd.read_excel(arquivo, sheet_name=aba)
    df.columns = [str(c).strip() for c in df.columns]

    col_data = encontrar_coluna(df, ["Date", "Data", "date"], True, "data_choque")
    col_news = encontrar_coluna(
        df,
        ["Oil supply news shock", "oil supply news shock", "Oil Supply News Shock"],
        True,
        "oil_supply_news_shock"
    )

    df["Data"] = df[col_data].apply(parse_data_mensal_choque)
    df["oil_supply_news_shock"] = to_num(df[col_news])
    df = df.dropna(subset=["Data"]).copy()

    df["Data"] = df["Data"].dt.to_period("M").dt.to_timestamp()
    df["oil_supply_news_shock_std"] = padronizar(df["oil_supply_news_shock"])

    print("[OK] Oil Supply News Shock importado. Será usado apenas no LP-IV.")
    return df[["Data", "oil_supply_news_shock", "oil_supply_news_shock_std"]]


def carregar_preparar_base():
    print("\n" + "=" * 100)
    print("1) LEITURA E PREPARAÇÃO DA BASE - PADRÃO CONFIG_COLUNAS")
    print("=" * 100)

    arquivo = localizar_arquivo(ARQUIVO_IPCA)
    print(f"Arquivo principal usado: {arquivo}")

    df = pd.read_excel(arquivo)
    df.columns = [str(c).strip() for c in df.columns]

    col_data = encontrar_coluna(df, CONFIG_COLUNAS["data"], True, "data")
    df[col_data] = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[col_data]).sort_values(col_data).reset_index(drop=True)
    df = df.rename(columns={col_data: "Data"})
    df["Data"] = df["Data"].dt.to_period("M").dt.to_timestamp()

    for extra in [carregar_selic_diaria(), carregar_stringency(), carregar_choque_oferta()]:
        if extra is not None:
            df = pd.merge(df, extra, on="Data", how="left")

    mapa = {}
    obrigatorias = {
        "petroleo", "cambio", "gasolina_refinaria", "gasolina",
        "etanol", "diesel", "ipca_geral", "ipca_transporte", "atividade"
    }

    for nome, candidatos in CONFIG_COLUNAS.items():
        if nome == "data":
            continue
        mapa[nome] = encontrar_coluna(df, candidatos, nome in obrigatorias, nome)

    print("\nColunas identificadas:")
    for k, v in mapa.items():
        print(f"- {k}: {v}")

    # Brent em reais
    df["petroleo_usd_nivel"] = to_num(df[mapa["petroleo"]])
    df["cambio_nivel"] = to_num(df[mapa["cambio"]])
    df["petroleo_brl_nivel"] = df["petroleo_usd_nivel"] * df["cambio_nivel"]

    df["dln_petroleo_brl"] = dlog100(df["petroleo_brl_nivel"])
    df["dln_petroleo_usd"] = dlog100(df["petroleo_usd_nivel"])
    df["dln_cambio"] = dlog100(df["cambio_nivel"])

    # Combustíveis
    df["dln_gasolina_refinaria"] = usar_variacao_ou_dlog(df[mapa["gasolina_refinaria"]])
    df["dln_gasolina"] = usar_variacao_ou_dlog(df[mapa["gasolina"]])
    df["dln_etanol"] = usar_variacao_ou_dlog(df[mapa["etanol"]])
    df["dln_diesel"] = usar_variacao_ou_dlog(df[mapa["diesel"]])

    # IPCA
    df["ipca_geral_mensal"] = usar_variacao_ou_dlog(df[mapa["ipca_geral"]])
    df["ipca_transporte_mensal"] = usar_variacao_ou_dlog(df[mapa["ipca_transporte"]])

    # Controles
    df["dln_atividade"] = dlog100(df[mapa["atividade"]])

    if "selic_diaria_media_mensal" in df.columns:
        df["selic_controle"] = to_num(df["selic_diaria_media_mensal"])
    elif mapa.get("selic") is not None:
        df["selic_controle"] = to_num(df[mapa["selic"]])
    else:
        df["selic_controle"] = 0.0
        print("[AVISO] Selic não encontrada. Controle ficará zero.")

    if mapa.get("expectativa") is not None:
        df["expectativa_controle"] = to_num(df[mapa["expectativa"]])
        print(f"[OK] Expectativa usada: {mapa['expectativa']}")
    else:
        df["expectativa_controle"] = 0.0
        print("[AVISO] Expectativa não encontrada. Controle ficará zero.")

    if "stringency_externo" in df.columns:
        df["stringency_controle"] = to_num(df["stringency_externo"]).fillna(0.0)
    elif mapa.get("stringency") is not None:
        df["stringency_controle"] = to_num(df[mapa["stringency"]]).fillna(0.0)
    else:
        df["stringency_controle"] = 0.0

    # Dummies mensais
    df["mes"] = df["Data"].dt.month
    dummies = pd.get_dummies(df["mes"], prefix="mes", drop_first=True, dtype=float)
    df = pd.concat([df, dummies], axis=1)
    dummy_cols = list(dummies.columns)

    # Regime Petrobras Modelo 8
    df["regime_pre_out2016"] = ((df["Data"] >= "2003-01-01") & (df["Data"] <= "2016-09-01")).astype(int)
    df["regime_pos_out2016"] = (df["Data"] >= "2016-10-01").astype(int)

    df = df[df["Data"] >= pd.to_datetime(DATA_INICIO)].copy().sort_values("Data").reset_index(drop=True)

    df.to_excel(OUTPUT_TABELAS / "base_modelo8_transformada.xlsx", index=False)

    print("\nBase final:")
    print(f"- Período: {df['Data'].min().date()} até {df['Data'].max().date()}")
    print(f"- Observações: {len(df)}")

    return df, dummy_cols, mapa


# =============================================================================
# 4. LOCAL PROJECTIONS
# =============================================================================

def controles_macro(dummy_cols):
    return ["dln_atividade", "selic_controle", "expectativa_controle", "stringency_controle"] + dummy_cols


def local_projection(df, y, shock, controls=None, lags_n=3, h_max=24,
                     horizontes=None, acumulada=True, nome_modelo="lp"):
    controls = controls or []
    base = df.copy()

    shock_std = f"{shock}_std_modelo"
    base[shock_std] = padronizar(base[shock])

    regressores_fixos = []
    regressores_fixos += criar_lags(base, y, lags_n, y)
    regressores_fixos += criar_lags(base, shock, lags_n, shock)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            if not str(c).startswith("mes_"):
                regressores_fixos += criar_lags(base, c, lags_n, c)

    if horizontes is None:
        horizontes = range(0, h_max + 1)

    resultados = []

    for h in horizontes:
        temp = montar_y_h(base.copy(), y, h, acumulada)
        X_cols = [shock_std] + [c for c in regressores_fixos if c in temp.columns]

        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(MIN_OBS, len(X_cols) + 10):
            resultados.append({
                "modelo": nome_modelo, "impulso": shock, "resposta": y,
                "h": h, "lags": lags_n, "coef": np.nan, "se": np.nan,
                "pvalor": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "significativo_90": False, "nobs": len(temp_reg)
            })
            continue

        try:
            res = ajustar_ols_hac(temp_reg[f"y_h{h}"], temp_reg[X_cols], h)
            coef = res.params.get(shock_std, np.nan)
            se = res.bse.get(shock_std, np.nan)
            ci_low = coef - Z * se
            ci_high = coef + Z * se

            resultados.append({
                "modelo": nome_modelo,
                "impulso": shock,
                "resposta": y,
                "h": h,
                "lags": lags_n,
                "coef": coef,
                "se": se,
                "t": res.tvalues.get(shock_std, np.nan),
                "pvalor": res.pvalues.get(shock_std, np.nan),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significativo_90": bool((ci_low > 0) or (ci_high < 0)),
                "nobs": int(res.nobs),
                "r2": res.rsquared,
                "hac_maxlags": hac_maxlags(h)
            })
        except Exception as e:
            print(f"[ERRO] LP {nome_modelo}, h={h}: {e}")

    return pd.DataFrame(resultados)


def local_projection_regimes(df, y, shock, controls=None, lags_n=3,
                             horizontes=None, acumulada=True, nome_modelo="lp_regimes"):
    controls = controls or []
    horizontes = HORIZONTES_RESUMO if horizontes is None else horizontes

    base = df.copy()
    shock_std = f"{shock}_std_modelo"
    base[shock_std] = padronizar(base[shock])

    regimes = {
        "pre_out2016": "regime_pre_out2016",
        "pos_out2016": "regime_pos_out2016"
    }

    interacoes = []
    for nome_regime, col_regime in regimes.items():
        col_int = f"{shock_std}_x_{nome_regime}"
        base[col_int] = base[shock_std] * base[col_regime]
        interacoes.append(col_int)

    regressores_fixos = []
    regressores_fixos += criar_lags(base, y, lags_n, y)
    regressores_fixos += criar_lags(base, shock, lags_n, shock)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            if not str(c).startswith("mes_"):
                regressores_fixos += criar_lags(base, c, lags_n, c)

    resultados = []
    resultados_wald = []

    for h in horizontes:
        temp = montar_y_h(base.copy(), y, h, acumulada)
        X_cols = interacoes + [c for c in regressores_fixos if c in temp.columns]
        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(MIN_OBS, len(X_cols) + 10):
            continue

        try:
            res = ajustar_ols_hac(temp_reg[f"y_h{h}"], temp_reg[X_cols], h)
        except Exception as e:
            print(f"[ERRO] LP regimes {nome_modelo}, h={h}: {e}")
            continue

        for nome_regime in regimes:
            col_int = f"{shock_std}_x_{nome_regime}"
            coef = res.params.get(col_int, np.nan)
            se = res.bse.get(col_int, np.nan)
            ci_low = coef - Z * se
            ci_high = coef + Z * se

            resultados.append({
                "modelo": nome_modelo,
                "impulso": shock,
                "resposta": y,
                "regime": nome_regime,
                "h": h,
                "lags": lags_n,
                "coef": coef,
                "se": se,
                "pvalor": res.pvalues.get(col_int, np.nan),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significativo_90": bool((ci_low > 0) or (ci_high < 0)),
                "nobs": int(res.nobs),
                "hac_maxlags": hac_maxlags(h)
            })

        nomes = list(res.params.index)
        ca = f"{shock_std}_x_pre_out2016"
        cb = f"{shock_std}_x_pos_out2016"

        if ca in nomes and cb in nomes:
            R = np.zeros((1, len(nomes)))
            R[0, nomes.index(ca)] = 1
            R[0, nomes.index(cb)] = -1

            try:
                wt = res.wald_test(R)
                resultados_wald.append({
                    "modelo": nome_modelo,
                    "impulso": shock,
                    "resposta": y,
                    "h": h,
                    "comparacao": "pre_out2016 vs pos_out2016",
                    "diferenca_coef": float(res.params[ca] - res.params[cb]),
                    "wald_stat": float(wt.statistic),
                    "pvalor_wald": float(wt.pvalue),
                    "diferenca_significativa_10": bool(float(wt.pvalue) < 0.10),
                    "nobs": int(res.nobs)
                })
            except Exception:
                pass

    return pd.DataFrame(resultados), pd.DataFrame(resultados_wald)


def local_projection_iv(df, y, endog, instrument, controls=None, lags_n=3,
                        horizontes=None, acumulada=True, nome_modelo="lp_iv"):
    controls = controls or []
    horizontes = HORIZONTES_RESUMO if horizontes is None else horizontes

    if instrument not in df.columns:
        print(f"[AVISO] Instrumento {instrument} não está na base. LP-IV não será estimada.")
        return pd.DataFrame(), pd.DataFrame()

    base = df.copy()
    endog_std = f"{endog}_std_iv"
    instr_std = f"{instrument}_std_iv"

    base[endog_std] = padronizar(base[endog])
    base[instr_std] = padronizar(base[instrument])

    regressores_fixos = []
    regressores_fixos += criar_lags(base, y, lags_n, y)
    regressores_fixos += criar_lags(base, endog, lags_n, endog)
    regressores_fixos += criar_lags(base, instrument, lags_n, instrument)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            if not str(c).startswith("mes_"):
                regressores_fixos += criar_lags(base, c, lags_n, c)

    resultados = []
    primeiros = []

    for h in horizontes:
        temp = montar_y_h(base.copy(), y, h, acumulada)
        X_controls = [c for c in regressores_fixos if c in temp.columns]
        cols = [f"y_h{h}", endog_std, instr_std] + X_controls
        temp_reg = temp[cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(MIN_OBS, len(cols) + 10):
            continue

        try:
            X1 = sm.add_constant(temp_reg[[instr_std] + X_controls], has_constant="add")
            fs = sm.OLS(temp_reg[endog_std], X1).fit(cov_type="HC1")

            try:
                ftest = fs.f_test(f"{instr_std} = 0")
                f_stat = float(ftest.fvalue)
                f_pvalor = float(ftest.pvalue)
            except Exception:
                t_instr = fs.tvalues.get(instr_std, np.nan)
                f_stat = float(t_instr ** 2)
                f_pvalor = float(fs.pvalues.get(instr_std, np.nan))

            temp_reg = temp_reg.copy()
            temp_reg[f"{endog_std}_hat"] = fs.predict(X1)

            res2 = ajustar_ols_hac(
                temp_reg[f"y_h{h}"],
                temp_reg[[f"{endog_std}_hat"] + X_controls],
                h
            )

            coef = res2.params.get(f"{endog_std}_hat", np.nan)
            se = res2.bse.get(f"{endog_std}_hat", np.nan)
            ci_low = coef - Z * se
            ci_high = coef + Z * se

            resultados.append({
                "modelo": nome_modelo,
                "impulso_instrumentado": endog,
                "instrumento": instrument,
                "resposta": y,
                "h": h,
                "lags": lags_n,
                "coef_iv": coef,
                "se_iv": se,
                "pvalor_iv": res2.pvalues.get(f"{endog_std}_hat", np.nan),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significativo_90": bool((ci_low > 0) or (ci_high < 0)),
                "first_stage_f": f_stat,
                "first_stage_pvalor": f_pvalor,
                "nobs": int(res2.nobs),
                "hac_maxlags": hac_maxlags(h)
            })

            primeiros.append({
                "modelo": nome_modelo,
                "resposta": y,
                "h": h,
                "first_stage_f": f_stat,
                "first_stage_pvalor": f_pvalor,
                "r2_primeiro_estagio": fs.rsquared,
                "nobs": int(fs.nobs)
            })

        except Exception as e:
            print(f"[ERRO] LP-IV {nome_modelo}, h={h}: {e}")

    return pd.DataFrame(resultados), pd.DataFrame(primeiros)


# =============================================================================
# 5. GRÁFICOS E SAÍDAS
# =============================================================================

def nome_seguro(texto):
    """
    Cria nome curto e seguro para arquivos de gráfico.

    No Windows, caminhos muito longos podem gerar FileNotFoundError mesmo quando
    a pasta existe. Por isso o nome é encurtado e recebe um hash para evitar
    colisões entre gráficos.
    """
    original = str(texto)

    # Remove extensão recebida, para não gerar nomes tipo _png.png.
    texto = original.replace(".png", "")
    texto = texto.replace("->", "_para_")
    texto = re.sub(r"[^A-Za-z0-9_]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")

    h = hashlib.md5(original.encode("utf-8")).hexdigest()[:8]

    # Nome curto o suficiente para evitar limite de caminho do Windows.
    texto = texto[:70].strip("_")

    return f"{texto}_{h}.png"


def grafico_lp(tab, titulo, caminho, coef_col="coef"):
    if tab is None or tab.empty or coef_col not in tab.columns:
        return False

    temp = tab.sort_values("h").replace([np.inf, -np.inf], np.nan)
    temp = temp.dropna(subset=["h", coef_col])

    if temp.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp["h"], temp[coef_col], marker="o", linewidth=2, label="Resposta estimada")

    if "ci_low" in temp.columns and "ci_high" in temp.columns:
        ax.fill_between(temp["h"], temp["ci_low"], temp["ci_high"], alpha=0.20, label=f"IC {int(CONF * 100)}%")

    ax.axhline(0, linewidth=1)
    ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(titulo)
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("Resposta acumulada")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.text(
        0.01, 0.01,
        "Local Projections acumuladas. IC 90%. HAC/Newey-West: maxlags = max(3, h+1).",
        fontsize=8
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=300)
    plt.close(fig)

    return True


def gerar_graficos_lp(tabela, subpasta, prefixo="LP", coef_col="coef"):
    if tabela is None or tabela.empty:
        return 0

    pasta = OUTPUT_GRAFICOS / subpasta
    pasta.mkdir(parents=True, exist_ok=True)

    n = 0

    if "impulso" in tabela.columns:
        grupos = ["modelo", "impulso", "resposta"]
    else:
        grupos = ["modelo", "impulso_instrumentado", "resposta"]

    for chaves, sub in tabela.groupby(grupos, dropna=False):
        if not isinstance(chaves, tuple):
            chaves = (chaves,)

        info = dict(zip(grupos, chaves))
        modelo = info.get("modelo", "")
        impulso = info.get("impulso", info.get("impulso_instrumentado", ""))
        resposta = info.get("resposta", "")

        titulo = f"{prefixo}: {impulso} → {resposta}"
        nome = nome_seguro(f"{prefixo}_{modelo}_{impulso}_para_{resposta}.png")

        ok = grafico_lp(sub, titulo, pasta / nome, coef_col=coef_col)
        n += int(ok)

    return n


def gerar_graficos_regimes(tabela):
    if tabela is None or tabela.empty:
        return 0

    pasta = OUTPUT_GRAFICOS / "04_bloco_D_regimes_petrobras_out2016"
    pasta.mkdir(parents=True, exist_ok=True)

    n = 0

    for (modelo, impulso, resposta), sub in tabela.groupby(["modelo", "impulso", "resposta"], dropna=False):
        fig, ax = plt.subplots(figsize=(10, 6))

        algum = False

        for regime, sr in sub.groupby("regime"):
            sr = sr.sort_values("h").replace([np.inf, -np.inf], np.nan)
            sr = sr.dropna(subset=["h", "coef"])

            if sr.empty:
                continue

            algum = True
            ax.plot(sr["h"], sr["coef"], marker="o", linewidth=2, label=str(regime))

            if "ci_low" in sr.columns and "ci_high" in sr.columns:
                ax.fill_between(sr["h"], sr["ci_low"], sr["ci_high"], alpha=0.12)

        if not algum:
            plt.close(fig)
            continue

        ax.axhline(0, linewidth=1)
        ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"Regimes Petrobras, corte out/2016: {impulso} → {resposta}")
        ax.set_xlabel("Horizonte h, em meses")
        ax.set_ylabel("Resposta acumulada")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Regime")

        fig.text(0.01, 0.01, "Regimes: pré-outubro/2016 e pós-outubro/2016. IC 90%. HAC/Newey-West.", fontsize=8)

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(pasta / nome_seguro(f"regimes_out2016_{modelo}_{impulso}_para_{resposta}.png"), dpi=300)
        plt.close(fig)

        n += 1

    return n


def gerar_graficos_primeiro_estagio(tabela):
    if tabela is None or tabela.empty or "first_stage_f" not in tabela.columns:
        return 0

    pasta = OUTPUT_GRAFICOS / "06_bloco_F_primeiro_estagio_lpiv"
    pasta.mkdir(parents=True, exist_ok=True)

    n = 0

    for (modelo, resposta), sub in tabela.groupby(["modelo", "resposta"], dropna=False):
        sub = sub.sort_values("h").replace([np.inf, -np.inf], np.nan)
        sub = sub.dropna(subset=["h", "first_stage_f"])

        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sub["h"], sub["first_stage_f"], marker="o", linewidth=2)
        ax.axhline(10, linestyle="--", linewidth=1, label="Regra prática F = 10")
        ax.set_title(f"Primeiro estágio LP-IV: {resposta}")
        ax.set_xlabel("Horizonte h, em meses")
        ax.set_ylabel("F-stat do instrumento")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(pasta / nome_seguro(f"primeiro_estagio_{modelo}_{resposta}.png"), dpi=300)
        plt.close(fig)

        n += 1

    return n


def salvar_resultados(lista, nome):
    dfs = []

    for x in lista:
        if x is not None and isinstance(x, pd.DataFrame) and not x.empty:
            dfs.append(x)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df.to_excel(OUTPUT_TABELAS / f"{nome}.xlsx", index=False)
    df.to_csv(OUTPUT_TABELAS / f"{nome}.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# 6. EXECUÇÃO DO MODELO 8
# =============================================================================

def executar_modelo8():
    print("\n" + "=" * 100)
    print("MODELO 8 - LOCAL PROJECTIONS | BRENT EM REAIS -> COMBUSTÍVEIS -> IPCA")
    print("=" * 100)

    df, dummy_cols, mapa = carregar_preparar_base()
    controles = controles_macro(dummy_cols)

    impulso_petroleo = "dln_petroleo_brl"

    combustiveis = [
        "dln_diesel",
        "dln_gasolina",
        "dln_etanol",
        "dln_gasolina_refinaria"
    ]

    inflacoes = [
        "ipca_geral_mensal",
        "ipca_transporte_mensal"
    ]

    # -------------------------------------------------------------------------
    # BLOCO A
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BLOCO A - PRINCIPAL: BRENT EM REAIS -> COMBUSTÍVEIS")
    print("=" * 100)

    resultados_a = []

    for y in combustiveis:
        print(f"Estimando Bloco A: {impulso_petroleo} -> {y}")
        resultados_a.append(
            local_projection(
                df, y, impulso_petroleo,
                controls=controles,
                lags_n=LAGS_PRINCIPAL,
                h_max=H_MAX,
                nome_modelo=f"bloco_A_principal_brent_brl_para_{y}"
            )
        )

    tabela_a = salvar_resultados(resultados_a, "bloco_A_principal_brent_brl_para_combustiveis")
    n_a = gerar_graficos_lp(tabela_a, "01_bloco_A_brent_brl_para_combustiveis", "Bloco A")
    print(f"[OK] Gráficos Bloco A: {n_a}")

    # -------------------------------------------------------------------------
    # BLOCO B
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BLOCO B - PRINCIPAL: COMBUSTÍVEIS -> IPCA")
    print("=" * 100)

    resultados_b = []

    for shock in combustiveis:
        for y in inflacoes:
            print(f"Estimando Bloco B: {shock} -> {y}")
            resultados_b.append(
                local_projection(
                    df, y, shock,
                    controls=controles,
                    lags_n=LAGS_PRINCIPAL,
                    h_max=H_MAX,
                    nome_modelo=f"bloco_B_principal_{shock}_para_{y}"
                )
            )

    tabela_b = salvar_resultados(resultados_b, "bloco_B_principal_combustiveis_para_ipca")
    n_b = gerar_graficos_lp(tabela_b, "02_bloco_B_combustiveis_para_ipca", "Bloco B")
    print(f"[OK] Gráficos Bloco B: {n_b}")

    # -------------------------------------------------------------------------
    # BLOCO C
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BLOCO C - COMPLEMENTAR: BRENT EM REAIS -> IPCA")
    print("=" * 100)

    resultados_c = []

    for y in inflacoes:
        print(f"Estimando Bloco C: {impulso_petroleo} -> {y}")
        resultados_c.append(
            local_projection(
                df, y, impulso_petroleo,
                controls=controles,
                lags_n=LAGS_PRINCIPAL,
                h_max=H_MAX,
                nome_modelo=f"bloco_C_complementar_brent_brl_para_{y}"
            )
        )

    tabela_c = salvar_resultados(resultados_c, "bloco_C_complementar_brent_brl_para_ipca")
    n_c = gerar_graficos_lp(tabela_c, "03_bloco_C_brent_brl_para_ipca_complementar", "Bloco C")
    print(f"[OK] Gráficos Bloco C: {n_c}")

    # Tabela resumo dos blocos principais
    tabelas_resumo = []
    for nome_bloco, tab in [
        ("A_brent_brl_para_combustiveis", tabela_a),
        ("B_combustiveis_para_ipca", tabela_b),
        ("C_brent_brl_para_ipca_complementar", tabela_c)
    ]:
        if tab is not None and not tab.empty:
            temp = tab.copy()
            temp["bloco"] = nome_bloco
            tabelas_resumo.append(temp[temp["h"].isin(HORIZONTES_RESUMO)].copy())

    if tabelas_resumo:
        tabela_resumo = pd.concat(tabelas_resumo, ignore_index=True)
        tabela_resumo.to_excel(OUTPUT_TABELAS / "tabela_resumo_blocos_A_B_C_h3_h6_h12_h24.xlsx", index=False)
        tabela_resumo.to_csv(OUTPUT_TABELAS / "tabela_resumo_blocos_A_B_C_h3_h6_h12_h24.csv", index=False, encoding="utf-8-sig")
        tabela_resumo[tabela_resumo["h"] == H_PRINCIPAL].to_excel(OUTPUT_TABELAS / "tabela_principal_h12_modelo8.xlsx", index=False)

    # -------------------------------------------------------------------------
    # BLOCO D - PETROBRAS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BLOCO D - PETROBRAS: PRÉ-OUT/2016 VS PÓS-OUT/2016")
    print("=" * 100)

    relacoes_regimes = [
        (impulso_petroleo, "dln_diesel"),
        (impulso_petroleo, "dln_gasolina_refinaria"),
        (impulso_petroleo, "dln_gasolina"),
        (impulso_petroleo, "ipca_transporte_mensal"),
        (impulso_petroleo, "ipca_geral_mensal"),
        ("dln_gasolina", "ipca_transporte_mensal"),
        ("dln_etanol", "ipca_transporte_mensal"),
        ("dln_gasolina_refinaria", "ipca_transporte_mensal"),
    ]

    resultados_regimes = []
    resultados_wald = []

    for shock, y in relacoes_regimes:
        print(f"Estimando Bloco D: {shock} -> {y}")
        tab_reg, tab_wald = local_projection_regimes(
            df, y, shock,
            controls=controles,
            lags_n=LAGS_PRINCIPAL,
            horizontes=HORIZONTES_RESUMO,
            nome_modelo=f"bloco_D_regimes_out2016_{shock}_para_{y}"
        )
        resultados_regimes.append(tab_reg)
        resultados_wald.append(tab_wald)

    tabela_regimes = salvar_resultados(resultados_regimes, "bloco_D_regimes_petrobras_out2016")
    tabela_wald = salvar_resultados(resultados_wald, "bloco_D_testes_wald_regimes_petrobras_out2016")
    n_reg = gerar_graficos_regimes(tabela_regimes)
    print(f"[OK] Gráficos Bloco D: {n_reg}")

    # -------------------------------------------------------------------------
    # BLOCO E - ROBUSTEZ 6 LAGS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BLOCO E - ROBUSTEZ COM 6 DEFASAGENS")
    print("=" * 100)

    relacoes_robustez = [
        (impulso_petroleo, "dln_diesel"),
        (impulso_petroleo, "dln_gasolina_refinaria"),
        (impulso_petroleo, "ipca_transporte_mensal"),
        (impulso_petroleo, "ipca_geral_mensal"),
        ("dln_gasolina", "ipca_transporte_mensal"),
        ("dln_etanol", "ipca_transporte_mensal"),
        ("dln_gasolina", "ipca_geral_mensal"),
    ]

    resultados_rob = []

    for shock, y in relacoes_robustez:
        print(f"Estimando Bloco E: {shock} -> {y}")
        resultados_rob.append(
            local_projection(
                df, y, shock,
                controls=controles,
                lags_n=6,
                h_max=H_MAX,
                horizontes=HORIZONTES_RESUMO,
                nome_modelo=f"bloco_E_robustez_6_lags_{shock}_para_{y}"
            )
        )

    tabela_rob = salvar_resultados(resultados_rob, "bloco_E_robustez_6_lags")
    n_rob = gerar_graficos_lp(tabela_rob, "05_bloco_E_robustez_6_lags", "Bloco E")
    print(f"[OK] Gráficos Bloco E: {n_rob}")

    # -------------------------------------------------------------------------
    # BLOCO F - LP-IV
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BLOCO F - LP-IV: OIL SUPPLY NEWS SHOCK COMO INSTRUMENTO")
    print("=" * 100)

    resultados_iv = []
    resultados_fs = []

    if "oil_supply_news_shock_std" in df.columns:
        for y in ["dln_diesel", "dln_gasolina_refinaria", "ipca_transporte_mensal", "ipca_geral_mensal"]:
            print(f"Estimando Bloco F: {impulso_petroleo} instrumentado -> {y}")
            tab_iv, tab_fs = local_projection_iv(
                df, y, impulso_petroleo, "oil_supply_news_shock_std",
                controls=controles,
                lags_n=LAGS_PRINCIPAL,
                horizontes=HORIZONTES_RESUMO,
                nome_modelo=f"bloco_F_lpiv_oil_supply_news_{impulso_petroleo}_para_{y}"
            )
            resultados_iv.append(tab_iv)
            resultados_fs.append(tab_fs)

        tabela_iv = salvar_resultados(resultados_iv, "bloco_F_lpiv_oil_supply_news")
        tabela_fs = salvar_resultados(resultados_fs, "bloco_F_primeiro_estagio_lpiv_oil_supply_news")

        n_iv = gerar_graficos_lp(tabela_iv, "06_bloco_F_lpiv_oil_supply_news", "Bloco F - LP-IV", coef_col="coef_iv")
        n_fs = gerar_graficos_primeiro_estagio(tabela_fs)

        print(f"[OK] Gráficos LP-IV: {n_iv}")
        print(f"[OK] Gráficos primeiro estágio: {n_fs}")
    else:
        print("[AVISO] Oil Supply News Shock não disponível. Bloco F não foi estimado.")

    # -------------------------------------------------------------------------
    # RELATÓRIO
    # -------------------------------------------------------------------------
    relatorio = []
    relatorio.append("RELATÓRIO DO MODELO 8\n")
    relatorio.append("=====================\n\n")
    relatorio.append("O Modelo 8 organiza o TCC em torno do canal Brent em reais -> combustíveis -> IPCA.\n\n")
    relatorio.append("Bloco A, principal: Brent em reais -> combustíveis.\n")
    relatorio.append("Bloco B, principal: combustíveis -> IPCA Geral e IPCA Transportes.\n")
    relatorio.append("Bloco C, complementar: Brent em reais -> IPCA Geral e IPCA Transportes.\n")
    relatorio.append("Bloco D, Petrobras: pré-outubro/2016 vs pós-outubro/2016.\n")
    relatorio.append("Bloco E, robustez: 6 defasagens.\n")
    relatorio.append("Bloco F, LP-IV: Oil Supply News Shock como instrumento do Brent em reais.\n\n")
    relatorio.append("Interpretação recomendada:\n")
    relatorio.append("O bloco C não deve ser removido, mas deve ser lido como complementar. ")
    relatorio.append("A evidência principal do canal está nos blocos A e B. ")
    relatorio.append("Se o IPCA Geral reagir pouco, isso reforça a interpretação de diluição do choque energético na cesta agregada.\n\n")
    relatorio.append("Pasta de saída: output_petroleo_lp_modelo8.\n")
    relatorio.append("Inferência: HAC/Newey-West, IC 90%, horizonte principal de 12 meses.\n")

    (OUTPUT_RESUMOS / "RELATORIO_MODELO8.txt").write_text("".join(relatorio), encoding="utf-8")

    print("\n" + "=" * 100)
    print("MODELO 8 FINALIZADO")
    print("=" * 100)
    print(f"Arquivos salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    executar_modelo8()
    os._exit(0)
