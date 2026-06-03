# -*- coding: utf-8 -*-
"""
petroleo_lp_modelo6.py

MODELO 6 - LOCAL PROJECTIONS DO TCC

IMPORTANTE:
Esta versão foi reescrita para seguir a mesma lógica de leitura do Modelo 5.
Ou seja, ela usa:

1. CONFIG_COLUNAS explícito;
2. localizar_arquivo();
3. encontrar_coluna();
4. carregar_preparar_base_ipca();
5. merge mensal por Data com Selic, Stringency e Oil Supply News Shock.

Se no terminal aparecer "Preparando variáveis..." ou "preparar_variaveis",
você NÃO está rodando este arquivo. Está rodando uma versão antiga.

Arquivos esperados:
- IPCA.xlsx
- STP-20260429165342557.csv
- Stringency_index.csv
- oilSupplyNewsShocks_2025M06.xlsx

Modelo:
- LP principal com 3 defasagens.
- Robustez com 6 defasagens.
- Petrobras em 3 regimes: 2003-2014, 2015-2022 e 2023+.
- LP-IV com Oil Supply News Shock como robustez.
"""

# =============================================================================
# 0. PACOTES
# =============================================================================

import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURAÇÕES GERAIS
# =============================================================================

ARQUIVO_IPCA = r"C:\Users\pedro\OneDrive\Documentos\TCC\IPCA.xlsx"
ARQUIVO_CHOQUE_OFERTA = r"C:\Users\pedro\OneDrive\Documentos\TCC\oilSupplyNewsShocks_2025M06.xlsx"
ARQUIVO_SELIC = r"C:\Users\pedro\OneDrive\Documentos\TCC\STP-20260429165342557.csv"
ARQUIVO_STRINGENCY = r"C:\Users\pedro\OneDrive\Documentos\TCC\Stringency_index.csv"

ABA_IPCA = 0
ABA_CHOQUE = "Monthly"

OUTPUT_DIR = Path("output_petroleo_lp_modelo6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABELAS = OUTPUT_DIR / "tabelas"
OUTPUT_TABELAS.mkdir(parents=True, exist_ok=True)
OUTPUT_RESUMOS = OUTPUT_DIR / "resumos"
OUTPUT_RESUMOS.mkdir(parents=True, exist_ok=True)

OUTPUT_GRAFICOS = OUTPUT_DIR / "graficos"
OUTPUT_GRAFICOS.mkdir(parents=True, exist_ok=True)

DATA_INICIO = "2003-01-01"

H_MAX = 24
H_PRINCIPAL = 12
HORIZONTES_RESUMO = [3, 6, 12, 24]

LAGS_PRINCIPAL = 3
LAGS_ROBUSTEZ = [6]

CONF = 0.90
Z_CRIT = stats.norm.ppf(1 - (1 - CONF) / 2)

USAR_DUMMIES_MENSAIS = True
MIN_OBS = 60


# =============================================================================
# 2. CONFIG_COLUNAS - MESMO PADRÃO DO MODELO 5
# =============================================================================

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

    # Mantive o padrão do Modelo 5 e acrescentei o nome real do seu Excel.
    "expectativa": [
        "Expectativa_Inflacao", "Expectativa_Inflação", "Focus_IPCA_12m",
        "IPCA_Focus_12m", "Expectativa", "espectativa_inflacao",
        "Espectativa_Inflacao", "Espectativa_Inflação"
    ],

    "stringency": ["Stringency", "stringency", "Stringency_Index", "Oxford_Stringency", "stringency_externo"]
}


# =============================================================================
# 3. FUNÇÕES AUXILIARES DE LEITURA
# =============================================================================

def localizar_arquivo(nome_arquivo, caminho_configurado=None):
    """
    Procura arquivo em:
    1. caminho configurado;
    2. pasta atual;
    3. pasta do script;
    4. OneDrive/Documentos/TCC;
    5. OneDrive/Documentos/TCC_python.
    """
    candidatos = []

    if caminho_configurado:
        candidatos.append(Path(caminho_configurado))

    candidatos.append(Path(nome_arquivo))
    candidatos.append(Path.cwd() / nome_arquivo)

    try:
        candidatos.append(Path(__file__).resolve().parent / nome_arquivo)
    except Exception:
        pass

    home = Path.home()
    candidatos.append(home / "OneDrive" / "Documentos" / "TCC" / nome_arquivo)
    candidatos.append(home / "OneDrive" / "Documentos" / "TCC_python" / nome_arquivo)
    candidatos.append(home / "Documents" / "TCC" / nome_arquivo)
    candidatos.append(home / "Documents" / "TCC_python" / nome_arquivo)

    vistos = set()
    candidatos_unicos = []
    for c in candidatos:
        try:
            chave = str(c.resolve())
        except Exception:
            chave = str(c)
        if chave not in vistos:
            vistos.add(chave)
            candidatos_unicos.append(c)

    for c in candidatos_unicos:
        if c.exists():
            return str(c)

    msg = [f"Não encontrei o arquivo: {nome_arquivo}", "", "Locais testados:"]
    msg += [f"- {c}" for c in candidatos_unicos]
    raise FileNotFoundError("\n".join(msg))


def normalizar_nome(x):
    x = str(x).strip().lower()
    trocas = {
        "á": "a", "à": "a", "ã": "a", "â": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u",
        "ç": "c"
    }
    for a, b in trocas.items():
        x = x.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "", x)


def encontrar_coluna(df, candidatos, obrigatoria=False, nome_logico=""):
    """
    Procura coluna com a mesma lógica do Modelo 5:
    - match exato;
    - match ignorando caixa;
    - match normalizado sem acento/espaço/underscore/hífen;
    - match parcial normalizado.
    """
    cols = list(df.columns)

    for c in candidatos:
        if c in cols:
            return c

    cols_lower = {str(c).lower(): c for c in cols}
    for c in candidatos:
        if str(c).lower() in cols_lower:
            return cols_lower[str(c).lower()]

    normalizadas = {normalizar_nome(c): c for c in cols}
    for c in candidatos:
        chave = normalizar_nome(c)
        if chave in normalizadas:
            return normalizadas[chave]

    for c in candidatos:
        chave = normalizar_nome(c)
        for k, v in normalizadas.items():
            if chave and (chave in k or k in chave):
                return v

    if obrigatoria:
        raise ValueError(
            f"Não encontrei a coluna obrigatória '{nome_logico}'.\n"
            f"Candidatos testados: {candidatos}\n"
            f"Colunas disponíveis: {cols}"
        )

    return None


def to_num(s):
    if isinstance(s, pd.Series):
        return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def safe_log_diff(s):
    s = to_num(s)
    s = s.where(s > 0)
    return np.log(s).diff()


def dlog100(s):
    return 100 * safe_log_diff(s)


def diff_se_precisa(s):
    """
    Se a série parecer índice/nível, usa Δlog*100.
    Se parecer variação/taxa mensal pronta, mantém a série.
    """
    s = to_num(s)
    med = s.dropna().median()
    if pd.notna(med) and abs(med) > 20:
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
    if prefix is None:
        prefix = var

    lag_cols = []
    for L in range(1, n_lags + 1):
        col = f"{prefix}_lag{L}"
        df[col] = df[var].shift(L)
        lag_cols.append(col)

    return lag_cols


def montar_y_h(temp, y, h, acumulada=True):
    if acumulada:
        cols_futuras = []
        for j in range(0, h + 1):
            col_fut = f"{y}_lead{j}"
            temp[col_fut] = temp[y].shift(-j)
            cols_futuras.append(col_fut)
        temp[f"y_h{h}"] = temp[cols_futuras].sum(axis=1, min_count=h + 1)
    else:
        temp[f"y_h{h}"] = temp[y].shift(-h)
    return temp


class ResultadoHAC:
    """
    Objeto mínimo para guardar OLS com HAC.
    Fica compatível com o que o restante do código usa:
    params, bse, tvalues, pvalues, nobs, rsquared e wald_test().
    """
    def __init__(self, params, bse, tvalues, pvalues, cov, nobs, rsquared):
        self.params = params
        self.bse = bse
        self.tvalues = tvalues
        self.pvalues = pvalues
        self._cov = cov
        self.nobs = nobs
        self.rsquared = rsquared

    def wald_test(self, R, scalar=True):
        R = np.asarray(R, dtype=float)
        beta = self.params.values.reshape(-1, 1)
        cov = self._cov.values
        diff = R @ beta
        meio = R @ cov @ R.T
        stat = float(diff.T @ np.linalg.pinv(meio) @ diff)
        pvalue = float(1 - stats.chi2.cdf(stat, R.shape[0]))

        class WaldResult:
            def __init__(self, statistic, pvalue):
                self.statistic = statistic
                self.pvalue = pvalue

        return WaldResult(stat, pvalue)


def ajustar_ols_hac(Y, X, h):
    """
    OLS com HAC/Newey-West implementado em numpy.
    É bem mais rápido do que statsmodels.fit(cov_type='HAC') quando há muitas LPs.
    """
    X = sm.add_constant(X, has_constant="add")
    dados = pd.concat([Y.rename("Y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    Yv = dados["Y"].astype(float).to_numpy()
    Xdf = dados.drop(columns=["Y"]).apply(pd.to_numeric, errors="coerce")
    Xv = Xdf.astype(float).to_numpy()

    nobs, k = Xv.shape
    if nobs <= k + 5:
        raise ValueError("Observações insuficientes para OLS-HAC.")

    XtX = Xv.T @ Xv
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-10)

    beta = XtX_inv @ (Xv.T @ Yv)
    resid = Yv - Xv @ beta

    xu = Xv * resid[:, None]
    S = xu.T @ xu

    L = min(hac_maxlags(h), nobs - 1)
    for ell in range(1, L + 1):
        peso = 1.0 - ell / (L + 1.0)
        gamma = xu[ell:].T @ xu[:-ell]
        S += peso * (gamma + gamma.T)

    cov = XtX_inv @ S @ XtX_inv
    diag = np.diag(cov)
    diag = np.where(diag < 0, np.nan, diag)
    bse = np.sqrt(diag)

    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = beta / bse
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(tvalues)))

    ssr = float(np.sum(resid ** 2))
    tss = float(np.sum((Yv - Yv.mean()) ** 2))
    r2 = np.nan if tss == 0 else 1 - ssr / tss

    idx = Xdf.columns
    return ResultadoHAC(
        params=pd.Series(beta, index=idx),
        bse=pd.Series(bse, index=idx),
        tvalues=pd.Series(tvalues, index=idx),
        pvalues=pd.Series(pvalues, index=idx),
        cov=pd.DataFrame(cov, index=idx, columns=idx),
        nobs=int(nobs),
        rsquared=r2
    )


# =============================================================================
# 4. LEITURA DAS BASES EXTERNAS
# =============================================================================

def carregar_selic_diaria():
    try:
        arquivo = localizar_arquivo("STP-20260429165342557.csv", ARQUIVO_SELIC)
    except FileNotFoundError:
        print("[AVISO] Arquivo da Selic diária não encontrado. Usarei a Selic da base principal, se existir.")
        return None

    df = pd.read_csv(arquivo, sep=";", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    col_data = encontrar_coluna(df, ["Data", "data", "Date"], obrigatoria=True, nome_logico="data_selic")
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
        arquivo = localizar_arquivo("Stringency_index.csv", ARQUIVO_STRINGENCY)
    except FileNotFoundError:
        print("[AVISO] Arquivo Stringency_index.csv não encontrado. Controle será zero.")
        return None

    df = pd.read_csv(arquivo, sep=";", encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "CountryCode" in df.columns:
        df = df[df["CountryCode"].astype(str).str.upper().eq("BRA")].copy()

    if "Jurisdiction" in df.columns:
        tmp = df[df["Jurisdiction"].astype(str).str.upper().eq("NAT_TOTAL")].copy()
        if not tmp.empty:
            df = tmp

    col_data = encontrar_coluna(df, ["Date", "Data", "date"], obrigatoria=True, nome_logico="data_stringency")
    col_str = encontrar_coluna(
        df,
        ["StringencyIndex_Average", "StringencyIndex_Average_ForDisplay", "Stringency", "Stringency_Index"],
        obrigatoria=True,
        nome_logico="stringency"
    )

    df[col_data] = pd.to_datetime(df[col_data].astype(str), format="%Y%m%d", errors="coerce")
    if df[col_data].isna().all():
        df[col_data] = pd.to_datetime(df[col_data], errors="coerce")

    df[col_str] = to_num(df[col_str])
    df = df.dropna(subset=[col_data]).copy()
    df["Data"] = df[col_data].dt.to_period("M").dt.to_timestamp()

    out = df.groupby("Data", as_index=False)[col_str].mean()
    out = out.rename(columns={col_str: "stringency_externo"})
    print("[OK] Stringency Index do Brasil importado e agregado para frequência mensal.")
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
        arquivo = localizar_arquivo("oilSupplyNewsShocks_2025M06.xlsx", ARQUIVO_CHOQUE_OFERTA)
    except FileNotFoundError:
        print("[AVISO] Arquivo oilSupplyNewsShocks_2025M06.xlsx não encontrado. LP-IV não será estimada.")
        return None

    xls = pd.ExcelFile(arquivo)
    aba = ABA_CHOQUE if ABA_CHOQUE in xls.sheet_names else ("Monthly" if "Monthly" in xls.sheet_names else xls.sheet_names[0])

    choque = pd.read_excel(arquivo, sheet_name=aba)
    choque.columns = [str(c).strip() for c in choque.columns]

    col_data = encontrar_coluna(choque, ["Date", "Data", "date"], obrigatoria=True, nome_logico="data_choque")
    col_news = encontrar_coluna(
        choque,
        ["Oil supply news shock", "oil supply news shock", "Oil Supply News Shock"],
        obrigatoria=True,
        nome_logico="oil_supply_news_shock"
    )

    choque["Data"] = choque[col_data].apply(parse_data_mensal_choque)
    choque["oil_supply_news_shock"] = to_num(choque[col_news])
    choque = choque.dropna(subset=["Data"]).copy()
    choque["Data"] = choque["Data"].dt.to_period("M").dt.to_timestamp()
    choque["oil_supply_news_shock_std"] = padronizar(choque["oil_supply_news_shock"])

    print("[OK] Oil Supply News Shock mensal importado.")
    return choque[["Data", "oil_supply_news_shock", "oil_supply_news_shock_std"]]


# =============================================================================
# 5. BASE PRINCIPAL - IGUAL AO MODELO 5
# =============================================================================

def carregar_preparar_base_ipca():
    print("\n" + "=" * 100)
    print("1) LEITURA E PREPARAÇÃO DA BASE PRINCIPAL - PADRÃO MODELO 5")
    print("=" * 100)

    arquivo_ipca = localizar_arquivo("IPCA.xlsx", ARQUIVO_IPCA)
    print(f"Arquivo IPCA usado: {arquivo_ipca}")

    df = pd.read_excel(arquivo_ipca, sheet_name=ABA_IPCA)
    df.columns = [str(c).strip() for c in df.columns]

    col_data = encontrar_coluna(df, CONFIG_COLUNAS["data"], obrigatoria=True, nome_logico="data")
    df[col_data] = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[col_data]).sort_values(col_data).reset_index(drop=True)
    df = df.rename(columns={col_data: "Data"})
    df["Data"] = df["Data"].dt.to_period("M").dt.to_timestamp()

    selic_ext = carregar_selic_diaria()
    if selic_ext is not None:
        df = pd.merge(df, selic_ext, on="Data", how="left")

    stringency_ext = carregar_stringency()
    if stringency_ext is not None:
        df = pd.merge(df, stringency_ext, on="Data", how="left")

    choque_ext = carregar_choque_oferta()
    if choque_ext is not None:
        df = pd.merge(df, choque_ext, on="Data", how="left")

    mapa = {}
    for nome_logico, candidatos in CONFIG_COLUNAS.items():
        if nome_logico == "data":
            continue

        obrigatoria = nome_logico in [
            "petroleo", "cambio", "gasolina_refinaria", "gasolina",
            "etanol", "diesel", "ipca_geral", "ipca_transporte", "atividade"
        ]

        mapa[nome_logico] = encontrar_coluna(
            df,
            candidatos,
            obrigatoria=obrigatoria,
            nome_logico=nome_logico
        )

    print("\nColunas identificadas:")
    for k, v in mapa.items():
        print(f"- {k}: {v}")

    df["petroleo_usd_nivel"] = to_num(df[mapa["petroleo"]])
    df["cambio_nivel"] = to_num(df[mapa["cambio"]])
    df["petroleo_brl_nivel"] = df["petroleo_usd_nivel"] * df["cambio_nivel"]

    df["dln_petroleo_usd"] = dlog100(df["petroleo_usd_nivel"])
    df["dln_cambio"] = dlog100(df["cambio_nivel"])
    df["dln_petroleo_brl"] = dlog100(df["petroleo_brl_nivel"])
    df["dln_petroleo_brl_std"] = padronizar(df["dln_petroleo_brl"])

    df["dln_gasolina_refinaria"] = diff_se_precisa(df[mapa["gasolina_refinaria"]])
    df["dln_gasolina"] = diff_se_precisa(df[mapa["gasolina"]])
    df["dln_etanol"] = diff_se_precisa(df[mapa["etanol"]])
    df["dln_diesel"] = diff_se_precisa(df[mapa["diesel"]])

    df["ipca_geral_mensal"] = diff_se_precisa(df[mapa["ipca_geral"]])
    df["ipca_transporte_mensal"] = diff_se_precisa(df[mapa["ipca_transporte"]])

    df["dln_atividade"] = dlog100(df[mapa["atividade"]])

    if "selic_diaria_media_mensal" in df.columns:
        df["selic_controle"] = to_num(df["selic_diaria_media_mensal"])
    elif mapa.get("selic") is not None:
        df["selic_controle"] = to_num(df[mapa["selic"]])
    else:
        df["selic_controle"] = 0.0
        print("[AVISO] Selic não encontrada. Controle Selic ficará igual a zero.")

    if mapa.get("expectativa") is not None:
        df["expectativa_controle"] = to_num(df[mapa["expectativa"]])
        print(f"[OK] Expectativa de inflação usada: {mapa['expectativa']}")
    else:
        df["expectativa_controle"] = 0.0
        print("[AVISO] Expectativa de inflação não encontrada. Controle ficará igual a zero.")

    if "stringency_externo" in df.columns:
        df["stringency_controle"] = to_num(df["stringency_externo"]).fillna(0.0)
    elif mapa.get("stringency") is not None:
        df["stringency_controle"] = to_num(df[mapa["stringency"]]).fillna(0.0)
    else:
        df["stringency_controle"] = 0.0
        print("[AVISO] Stringency não encontrado. Controle ficará igual a zero.")

    df["mes"] = df["Data"].dt.month
    dummy_cols = []
    if USAR_DUMMIES_MENSAIS:
        dummies = pd.get_dummies(df["mes"], prefix="mes", drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        dummy_cols = list(dummies.columns)

    df["regime_2003_2014"] = ((df["Data"] >= "2003-01-01") & (df["Data"] <= "2014-12-01")).astype(int)
    df["regime_2015_2022"] = ((df["Data"] >= "2015-01-01") & (df["Data"] <= "2022-12-01")).astype(int)
    df["regime_2023_2026"] = (df["Data"] >= "2023-01-01").astype(int)

    df = df[df["Data"] >= pd.to_datetime(DATA_INICIO)].copy()
    df = df.sort_values("Data").reset_index(drop=True)

    df.to_excel(OUTPUT_TABELAS / "base_modelo6_transformada.xlsx", index=False)

    print("\nBase final:")
    print(f"- Período: {df['Data'].min().date()} até {df['Data'].max().date()}")
    print(f"- Observações: {len(df)}")
    print(f"- Observações com Oil Supply News Shock: {df['oil_supply_news_shock_std'].notna().sum() if 'oil_supply_news_shock_std' in df.columns else 0}")

    return df, dummy_cols, mapa


def controles_macro(dummy_cols):
    controles = ["dln_atividade", "selic_controle", "expectativa_controle", "stringency_controle"]
    return controles + dummy_cols


# =============================================================================
# 6. LOCAL PROJECTIONS
# =============================================================================

def local_projection(
    df,
    y,
    shock,
    controls=None,
    lags_n=3,
    h_max=24,
    horizontes=None,
    acumulada=True,
    nome_modelo="lp"
):
    controls = controls or []
    base = df.copy()

    shock_std = f"{shock}_std_modelo"
    base[shock_std] = padronizar(base[shock])

    regressores_fixos = []
    regressores_fixos += criar_lags(base, y, lags_n, prefix=y)
    regressores_fixos += criar_lags(base, shock, lags_n, prefix=shock)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            if not c.startswith("mes_"):
                regressores_fixos += criar_lags(base, c, lags_n, prefix=c)

    if horizontes is None:
        horizontes = range(0, h_max + 1)

    resultados = []

    for h in horizontes:
        temp = base.copy()
        temp = montar_y_h(temp, y, h, acumulada=acumulada)

        X_cols = [shock_std] + [c for c in regressores_fixos if c in temp.columns]
        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(MIN_OBS, len(X_cols) + 10):
            resultados.append({
                "modelo": nome_modelo,
                "impulso": shock,
                "resposta": y,
                "h": h,
                "lags": lags_n,
                "coef": np.nan,
                "se": np.nan,
                "pvalor": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "significativo_90": False,
                "nobs": len(temp_reg)
            })
            continue

        Y = temp_reg[f"y_h{h}"]
        X = temp_reg[X_cols]

        try:
            res = ajustar_ols_hac(Y, X, h)
            coef = res.params.get(shock_std, np.nan)
            se = res.bse.get(shock_std, np.nan)
            ci_low = coef - Z_CRIT * se
            ci_high = coef + Z_CRIT * se

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


def local_projection_regimes(
    df,
    y,
    shock,
    controls=None,
    lags_n=3,
    horizontes=None,
    acumulada=True,
    nome_modelo="lp_regimes"
):
    controls = controls or []
    if horizontes is None:
        horizontes = HORIZONTES_RESUMO

    base = df.copy()
    shock_std = f"{shock}_std_modelo"
    base[shock_std] = padronizar(base[shock])

    regimes = {
        "2003_2014": "regime_2003_2014",
        "2015_2022": "regime_2015_2022",
        "2023_2026": "regime_2023_2026"
    }

    interacoes = []
    for nome_regime, col_regime in regimes.items():
        col_int = f"{shock_std}_x_{nome_regime}"
        base[col_int] = base[shock_std] * base[col_regime]
        interacoes.append(col_int)

    regressores_fixos = []
    regressores_fixos += criar_lags(base, y, lags_n, prefix=y)
    regressores_fixos += criar_lags(base, shock, lags_n, prefix=shock)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            if not c.startswith("mes_"):
                regressores_fixos += criar_lags(base, c, lags_n, prefix=c)

    resultados = []
    resultados_wald = []

    for h in horizontes:
        temp = base.copy()
        temp = montar_y_h(temp, y, h, acumulada=acumulada)

        X_cols = interacoes + [c for c in regressores_fixos if c in temp.columns]
        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(MIN_OBS, len(X_cols) + 10):
            continue

        Y = temp_reg[f"y_h{h}"]
        X = temp_reg[X_cols]

        try:
            res = ajustar_ols_hac(Y, X, h)
        except Exception as e:
            print(f"[ERRO] LP regimes {nome_modelo}, h={h}: {e}")
            continue

        for nome_regime in regimes:
            col_int = f"{shock_std}_x_{nome_regime}"
            coef = res.params.get(col_int, np.nan)
            se = res.bse.get(col_int, np.nan)
            ci_low = coef - Z_CRIT * se
            ci_high = coef + Z_CRIT * se

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
        pares = [
            ("2003_2014", "2015_2022"),
            ("2015_2022", "2023_2026"),
            ("2003_2014", "2023_2026")
        ]

        for a, b in pares:
            ca = f"{shock_std}_x_{a}"
            cb = f"{shock_std}_x_{b}"
            if ca in nomes and cb in nomes:
                R = np.zeros((1, len(nomes)))
                R[0, nomes.index(ca)] = 1
                R[0, nomes.index(cb)] = -1
                try:
                    wt = res.wald_test(R, scalar=True)
                    resultados_wald.append({
                        "modelo": nome_modelo,
                        "impulso": shock,
                        "resposta": y,
                        "h": h,
                        "comparacao": f"{a} vs {b}",
                        "diferenca_coef": float(res.params[ca] - res.params[cb]),
                        "wald_stat": float(wt.statistic),
                        "pvalor_wald": float(wt.pvalue),
                        "diferenca_significativa_10": bool(float(wt.pvalue) < 0.10),
                        "nobs": int(res.nobs)
                    })
                except Exception:
                    pass

    return pd.DataFrame(resultados), pd.DataFrame(resultados_wald)


def local_projection_iv(
    df,
    y,
    endog,
    instrument,
    controls=None,
    lags_n=3,
    horizontes=None,
    acumulada=True,
    nome_modelo="lp_iv"
):
    controls = controls or []
    if horizontes is None:
        horizontes = HORIZONTES_RESUMO

    if instrument not in df.columns:
        print(f"[AVISO] Instrumento {instrument} não está na base. LP-IV não será estimada.")
        return pd.DataFrame(), pd.DataFrame()

    base = df.copy()
    endog_std = f"{endog}_std_iv"
    instr_std = f"{instrument}_std_iv"

    base[endog_std] = padronizar(base[endog])
    base[instr_std] = padronizar(base[instrument])

    regressores_fixos = []
    regressores_fixos += criar_lags(base, y, lags_n, prefix=y)
    regressores_fixos += criar_lags(base, endog, lags_n, prefix=endog)
    regressores_fixos += criar_lags(base, instrument, lags_n, prefix=instrument)

    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            if not c.startswith("mes_"):
                regressores_fixos += criar_lags(base, c, lags_n, prefix=c)

    resultados = []
    primeiros = []

    for h in horizontes:
        temp = base.copy()
        temp = montar_y_h(temp, y, h, acumulada=acumulada)

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

            X2_cols = [f"{endog_std}_hat"] + X_controls
            res2 = ajustar_ols_hac(temp_reg[f"y_h{h}"], temp_reg[X2_cols], h)

            coef = res2.params.get(f"{endog_std}_hat", np.nan)
            se = res2.bse.get(f"{endog_std}_hat", np.nan)
            ci_low = coef - Z_CRIT * se
            ci_high = coef + Z_CRIT * se

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
# 7. GRÁFICOS
# =============================================================================

def nome_arquivo_seguro(texto):
    """
    Cria nomes seguros para arquivos de gráfico.
    """
    texto = str(texto)
    texto = texto.replace("->", "_para_")
    texto = texto.replace(" ", "_")
    texto = re.sub(r"[^A-Za-z0-9_]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto[:180]


def grafico_lp_padrao(tab, titulo, caminho, coluna_coef="coef", coluna_low="ci_low", coluna_high="ci_high"):
    """
    Gera gráfico padrão de resposta acumulada com IC 90%.
    """
    if tab is None or tab.empty:
        return False

    cols = ["h", coluna_coef, coluna_low, coluna_high]
    for c in cols:
        if c not in tab.columns:
            return False

    temp = tab[cols].copy()
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna(subset=["h", coluna_coef])
    if temp.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp["h"], temp[coluna_coef], marker="o", linewidth=2, label="Resposta estimada")

    if temp[coluna_low].notna().any() and temp[coluna_high].notna().any():
        ax.fill_between(
            temp["h"],
            temp[coluna_low],
            temp[coluna_high],
            alpha=0.20,
            label=f"IC {int(CONF * 100)}%"
        )

    ax.axhline(0, linewidth=1)
    ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(titulo)
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("Resposta acumulada")
    ax.grid(True, alpha=0.3)
    ax.legend()

    nota = (
        f"Local Projections acumuladas. IC {int(CONF * 100)}%. "
        f"HAC/Newey-West: maxlags = max(3, h+1)."
    )
    fig.text(0.01, 0.01, nota, fontsize=8)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=300)
    plt.close(fig)
    return True


def gerar_graficos_lp(tabela, subpasta, prefixo="lp", coluna_coef="coef"):
    """
    Gera um gráfico para cada combinação impulso-resposta-modelo.
    """
    if tabela is None or tabela.empty:
        return 0

    pasta = OUTPUT_GRAFICOS / subpasta
    pasta.mkdir(parents=True, exist_ok=True)

    n = 0
    grupos = ["modelo", "impulso", "resposta"]
    grupos = [g for g in grupos if g in tabela.columns]

    if not grupos:
        return 0

    for chaves, sub in tabela.groupby(grupos, dropna=False):
        if not isinstance(chaves, tuple):
            chaves = (chaves,)

        info = dict(zip(grupos, chaves))
        modelo = info.get("modelo", "")
        impulso = info.get("impulso", info.get("impulso_instrumentado", ""))
        resposta = info.get("resposta", "")

        titulo = f"{prefixo}: {impulso} → {resposta}"
        if modelo:
            titulo += f" | {modelo}"

        nome = nome_arquivo_seguro(f"{prefixo}_{modelo}_{impulso}_para_{resposta}.png")
        ok = grafico_lp_padrao(
            sub.sort_values("h"),
            titulo=titulo,
            caminho=pasta / nome,
            coluna_coef=coluna_coef,
            coluna_low="ci_low",
            coluna_high="ci_high"
        )
        n += int(ok)

    return n


def gerar_graficos_regimes(tabela):
    """
    Gera gráfico com as três respostas por regime Petrobras.
    """
    if tabela is None or tabela.empty:
        return 0

    pasta = OUTPUT_GRAFICOS / "regimes_petrobras"
    pasta.mkdir(parents=True, exist_ok=True)

    n = 0
    grupos = ["modelo", "impulso", "resposta"]
    grupos = [g for g in grupos if g in tabela.columns]

    for chaves, sub in tabela.groupby(grupos, dropna=False):
        if not isinstance(chaves, tuple):
            chaves = (chaves,)

        info = dict(zip(grupos, chaves))
        modelo = info.get("modelo", "")
        impulso = info.get("impulso", "")
        resposta = info.get("resposta", "")

        fig, ax = plt.subplots(figsize=(10, 6))

        algum = False
        for regime, sr in sub.groupby("regime"):
            sr = sr.sort_values("h").replace([np.inf, -np.inf], np.nan)
            sr = sr.dropna(subset=["h", "coef"])
            if sr.empty:
                continue

            algum = True
            ax.plot(sr["h"], sr["coef"], marker="o", linewidth=2, label=str(regime))

            if sr["ci_low"].notna().any() and sr["ci_high"].notna().any():
                ax.fill_between(sr["h"], sr["ci_low"], sr["ci_high"], alpha=0.12)

        if not algum:
            plt.close(fig)
            continue

        ax.axhline(0, linewidth=1)
        ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"Regimes Petrobras: {impulso} → {resposta}")
        ax.set_xlabel("Horizonte h, em meses")
        ax.set_ylabel("Resposta acumulada")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Regime")

        nota = (
            "Regimes: 2003-2014, 2015-2022 e 2023+. "
            f"IC {int(CONF * 100)}%. HAC/Newey-West."
        )
        fig.text(0.01, 0.01, nota, fontsize=8)

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        nome = nome_arquivo_seguro(f"regimes_{modelo}_{impulso}_para_{resposta}.png")
        fig.savefig(pasta / nome, dpi=300)
        plt.close(fig)
        n += 1

    return n


def gerar_graficos_primeiro_estagio(tabela):
    """
    Gera gráficos do F-stat do primeiro estágio da LP-IV.
    """
    if tabela is None or tabela.empty:
        return 0
    if "first_stage_f" not in tabela.columns:
        return 0

    pasta = OUTPUT_GRAFICOS / "lp_iv_primeiro_estagio"
    pasta.mkdir(parents=True, exist_ok=True)

    n = 0
    grupos = ["modelo", "resposta"]
    grupos = [g for g in grupos if g in tabela.columns]

    for chaves, sub in tabela.groupby(grupos, dropna=False):
        if not isinstance(chaves, tuple):
            chaves = (chaves,)
        info = dict(zip(grupos, chaves))
        modelo = info.get("modelo", "")
        resposta = info.get("resposta", "")

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

        nome = nome_arquivo_seguro(f"primeiro_estagio_{modelo}_{resposta}.png")
        fig.savefig(pasta / nome, dpi=300)
        plt.close(fig)
        n += 1

    return n


# =============================================================================
# 8. EXECUÇÃO
# =============================================================================

def salvar_resultados(lista, nome):
    if not lista:
        return pd.DataFrame()

    lista = [x for x in lista if x is not None and not x.empty]
    if not lista:
        return pd.DataFrame()

    df = pd.concat(lista, ignore_index=True)
    df.to_excel(OUTPUT_TABELAS / f"{nome}.xlsx", index=False)
    df.to_csv(OUTPUT_TABELAS / f"{nome}.csv", index=False, encoding="utf-8-sig")
    return df


def executar_modelo6():
    print("\n" + "=" * 100)
    print("MODELO 6 - LOCAL PROJECTIONS")
    print("VERSÃO COM CONFIG_COLUNAS IGUAL AO MODELO 5")
    print("=" * 100)

    df, dummy_cols, mapa = carregar_preparar_base_ipca()
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

    print("\n" + "=" * 100)
    print("2) LP PRINCIPAL COM 3 DEFASAGENS")
    print("=" * 100)

    resultados_lp = []

    for y in combustiveis:
        print(f"Estimando LP principal: {impulso_petroleo} -> {y}")
        resultados_lp.append(
            local_projection(
                df=df,
                y=y,
                shock=impulso_petroleo,
                controls=controles,
                lags_n=LAGS_PRINCIPAL,
                h_max=H_MAX,
                nome_modelo=f"principal_petroleo_brl_para_{y}"
            )
        )

    for shock in combustiveis:
        for y in inflacoes:
            print(f"Estimando LP principal: {shock} -> {y}")
            resultados_lp.append(
                local_projection(
                    df=df,
                    y=y,
                    shock=shock,
                    controls=controles,
                    lags_n=LAGS_PRINCIPAL,
                    h_max=H_MAX,
                    nome_modelo=f"principal_{shock}_para_{y}"
                )
            )

    for y in inflacoes:
        print(f"Estimando LP principal: {impulso_petroleo} -> {y}")
        resultados_lp.append(
            local_projection(
                df=df,
                y=y,
                shock=impulso_petroleo,
                controls=controles,
                lags_n=LAGS_PRINCIPAL,
                h_max=H_MAX,
                nome_modelo=f"principal_petroleo_brl_para_{y}"
            )
        )

    tabela_lp = salvar_resultados(resultados_lp, "resultados_lp_principal_3_lags")
    n_graficos_lp = gerar_graficos_lp(tabela_lp, subpasta="lp_principal", prefixo="LP principal")
    print(f"[OK] Gráficos LP principal gerados: {n_graficos_lp}")

    if not tabela_lp.empty:
        tabela_lp[tabela_lp["h"].isin(HORIZONTES_RESUMO)].to_excel(
            OUTPUT_TABELAS / "tabela_lp_principal_h3_h6_h12_h24.xlsx",
            index=False
        )
        tabela_lp[tabela_lp["h"] == H_PRINCIPAL].to_excel(
            OUTPUT_TABELAS / "ranking_lp_h12.xlsx",
            index=False
        )

    print("\n" + "=" * 100)
    print("3) ROBUSTEZ COM 6 DEFASAGENS")
    print("=" * 100)

    relacoes_robustez = [
        (impulso_petroleo, "dln_diesel"),
        (impulso_petroleo, "dln_gasolina_refinaria"),
        (impulso_petroleo, "ipca_transporte_mensal"),
        (impulso_petroleo, "ipca_geral_mensal"),
        ("dln_gasolina", "ipca_transporte_mensal"),
        ("dln_gasolina_refinaria", "ipca_transporte_mensal")
    ]

    resultados_rob = []
    for shock, y in relacoes_robustez:
        print(f"Estimando robustez 6 lags: {shock} -> {y}")
        resultados_rob.append(
            local_projection(
                df=df,
                y=y,
                shock=shock,
                controls=controles,
                lags_n=6,
                h_max=H_MAX,
                horizontes=HORIZONTES_RESUMO,
                nome_modelo=f"robustez_6_lags_{shock}_para_{y}"
            )
        )

    tabela_rob = salvar_resultados(resultados_rob, "resultados_lp_robustez_6_lags")
    n_graficos_rob = gerar_graficos_lp(tabela_rob, subpasta="robustez_6_lags", prefixo="Robustez 6 lags")
    print(f"[OK] Gráficos robustez gerados: {n_graficos_rob}")

    print("\n" + "=" * 100)
    print("4) REGIMES PETROBRAS EM TRÊS PERÍODOS")
    print("=" * 100)

    relacoes_regimes = [
        (impulso_petroleo, "dln_diesel"),
        (impulso_petroleo, "dln_gasolina_refinaria"),
        (impulso_petroleo, "ipca_transporte_mensal"),
        (impulso_petroleo, "ipca_geral_mensal"),
        ("dln_gasolina_refinaria", "ipca_transporte_mensal"),
        ("dln_gasolina", "ipca_transporte_mensal"),
        ("dln_etanol", "ipca_transporte_mensal")
    ]

    resultados_regimes = []
    resultados_wald = []

    for shock, y in relacoes_regimes:
        print(f"Estimando regimes Petrobras: {shock} -> {y}")
        tab_reg, tab_wald = local_projection_regimes(
            df=df,
            y=y,
            shock=shock,
            controls=controles,
            lags_n=LAGS_PRINCIPAL,
            horizontes=HORIZONTES_RESUMO,
            nome_modelo=f"regimes_{shock}_para_{y}"
        )
        resultados_regimes.append(tab_reg)
        resultados_wald.append(tab_wald)

    tabela_regimes = salvar_resultados(resultados_regimes, "resultados_lp_regimes_petrobras_3_periodos")
    tabela_wald = salvar_resultados(resultados_wald, "testes_wald_regimes_petrobras_3_periodos")
    n_graficos_reg = gerar_graficos_regimes(tabela_regimes)
    print(f"[OK] Gráficos de regimes Petrobras gerados: {n_graficos_reg}")

    print("\n" + "=" * 100)
    print("5) LP-IV COM OIL SUPPLY NEWS SHOCK")
    print("=" * 100)

    resultados_iv = []
    resultados_fs = []

    if "oil_supply_news_shock_std" in df.columns:
        for y in ["dln_diesel", "dln_gasolina_refinaria", "ipca_transporte_mensal", "ipca_geral_mensal"]:
            print(f"Estimando LP-IV: {impulso_petroleo} instrumentado -> {y}")
            tab_iv, tab_fs = local_projection_iv(
                df=df,
                y=y,
                endog=impulso_petroleo,
                instrument="oil_supply_news_shock_std",
                controls=controles,
                lags_n=LAGS_PRINCIPAL,
                horizontes=HORIZONTES_RESUMO,
                nome_modelo=f"lpiv_petroleo_brl_para_{y}"
            )
            resultados_iv.append(tab_iv)
            resultados_fs.append(tab_fs)

        tabela_iv = salvar_resultados(resultados_iv, "resultados_lp_iv_oil_supply_news")
        tabela_fs = salvar_resultados(resultados_fs, "primeiro_estagio_lp_iv_oil_supply_news")
        n_graficos_iv = gerar_graficos_lp(
            tabela_iv,
            subpasta="lp_iv_oil_supply_news",
            prefixo="LP-IV",
            coluna_coef="coef_iv"
        )
        n_graficos_fs = gerar_graficos_primeiro_estagio(tabela_fs)
        print(f"[OK] Gráficos LP-IV gerados: {n_graficos_iv}")
        print(f"[OK] Gráficos primeiro estágio gerados: {n_graficos_fs}")
    else:
        print("[AVISO] Oil Supply News Shock não está disponível. LP-IV não foi estimada.")

    relatorio = []
    relatorio.append("RELATÓRIO DO MODELO 6\n")
    relatorio.append("=====================\n\n")
    relatorio.append("O Modelo 6 segue o padrão de leitura do Modelo 5, com CONFIG_COLUNAS explícito.\n")
    relatorio.append("Choque principal: Brent em reais = Brent em dólar multiplicado pelo câmbio.\n")
    relatorio.append("Modelo principal: Local Projections acumuladas com 3 defasagens.\n")
    relatorio.append("Robustez: 6 defasagens nas relações principais.\n")
    relatorio.append("Regimes Petrobras: 2003-2014, 2015-2022 e 2023 em diante.\n")
    relatorio.append("LP-IV: Oil Supply News Shock usado apenas como robustez de identificação.\n")
    relatorio.append("Inferência: HAC/Newey-West com maxlags = max(3, h+1).\n")
    relatorio.append("Gráficos: salvos em output_petroleo_lp_modelo6/graficos.\n\n")
    relatorio.append("Interpretação recomendada:\n")
    relatorio.append(
        "Os resultados devem ser lidos como evidência dinâmica reduzida. "
        "A evidência causal mais forte fica restrita ao exercício LP-IV, que deve ser tratado como robustez. "
        "O resultado central esperado é maior transmissão para combustíveis e IPCA Transportes, "
        "com resposta menor, menos persistente ou menos precisa no IPCA Geral.\n"
    )

    (OUTPUT_RESUMOS / "RELATORIO_MODELO6.txt").write_text("".join(relatorio), encoding="utf-8")

    print("\n" + "=" * 100)
    print("MODELO 6 FINALIZADO")
    print("=" * 100)
    print(f"Arquivos salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    executar_modelo6()
