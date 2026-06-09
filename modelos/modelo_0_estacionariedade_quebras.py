# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PASTA_BASE = Path(
    r"C:\Users\pedro\OneDrive\Documentos\TCC_python\base_de_dados"
)

PASTA_RESULTADOS = Path(
    r"C:\Users\pedro\OneDrive\Documentos\TCC_python\resultados"
)

ARQUIVO_BASE = PASTA_BASE / "IPCA.xlsx"

DATA_INICIO = "2003-01-01"
DATA_FIM = "2025-09-01"

HORIZONTES_ARTIGO = [0, 1, 2, 3, 6, 12]
H_MAX = 12
Z_90 = 1.645

# ============================================================
# MODELO 0 — AUDITORIA DE ESTACIONARIEDADE E QUEBRAS
# ============================================================

from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from arch.unitroot import PhillipsPerron, DFGLS

PASTA_SAIDA = (
    PASTA_RESULTADOS
    / "modelo_0_estacionariedade_quebras"
)
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

base = pd.read_excel(
    ARQUIVO_BASE,
    sheet_name="Sheet1"
)

base["Data"] = pd.to_datetime(
    base["Data"],
    errors="coerce"
)

base = (
    base
    .dropna(subset=["Data"])
    .sort_values("Data")
)

base = base[
    (base["Data"] >= DATA_INICIO)
    & (base["Data"] <= DATA_FIM)
].copy()

base = base.reset_index(drop=True)

base["dlog_petroleo_usd"] = (
    100 * np.log(base["Preco_Barril"]).diff()
)

base["dlog_cambio"] = (
    100 * np.log(base["Cambio"]).diff()
)

base["dlog_atividade"] = (
    100 * np.log(base["Atividade"]).diff()
)

base["selic_nivel"] = pd.to_numeric(
    base["Selic.1"],
    errors="coerce"
)

base["delta_selic"] = (
    base["selic_nivel"].diff()
)

base["expectativa_nivel"] = pd.to_numeric(
    base["Expectativa_inflacao"],
    errors="coerce"
)

base["delta_expectativa"] = (
    base["expectativa_nivel"].diff()
)

base["delta_gasolina_refinaria"] = (
    pd.to_numeric(
        base["Var_GasolinaABrasil_media"],
        errors="coerce"
    ).diff()
)

base["delta_diesel"] = (
    pd.to_numeric(
        base["Var_Oleo_diesel"],
        errors="coerce"
    ).diff()
)

variaveis_teste = {
    "petroleo_usd_dlog": "dlog_petroleo_usd",
    "cambio_dlog": "dlog_cambio",
    "atividade_dlog": "dlog_atividade",
    "selic_nivel": "selic_nivel",
    "delta_selic": "delta_selic",
    "expectativa_nivel": "expectativa_nivel",
    "delta_expectativa": "delta_expectativa",
    "gasolina_refinaria": "Var_GasolinaABrasil_media",
    "delta_gasolina_refinaria": "delta_gasolina_refinaria",
    "gasolina_consumidor": "Var_Gasolina",
    "etanol": "Var_Etanol",
    "diesel": "Var_Oleo_diesel",
    "delta_diesel": "delta_diesel",
    "ipca_geral": "Var_IPCA_Geral",
    "ipca_transportes": "Var_IPCA_Trans",
}

resultados = []

for nome, coluna in variaveis_teste.items():

    serie = pd.to_numeric(
        base[coluna],
        errors="coerce"
    ).dropna()

    print("")
    print("=" * 75)
    print(f"Testes de estacionariedade: {nome}")
    print("=" * 75)

    adf = adfuller(
        serie,
        regression="c",
        autolag="AIC"
    )

    kpss_resultado = kpss(
        serie,
        regression="c",
        nlags="auto"
    )

    pp = PhillipsPerron(
        serie,
        trend="c"
    )

    dfgls = DFGLS(
        serie,
        trend="c"
    )

    # Zivot-Andrews:
    # hipótese nula = raiz unitária sem quebra;
    # alternativa = estacionária com uma quebra endógena.
    za = zivot_andrews(
        serie,
        trim=0.15,
        maxlag=None,
        regression="ct",
        autolag="AIC"
    )

    indice_quebra = int(za[3])
    data_quebra = (
        serie.index[indice_quebra]
        if indice_quebra < len(serie.index)
        else np.nan
    )

    if isinstance(data_quebra, (int, np.integer)):
        data_quebra_calendario = base.loc[
            data_quebra,
            "Data"
        ]
    else:
        data_quebra_calendario = pd.NaT

    kpss_limite_superior = bool(
        np.isclose(kpss_resultado[1], 0.10)
    )

    resultados.append({
        "variavel": nome,
        "observacoes": len(serie),
        "adf_estatistica": adf[0],
        "adf_p_valor": adf[1],
        "adf_lags": adf[2],
        "kpss_estatistica": kpss_resultado[0],
        "kpss_p_valor_retornado": kpss_resultado[1],
        "kpss_reportar_como": (
            "p > 0,10"
            if kpss_limite_superior
            else f"p = {kpss_resultado[1]:.3f}"
        ),
        "pp_estatistica": pp.stat,
        "pp_p_valor": pp.pvalue,
        "dfgls_estatistica": dfgls.stat,
        "dfgls_p_valor": dfgls.pvalue,
        "zivot_andrews_estatistica": za[0],
        "zivot_andrews_p_valor": za[1],
        "zivot_andrews_lag": za[4],
        "zivot_andrews_indice_quebra": indice_quebra,
        "zivot_andrews_data_quebra": data_quebra_calendario,
    })

tabela = pd.DataFrame(resultados)

tabela.to_excel(
    PASTA_SAIDA
    / "testes_estacionariedade_com_quebras.xlsx",
    index=False
)

tabela[
    tabela["variavel"].isin([
        "gasolina_refinaria",
        "delta_gasolina_refinaria",
        "diesel",
        "delta_diesel",
        "selic_nivel",
        "delta_selic",
        "expectativa_nivel",
        "delta_expectativa",
    ])
].to_excel(
    PASTA_SAIDA
    / "tabela_principal_estacionariedade_banca.xlsx",
    index=False
)

print("")
print("Modelo 0 concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
