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
# MODELO 3B — REGIMES COM CONTROLES DE EVENTOS EXTREMOS
# ============================================================

PASTA_SAIDA = (
    PASTA_RESULTADOS
    / "modelo_3b_regimes_eventos"
)
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

LAGS = 3

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

base["selic"] = pd.to_numeric(
    base["Selic.1"],
    errors="coerce"
)

base["expectativa"] = pd.to_numeric(
    base["Expectativa_inflacao"],
    errors="coerce"
)

base["choque_petroleo"] = (
    base["dlog_petroleo_usd"]
    / base["dlog_petroleo_usd"].std()
)

base["regime_pos"] = (
    base["Data"] >= "2016-09-01"
).astype(float)

base["regime_pos"] = (
    base["regime_pos"]
    .shift(1)
    .fillna(0)
)

base["choque_pre"] = (
    base["choque_petroleo"]
    * (1 - base["regime_pos"])
)

base["choque_pos"] = (
    base["choque_petroleo"]
    * base["regime_pos"]
)

base["dummy_greve_2018"] = (
    base["Data"].between(
        "2018-05-01",
        "2018-06-01"
    )
).astype(float)

base["dummy_pandemia"] = (
    base["Data"].between(
        "2020-03-01",
        "2021-06-01"
    )
).astype(float)

base["dummy_guerra_ucrania"] = (
    base["Data"].between(
        "2022-02-01",
        "2022-12-01"
    )
).astype(float)

base["mes"] = base["Data"].dt.month

dummies_mes = pd.get_dummies(
    base["mes"],
    prefix="mes",
    drop_first=True,
    dtype=float
)

base = pd.concat(
    [base, dummies_mes],
    axis=1
)

colunas_dummies = list(
    dummies_mes.columns
)

variaveis_dependentes = {
    "gasolina_refinaria": (
        "Var_GasolinaABrasil_media"
    ),
    "gasolina_consumidor": "Var_Gasolina",
    "etanol": "Var_Etanol",
    "diesel": "Var_Oleo_diesel",
    "ipca_geral": "Var_IPCA_Geral",
    "ipca_transportes": "Var_IPCA_Trans",
}

controles = [
    "dlog_cambio",
    "dlog_atividade",
    "selic",
    "expectativa",
]

for lag in range(1, LAGS + 1):

    base[f"choque_lag{lag}"] = (
        base["choque_petroleo"].shift(lag)
    )

    for controle in controles:

        base[f"{controle}_lag{lag}"] = (
            base[controle].shift(lag)
        )

resultados = []

for nome_variavel, coluna_y in variaveis_dependentes.items():

    dados = base.copy()

    for lag in range(1, LAGS + 1):

        dados[f"y_lag{lag}"] = (
            dados[coluna_y].shift(lag)
        )

    for h in range(0, H_MAX + 1):

        colunas_futuras = []

        for j in range(0, h + 1):

            coluna_futura = f"y_futuro_{j}"

            dados[coluna_futura] = (
                dados[coluna_y].shift(-j)
            )

            colunas_futuras.append(
                coluna_futura
            )

        dados[f"y_acumulado_h{h}"] = (
            dados[colunas_futuras]
            .sum(
                axis=1,
                min_count=h + 1
            )
        )

        regressores = [
            "choque_pre",
            "choque_pos",
            "regime_pos",
            "dlog_cambio",
            "dlog_atividade",
            "selic",
            "expectativa",
            "dummy_greve_2018",
            "dummy_pandemia",
            "dummy_guerra_ucrania",
        ]

        for lag in range(1, LAGS + 1):

            regressores += [
                f"y_lag{lag}",
                f"choque_lag{lag}",
            ]

            for controle in controles:

                regressores.append(
                    f"{controle}_lag{lag}"
                )

        regressores += colunas_dummies

        amostra = dados[
            [f"y_acumulado_h{h}"]
            + regressores
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        y = amostra[
            f"y_acumulado_h{h}"
        ]

        X = sm.add_constant(
            amostra[regressores],
            has_constant="add"
        )

        resultado = sm.OLS(
            y,
            X
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": max(1, h + 1)
            }
        )

        nomes = list(
            resultado.params.index
        )

        R = np.zeros(
            (1, len(nomes))
        )

        R[
            0,
            nomes.index("choque_pos")
        ] = 1

        R[
            0,
            nomes.index("choque_pre")
        ] = -1

        wald = resultado.wald_test(
            R,
            scalar=True
        )

        resultados.append({
            "variavel": nome_variavel,
            "h": h,
            "coeficiente_pre": (
                resultado.params[
                    "choque_pre"
                ]
            ),
            "p_valor_pre": (
                resultado.pvalues[
                    "choque_pre"
                ]
            ),
            "coeficiente_pos": (
                resultado.params[
                    "choque_pos"
                ]
            ),
            "p_valor_pos": (
                resultado.pvalues[
                    "choque_pos"
                ]
            ),
            "diferenca_pos_menos_pre": (
                resultado.params[
                    "choque_pos"
                ]
                - resultado.params[
                    "choque_pre"
                ]
            ),
            "estatistica_wald": float(
                np.asarray(
                    wald.statistic
                ).squeeze()
            ),
            "p_valor_wald": float(
                np.asarray(
                    wald.pvalue
                ).squeeze()
            ),
            "observacoes": int(
                resultado.nobs
            ),
        })

        dados = dados.drop(
            columns=colunas_futuras
        )

tabela = pd.DataFrame(
    resultados
)

tabela.to_excel(
    PASTA_SAIDA
    / "regimes_com_eventos_extremos.xlsx",
    index=False
)

tabela[
    tabela["h"].isin(
        HORIZONTES_ARTIGO
    )
].to_excel(
    PASTA_SAIDA
    / "tabela_artigo_regimes_eventos.xlsx",
    index=False
)

print("")
print("Modelo 3B concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
