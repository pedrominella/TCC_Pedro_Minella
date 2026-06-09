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
# MODELO 1B — ROBUSTEZ DAS LOCAL PROJECTIONS
# ============================================================
#
# Exercícios:
# 1. especificação original;
# 2. variável dependente em diferença adicional;
# 3. controles para greve, pandemia e período pós-2016;
# 4. choque do petróleo winsorizado em 1% e 99%.
#
# A diferença adicional muda o objeto econômico:
# ela mede a resposta da aceleração da inflação do combustível,
# não o pass-through acumulado original.
# ============================================================

PASTA_SAIDA = (
    PASTA_RESULTADOS
    / "modelo_1b_lp_robustez_banca"
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

desvio_petroleo = (
    base["dlog_petroleo_usd"].std()
)

base["choque_petroleo"] = (
    base["dlog_petroleo_usd"]
    / desvio_petroleo
)

limite_inferior = (
    base["dlog_petroleo_usd"].quantile(0.01)
)

limite_superior = (
    base["dlog_petroleo_usd"].quantile(0.99)
)

base["petroleo_winsorizado"] = (
    base["dlog_petroleo_usd"]
    .clip(
        lower=limite_inferior,
        upper=limite_superior
    )
)

base["choque_petroleo_winsorizado"] = (
    base["petroleo_winsorizado"]
    / base["petroleo_winsorizado"].std()
)

# Eventos extremos e institucionais
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

base["dummy_pos_2016"] = (
    base["Data"] >= "2016-09-01"
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

controles = [
    "dlog_cambio",
    "dlog_atividade",
    "selic",
    "expectativa",
]

variaveis = {
    "gasolina_refinaria": (
        "Var_GasolinaABrasil_media"
    ),
    "diesel": "Var_Oleo_diesel",
}

especificacoes = {
    "original": {
        "choque": "choque_petroleo",
        "diferenca_adicional": False,
        "eventos": False,
    },
    "diferenca_adicional": {
        "choque": "choque_petroleo",
        "diferenca_adicional": True,
        "eventos": False,
    },
    "eventos_extremos": {
        "choque": "choque_petroleo",
        "diferenca_adicional": False,
        "eventos": True,
    },
    "choque_winsorizado": {
        "choque": "choque_petroleo_winsorizado",
        "diferenca_adicional": False,
        "eventos": False,
    },
}

resultados = []

for nome_variavel, coluna_original in variaveis.items():

    for nome_especificacao, configuracao in especificacoes.items():

        print("")
        print("=" * 75)
        print(
            f"LP robustez: {nome_variavel} "
            f"| {nome_especificacao}"
        )
        print("=" * 75)

        dados = base.copy()

        if configuracao["diferenca_adicional"]:

            dados["y_modelo"] = (
                pd.to_numeric(
                    dados[coluna_original],
                    errors="coerce"
                ).diff()
            )

            objeto_economico = (
                "Mudança da inflação mensal do combustível"
            )

        else:

            dados["y_modelo"] = pd.to_numeric(
                dados[coluna_original],
                errors="coerce"
            )

            objeto_economico = (
                "Resposta acumulada da variação mensal"
            )

        choque = configuracao["choque"]

        for lag in range(1, LAGS + 1):

            dados[f"y_lag{lag}"] = (
                dados["y_modelo"].shift(lag)
            )

            dados[f"choque_lag{lag}"] = (
                dados[choque].shift(lag)
            )

            for controle in controles:

                dados[f"{controle}_lag{lag}"] = (
                    dados[controle].shift(lag)
                )

        for h in range(0, H_MAX + 1):

            colunas_futuras = []

            for j in range(0, h + 1):

                coluna_futura = f"y_futuro_{j}"

                dados[coluna_futura] = (
                    dados["y_modelo"].shift(-j)
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
                choque,
                "dlog_cambio",
                "dlog_atividade",
                "selic",
                "expectativa",
            ]

            if configuracao["eventos"]:

                regressores += [
                    "dummy_greve_2018",
                    "dummy_pandemia",
                    "dummy_pos_2016",
                ]

            for lag in range(1, LAGS + 1):

                regressores.append(
                    f"y_lag{lag}"
                )

                regressores.append(
                    f"choque_lag{lag}"
                )

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

            coeficiente = (
                resultado.params[choque]
            )

            erro_padrao = (
                resultado.bse[choque]
            )

            resultados.append({
                "variavel": nome_variavel,
                "especificacao": nome_especificacao,
                "objeto_economico": objeto_economico,
                "h": h,
                "coeficiente": coeficiente,
                "erro_padrao": erro_padrao,
                "estatistica_t": (
                    resultado.tvalues[choque]
                ),
                "p_valor": (
                    resultado.pvalues[choque]
                ),
                "limite_inferior_90": (
                    coeficiente
                    - Z_90 * erro_padrao
                ),
                "limite_superior_90": (
                    coeficiente
                    + Z_90 * erro_padrao
                ),
                "observacoes": int(
                    resultado.nobs
                ),
            })

            dados = dados.drop(
                columns=colunas_futuras
            )

tabela = pd.DataFrame(resultados)

tabela.to_excel(
    PASTA_SAIDA
    / "resultados_lp_robustez_banca.xlsx",
    index=False
)

tabela[
    tabela["h"].isin(
        HORIZONTES_ARTIGO
    )
].to_excel(
    PASTA_SAIDA
    / "tabela_artigo_lp_robustez_banca.xlsx",
    index=False
)

for nome_variavel in variaveis.keys():

    plt.figure(figsize=(9, 5))

    for nome_especificacao in especificacoes.keys():

        grafico = tabela[
            (tabela["variavel"] == nome_variavel)
            & (
                tabela["especificacao"]
                == nome_especificacao
            )
        ].copy()

        plt.plot(
            grafico["h"],
            grafico["coeficiente"],
            marker="o",
            label=nome_especificacao
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Horizonte, em meses")
    plt.ylabel("Resposta acumulada")
    plt.title(
        f"Robustez das LPs: {nome_variavel}"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / f"comparacao_robustez_{nome_variavel}.png",
        dpi=300
    )

    plt.close()

print("")
print("Modelo 1B concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
