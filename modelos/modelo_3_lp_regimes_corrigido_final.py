
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

H_MAX = 24
HORIZONTES_ARTIGO = [0, 1, 2, 3, 6, 12]

Z_90 = 1.645


# ============================================================
# MODELO 3 — LOCAL PROJECTIONS POR REGIMES
# ============================================================

PASTA_SAIDA = PASTA_RESULTADOS / "modelo_3_lp_regimes_completo"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

LAGS = 3

CORTES = {
    "principal_set2016": "2016-09-01",
    "alternativo_jan2015": "2015-01-01",
    "alternativo_jan2017": "2017-01-01",
}

base_original = pd.read_excel(
    ARQUIVO_BASE,
    sheet_name="Sheet1"
)

base_original["Data"] = pd.to_datetime(
    base_original["Data"],
    errors="coerce"
)

base_original = (
    base_original
    .dropna(subset=["Data"])
    .sort_values("Data")
)

base_original = base_original[
    (base_original["Data"] >= DATA_INICIO)
    & (base_original["Data"] <= DATA_FIM)
].copy()

base_original = base_original.reset_index(drop=True)

base_original["dlog_petroleo_usd"] = (
    100 * np.log(base_original["Preco_Barril"]).diff()
)

base_original["dlog_cambio"] = (
    100 * np.log(base_original["Cambio"]).diff()
)

base_original["dlog_atividade"] = (
    100 * np.log(base_original["Atividade"]).diff()
)

base_original["selic"] = pd.to_numeric(
    base_original["Selic.1"],
    errors="coerce"
)

base_original["expectativa"] = pd.to_numeric(
    base_original["Expectativa_inflacao"],
    errors="coerce"
)

desvio_petroleo = base_original["dlog_petroleo_usd"].std()

base_original["choque_petroleo"] = (
    base_original["dlog_petroleo_usd"]
    / desvio_petroleo
)

base_original["mes"] = base_original["Data"].dt.month

dummies_mes = pd.get_dummies(
    base_original["mes"],
    prefix="mes",
    drop_first=True,
    dtype=float
)

base_original = pd.concat(
    [base_original, dummies_mes],
    axis=1
)

colunas_dummies = list(dummies_mes.columns)

variaveis_dependentes = {
    "gasolina_refinaria": "Var_GasolinaABrasil_media",
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

todos_resultados = []

for nome_corte, data_corte in CORTES.items():

    base = base_original.copy()

    base["regime_pos"] = (
        base["Data"] >= data_corte
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

    for lag in range(1, LAGS + 1):

        base[f"choque_petroleo_lag{lag}"] = (
            base["choque_petroleo"].shift(lag)
        )

        for controle in controles:

            base[f"{controle}_lag{lag}"] = (
                base[controle].shift(lag)
            )

    for nome_resultado, coluna_y in variaveis_dependentes.items():

        print("")
        print("=" * 75)
        print(
            f"Regimes: {nome_resultado} "
            f"| corte {nome_corte}"
        )
        print("=" * 75)

        base_modelo = base.copy()

        for lag in range(1, LAGS + 1):

            base_modelo[f"y_lag{lag}"] = (
                base_modelo[coluna_y].shift(lag)
            )

        for h in range(0, H_MAX + 1):

            colunas_futuras = []

            for j in range(0, h + 1):

                nome_futuro = f"y_futuro_{j}"

                base_modelo[nome_futuro] = (
                    base_modelo[coluna_y].shift(-j)
                )

                colunas_futuras.append(nome_futuro)

            base_modelo[f"y_acumulado_h{h}"] = (
                base_modelo[colunas_futuras]
                .sum(axis=1, min_count=h + 1)
            )

            regressores = [
                "choque_pre",
                "choque_pos",
                "regime_pos",
                "dlog_cambio",
                "dlog_atividade",
                "selic",
                "expectativa",
            ]

            for lag in range(1, LAGS + 1):

                regressores.append(f"y_lag{lag}")
                regressores.append(f"choque_petroleo_lag{lag}")

                for controle in controles:

                    regressores.append(f"{controle}_lag{lag}")

            regressores = regressores + colunas_dummies

            dados_regressao = base_modelo[
                [f"y_acumulado_h{h}"] + regressores
            ].replace([np.inf, -np.inf], np.nan).dropna()

            y = dados_regressao[f"y_acumulado_h{h}"]

            X = sm.add_constant(
                dados_regressao[regressores],
                has_constant="add"
            )

            resultado = sm.OLS(y, X).fit(
                cov_type="HAC",
                cov_kwds={
                    "maxlags": max(1, h + 1)
                }
            )

            coef_pre = resultado.params["choque_pre"]
            erro_pre = resultado.bse["choque_pre"]

            coef_pos = resultado.params["choque_pos"]
            erro_pos = resultado.bse["choque_pos"]

            nomes_parametros = list(resultado.params.index)

            matriz_wald = np.zeros(
                (1, len(nomes_parametros))
            )

            indice_pre = nomes_parametros.index("choque_pre")
            indice_pos = nomes_parametros.index("choque_pos")

            matriz_wald[0, indice_pos] = 1
            matriz_wald[0, indice_pre] = -1

            teste_wald = resultado.wald_test(
                matriz_wald,
                scalar=True
            )

            diferenca = coef_pos - coef_pre

            variancia_diferenca = float(
                np.asarray(
                    matriz_wald
                    @ resultado.cov_params().values
                    @ matriz_wald.T
                ).squeeze()
            )

            erro_diferenca = np.sqrt(
                max(variancia_diferenca, 0)
            )

            todos_resultados.append({
                "corte": nome_corte,
                "data_corte": data_corte,
                "variavel": nome_resultado,
                "h": h,
                "coeficiente_pre": coef_pre,
                "erro_padrao_pre": erro_pre,
                "p_valor_pre": resultado.pvalues["choque_pre"],
                "limite_inferior_pre_90": coef_pre - Z_90 * erro_pre,
                "limite_superior_pre_90": coef_pre + Z_90 * erro_pre,
                "coeficiente_pos": coef_pos,
                "erro_padrao_pos": erro_pos,
                "p_valor_pos": resultado.pvalues["choque_pos"],
                "limite_inferior_pos_90": coef_pos - Z_90 * erro_pos,
                "limite_superior_pos_90": coef_pos + Z_90 * erro_pos,
                "diferenca_pos_menos_pre": diferenca,
                "erro_padrao_diferenca": erro_diferenca,
                "p_valor_wald": float(
                    np.asarray(teste_wald.pvalue).squeeze()
                ),
                "estatistica_wald": float(
                    np.asarray(teste_wald.statistic).squeeze()
                ),
                "observacoes": int(resultado.nobs),
            })

            base_modelo = base_modelo.drop(
                columns=colunas_futuras
            )

tabela_final = pd.DataFrame(todos_resultados)

tabela_final.to_excel(
    PASTA_SAIDA / "resultados_consolidados_modelo_3.xlsx",
    index=False
)

tabela_artigo = tabela_final.loc[
    (tabela_final["corte"].astype(str) == "principal_set2016")
    & (tabela_final["h"].isin(HORIZONTES_ARTIGO))
].copy()

tabela_artigo = tabela_artigo.sort_values(
    ["variavel", "h"]
).reset_index(drop=True)

if tabela_artigo.empty:
    raise RuntimeError(
        "A tabela resumida do Modelo 3 ficou vazia. "
        "Verifique os valores da coluna 'corte' em resultados_consolidados_modelo_3.xlsx."
    )

tabela_artigo.to_excel(
    PASTA_SAIDA / "tabela_resumida_artigo_modelo_3.xlsx",
    index=False,
    engine="openpyxl"
)

tabela_artigo.to_csv(
    PASTA_SAIDA / "tabela_resumida_artigo_modelo_3.csv",
    index=False,
    encoding="utf-8-sig"
)

for nome_resultado in variaveis_dependentes.keys():

    tabela_grafico = tabela_final[
        (tabela_final["corte"] == "principal_set2016")
        & (tabela_final["variavel"] == nome_resultado)
    ].copy()

    plt.figure(figsize=(9, 5))

    plt.plot(
        tabela_grafico["h"],
        tabela_grafico["coeficiente_pre"],
        marker="o",
        label="Pré-setembro de 2016"
    )

    plt.fill_between(
        tabela_grafico["h"],
        tabela_grafico["limite_inferior_pre_90"],
        tabela_grafico["limite_superior_pre_90"],
        alpha=0.15
    )

    plt.plot(
        tabela_grafico["h"],
        tabela_grafico["coeficiente_pos"],
        marker="s",
        label="Pós-setembro de 2016"
    )

    plt.fill_between(
        tabela_grafico["h"],
        tabela_grafico["limite_inferior_pos_90"],
        tabela_grafico["limite_superior_pos_90"],
        alpha=0.15
    )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Horizonte, em meses")
    plt.ylabel("Resposta acumulada")
    plt.title(f"Regimes: petróleo em dólares -> {nome_resultado}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA / f"grafico_regimes_{nome_resultado}.png",
        dpi=300
    )

    plt.close()

print("")
print("Modelo 3 concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
