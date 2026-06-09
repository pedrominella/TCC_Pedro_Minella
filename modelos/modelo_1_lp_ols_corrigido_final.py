
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
# MODELO 1 — LOCAL PROJECTIONS OLS
# ============================================================

PASTA_SAIDA = PASTA_RESULTADOS / "modelo_1_lp_ols_completo"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

LAGS_PRINCIPAL = 3
LAGS_ROBUSTEZ = [3, 6, 12]

base = pd.read_excel(ARQUIVO_BASE, sheet_name="Sheet1")

base["Data"] = pd.to_datetime(base["Data"], errors="coerce")
base = base.dropna(subset=["Data"]).sort_values("Data")
base = base[
    (base["Data"] >= DATA_INICIO)
    & (base["Data"] <= DATA_FIM)
].copy()
base = base.reset_index(drop=True)

base["dlog_petroleo_usd"] = 100 * np.log(base["Preco_Barril"]).diff()
base["dlog_cambio"] = 100 * np.log(base["Cambio"]).diff()
base["dlog_atividade"] = 100 * np.log(base["Atividade"]).diff()

base["selic"] = pd.to_numeric(base["Selic.1"], errors="coerce")
base["expectativa"] = pd.to_numeric(
    base["Expectativa_inflacao"],
    errors="coerce"
)

desvio_petroleo = base["dlog_petroleo_usd"].std()
base["choque_petroleo"] = base["dlog_petroleo_usd"] / desvio_petroleo

base["mes"] = base["Data"].dt.month

dummies_mes = pd.get_dummies(
    base["mes"],
    prefix="mes",
    drop_first=True,
    dtype=float
)

base = pd.concat([base, dummies_mes], axis=1)
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
testes_residuos = []

for numero_lags in LAGS_ROBUSTEZ:

    base_lags = base.copy()

    for lag in range(1, numero_lags + 1):

        base_lags[f"choque_petroleo_lag{lag}"] = (
            base_lags["choque_petroleo"].shift(lag)
        )

        for controle in controles:

            base_lags[f"{controle}_lag{lag}"] = (
                base_lags[controle].shift(lag)
            )

    for nome_resultado, coluna_y in variaveis_dependentes.items():

        print("")
        print("=" * 75)
        print(
            f"LP-OLS: petróleo em dólares -> {nome_resultado} "
            f"| {numero_lags} defasagens"
        )
        print("=" * 75)

        base_modelo = base_lags.copy()

        for lag in range(1, numero_lags + 1):

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
                "choque_petroleo",
                "dlog_cambio",
                "dlog_atividade",
                "selic",
                "expectativa",
            ]

            for lag in range(1, numero_lags + 1):

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

            coeficiente = resultado.params["choque_petroleo"]
            erro_padrao = resultado.bse["choque_petroleo"]

            todos_resultados.append({
                "variavel": nome_resultado,
                "h": h,
                "lags": numero_lags,
                "coeficiente": coeficiente,
                "erro_padrao": erro_padrao,
                "estatistica_t": resultado.tvalues["choque_petroleo"],
                "p_valor": resultado.pvalues["choque_petroleo"],
                "limite_inferior_90": coeficiente - Z_90 * erro_padrao,
                "limite_superior_90": coeficiente + Z_90 * erro_padrao,
                "observacoes": int(resultado.nobs),
                "desvio_padrao_petroleo": desvio_petroleo,
            })

            if h in HORIZONTES_ARTIGO:

                residuo = resultado.resid

                try:
                    teste_bg = sm.stats.acorr_breusch_godfrey(
                        resultado,
                        nlags=max(1, min(12, h + 1))
                    )

                    bg_estatistica = teste_bg[0]
                    bg_p_valor = teste_bg[1]

                except Exception:

                    bg_estatistica = np.nan
                    bg_p_valor = np.nan

                testes_residuos.append({
                    "variavel": nome_resultado,
                    "h": h,
                    "lags": numero_lags,
                    "bg_estatistica": bg_estatistica,
                    "bg_p_valor": bg_p_valor,
                })

            base_modelo = base_modelo.drop(
                columns=colunas_futuras
            )

tabela_final = pd.DataFrame(todos_resultados)

tabela_final.to_excel(
    PASTA_SAIDA / "resultados_consolidados_modelo_1.xlsx",
    index=False
)

tabela_artigo = tabela_final[
    (tabela_final["h"].isin(HORIZONTES_ARTIGO))
    & (tabela_final["lags"] == LAGS_PRINCIPAL)
].copy()

tabela_artigo.to_excel(
    PASTA_SAIDA / "tabela_resumida_artigo_modelo_1.xlsx",
    index=False
)

tabela_testes_residuos = pd.DataFrame(testes_residuos)

tabela_testes_residuos["observacao_metodologica"] = np.where(
    tabela_testes_residuos["h"] == 0,
    "Teste BG diretamente interpretável em h=0.",
    "Em h>0, a variável acumulada gera sobreposição mecânica e autocorrelação esperada; usar HAC."
)

tabela_testes_residuos.to_excel(
    PASTA_SAIDA / "testes_residuos_modelo_1.xlsx",
    index=False
)

tabela_testes_residuos[
    tabela_testes_residuos["h"] == 0
].to_excel(
    PASTA_SAIDA / "testes_residuos_h0_modelo_1.xlsx",
    index=False
)

for nome_resultado in variaveis_dependentes.keys():

    tabela_grafico = tabela_final[
        (tabela_final["variavel"] == nome_resultado)
        & (tabela_final["lags"] == LAGS_PRINCIPAL)
    ].copy()

    plt.figure(figsize=(9, 5))

    plt.plot(
        tabela_grafico["h"],
        tabela_grafico["coeficiente"],
        marker="o"
    )

    plt.fill_between(
        tabela_grafico["h"],
        tabela_grafico["limite_inferior_90"],
        tabela_grafico["limite_superior_90"],
        alpha=0.20
    )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Horizonte, em meses")
    plt.ylabel("Resposta acumulada")
    plt.title(f"LP-OLS: petróleo em dólares -> {nome_resultado}")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA / f"grafico_lp_ols_{nome_resultado}.png",
        dpi=300
    )

    plt.close()

print("")
print("Modelo 1 concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
