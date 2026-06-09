
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


from linearmodels.iv import IV2SLS

# ============================================================
# MODELO 2 — LOCAL PROJECTIONS IV COM KÄNZIG
# ============================================================

ARQUIVO_KANZIG = PASTA_BASE / "oilSupplyNewsShocks_2025M06.xlsx"

PASTA_SAIDA = PASTA_RESULTADOS / "modelo_2_lp_iv_kanzig_completo"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

LAGS = 3

base = pd.read_excel(ARQUIVO_BASE, sheet_name="Sheet1")

base["Data"] = pd.to_datetime(base["Data"], errors="coerce")
base = base.dropna(subset=["Data"]).sort_values("Data")
base = base[
    (base["Data"] >= DATA_INICIO)
    & (base["Data"] <= "2025-06-01")
].copy()
base = base.reset_index(drop=True)

kanzig = pd.read_excel(
    ARQUIVO_KANZIG,
    sheet_name="Monthly"
)

kanzig.columns = [
    str(coluna).strip()
    for coluna in kanzig.columns
]

kanzig["Data"] = pd.to_datetime(
    kanzig["Date"]
    .astype(str)
    .str.strip()
    .str.replace("M", "-", regex=False)
    + "-01",
    errors="coerce"
)

kanzig["choque_kanzig"] = pd.to_numeric(
    kanzig["Oil supply news shock"],
    errors="coerce"
)

kanzig = kanzig[
    ["Data", "choque_kanzig"]
].copy()

base = base.merge(
    kanzig,
    on="Data",
    how="left"
)

base["choque_kanzig"] = base["choque_kanzig"].fillna(0)

base["dlog_petroleo_usd"] = 100 * np.log(base["Preco_Barril"]).diff()
base["dlog_cambio"] = 100 * np.log(base["Cambio"]).diff()
base["dlog_atividade"] = 100 * np.log(base["Atividade"]).diff()

base["selic"] = pd.to_numeric(base["Selic.1"], errors="coerce")
base["expectativa"] = pd.to_numeric(
    base["Expectativa_inflacao"],
    errors="coerce"
)

desvio_petroleo = base["dlog_petroleo_usd"].std()
desvio_kanzig = base["choque_kanzig"].std()

base["petroleo_endogeno"] = (
    base["dlog_petroleo_usd"]
    / desvio_petroleo
)

base["instrumento_kanzig"] = (
    base["choque_kanzig"]
    / desvio_kanzig
)

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

for lag in range(1, LAGS + 1):

    base[f"petroleo_lag{lag}"] = (
        base["petroleo_endogeno"].shift(lag)
    )

    base[f"kanzig_lag{lag}"] = (
        base["instrumento_kanzig"].shift(lag)
    )

    for controle in controles:

        base[f"{controle}_lag{lag}"] = (
            base[controle].shift(lag)
        )

todos_resultados = []
primeiros_estagios = []

for nome_resultado, coluna_y in variaveis_dependentes.items():

    print("")
    print("=" * 75)
    print(f"LP-IV Känzig -> {nome_resultado}")
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

        exogenas = [
            "dlog_cambio",
            "dlog_atividade",
            "selic",
            "expectativa",
        ]

        for lag in range(1, LAGS + 1):

            exogenas.append(f"y_lag{lag}")
            exogenas.append(f"petroleo_lag{lag}")
            exogenas.append(f"kanzig_lag{lag}")

            for controle in controles:

                exogenas.append(f"{controle}_lag{lag}")

        exogenas = exogenas + colunas_dummies

        colunas_regressao = (
            [f"y_acumulado_h{h}"]
            + ["petroleo_endogeno"]
            + ["instrumento_kanzig"]
            + exogenas
        )

        dados_regressao = base_modelo[
            colunas_regressao
        ].replace([np.inf, -np.inf], np.nan).dropna()

        y = dados_regressao[f"y_acumulado_h{h}"]

        X_exogenas = sm.add_constant(
            dados_regressao[exogenas],
            has_constant="add"
        )

        X_endogena = dados_regressao[
            ["petroleo_endogeno"]
        ]

        Z_instrumento = dados_regressao[
            ["instrumento_kanzig"]
        ]

        resultado_iv = IV2SLS(
            dependent=y,
            exog=X_exogenas,
            endog=X_endogena,
            instruments=Z_instrumento
        ).fit(
            cov_type="kernel",
            kernel="bartlett",
            bandwidth=max(1, h + 1)
        )

        X_primeiro_estagio = sm.add_constant(
            dados_regressao[
                ["instrumento_kanzig"] + exogenas
            ],
            has_constant="add"
        )

        primeiro_estagio = sm.OLS(
            dados_regressao["petroleo_endogeno"],
            X_primeiro_estagio
        ).fit(
            cov_type="HC1"
        )

        teste_f = primeiro_estagio.f_test(
            "instrumento_kanzig = 0"
        )

        coeficiente = resultado_iv.params["petroleo_endogeno"]
        erro_padrao = resultado_iv.std_errors["petroleo_endogeno"]

        todos_resultados.append({
            "variavel": nome_resultado,
            "h": h,
            "coeficiente": coeficiente,
            "erro_padrao": erro_padrao,
            "estatistica_t": resultado_iv.tstats["petroleo_endogeno"],
            "p_valor": resultado_iv.pvalues["petroleo_endogeno"],
            "limite_inferior_90": coeficiente - Z_90 * erro_padrao,
            "limite_superior_90": coeficiente + Z_90 * erro_padrao,
            "estatistica_f_primeiro_estagio": float(
                np.asarray(teste_f.fvalue).squeeze()
            ),
            "p_primeiro_estagio": float(
                np.asarray(teste_f.pvalue).squeeze()
            ),
            "coeficiente_instrumento_primeiro_estagio": (
                primeiro_estagio.params["instrumento_kanzig"]
            ),
            "p_instrumento_primeiro_estagio": (
                primeiro_estagio.pvalues["instrumento_kanzig"]
            ),
            "observacoes": int(resultado_iv.nobs),
        })

        if h in HORIZONTES_ARTIGO:

            primeiros_estagios.append({
                "variavel": nome_resultado,
                "h": h,
                "coeficiente_instrumento": (
                    primeiro_estagio.params["instrumento_kanzig"]
                ),
                "erro_padrao_instrumento": (
                    primeiro_estagio.bse["instrumento_kanzig"]
                ),
                "p_valor_instrumento": (
                    primeiro_estagio.pvalues["instrumento_kanzig"]
                ),
                "estatistica_f": float(
                    np.asarray(teste_f.fvalue).squeeze()
                ),
            })

        base_modelo = base_modelo.drop(
            columns=colunas_futuras
        )

tabela_final = pd.DataFrame(todos_resultados)

tabela_final.to_excel(
    PASTA_SAIDA / "resultados_consolidados_modelo_2.xlsx",
    index=False
)

tabela_artigo = tabela_final[
    tabela_final["h"].isin(HORIZONTES_ARTIGO)
].copy()

tabela_artigo.to_excel(
    PASTA_SAIDA / "tabela_resumida_artigo_modelo_2.xlsx",
    index=False
)

pd.DataFrame(primeiros_estagios).to_excel(
    PASTA_SAIDA / "primeiro_estagio_modelo_2.xlsx",
    index=False
)

for nome_resultado in variaveis_dependentes.keys():

    tabela_grafico = tabela_final[
        tabela_final["variavel"] == nome_resultado
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
    plt.title(f"LP-IV com Känzig -> {nome_resultado}")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA / f"grafico_lp_iv_{nome_resultado}.png",
        dpi=300
    )

    plt.close()

print("")
print("Modelo 2 concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
