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
# MODELO 2B — LP-IV COM PRIMEIRA ETAPA DETALHADA
# ============================================================
#
# Saídas:
# - uma linha por variável e horizonte;
# - coeficiente, erro-padrão e F exato;
# - forma reduzida;
# - teste Anderson-Rubin para H0: beta = 0.
# ============================================================

from linearmodels.iv import IV2SLS

ARQUIVO_KANZIG = (
    PASTA_BASE
    / "oilSupplyNewsShocks_2025M06.xlsx"
)

PASTA_SAIDA = (
    PASTA_RESULTADOS
    / "modelo_2b_lp_iv_detalhado"
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

base["choque_kanzig"] = (
    base["choque_kanzig"]
    .fillna(0)
)

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

base["petroleo_endogeno"] = (
    base["dlog_petroleo_usd"]
    / base["dlog_petroleo_usd"].std()
)

base["instrumento_kanzig"] = (
    base["choque_kanzig"]
    / base["choque_kanzig"].std()
)

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

segunda_etapa = []
primeira_etapa = []
forma_reduzida = []

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

        exogenas = [
            "dlog_cambio",
            "dlog_atividade",
            "selic",
            "expectativa",
        ]

        for lag in range(1, LAGS + 1):

            exogenas += [
                f"y_lag{lag}",
                f"petroleo_lag{lag}",
                f"kanzig_lag{lag}",
            ]

            for controle in controles:

                exogenas.append(
                    f"{controle}_lag{lag}"
                )

        exogenas += colunas_dummies

        colunas = (
            [f"y_acumulado_h{h}"]
            + ["petroleo_endogeno"]
            + ["instrumento_kanzig"]
            + exogenas
        )

        amostra = dados[
            colunas
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        y = amostra[
            f"y_acumulado_h{h}"
        ]

        X_exog = sm.add_constant(
            amostra[exogenas],
            has_constant="add"
        )

        resultado_iv = IV2SLS(
            dependent=y,
            exog=X_exog,
            endog=amostra[
                ["petroleo_endogeno"]
            ],
            instruments=amostra[
                ["instrumento_kanzig"]
            ]
        ).fit(
            cov_type="kernel",
            kernel="bartlett",
            bandwidth=max(1, h + 1)
        )

        coef_iv = (
            resultado_iv.params[
                "petroleo_endogeno"
            ]
        )

        se_iv = (
            resultado_iv.std_errors[
                "petroleo_endogeno"
            ]
        )

        segunda_etapa.append({
            "variavel": nome_variavel,
            "h": h,
            "coeficiente_iv": coef_iv,
            "erro_padrao_iv": se_iv,
            "p_valor_iv": (
                resultado_iv.pvalues[
                    "petroleo_endogeno"
                ]
            ),
            "limite_inferior_90": (
                coef_iv - Z_90 * se_iv
            ),
            "limite_superior_90": (
                coef_iv + Z_90 * se_iv
            ),
            "observacoes": int(
                resultado_iv.nobs
            ),
        })

        # Primeira etapa exata
        X_primeiro = sm.add_constant(
            amostra[
                ["instrumento_kanzig"]
                + exogenas
            ],
            has_constant="add"
        )

        resultado_primeiro = sm.OLS(
            amostra["petroleo_endogeno"],
            X_primeiro
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": max(1, h + 1)
            }
        )

        teste_f = (
            resultado_primeiro
            .f_test(
                "instrumento_kanzig = 0"
            )
        )

        primeira_etapa.append({
            "variavel": nome_variavel,
            "h": h,
            "coeficiente_instrumento": (
                resultado_primeiro.params[
                    "instrumento_kanzig"
                ]
            ),
            "erro_padrao_instrumento": (
                resultado_primeiro.bse[
                    "instrumento_kanzig"
                ]
            ),
            "estatistica_t_instrumento": (
                resultado_primeiro.tvalues[
                    "instrumento_kanzig"
                ]
            ),
            "p_valor_instrumento": (
                resultado_primeiro.pvalues[
                    "instrumento_kanzig"
                ]
            ),
            "estatistica_f_robusta": float(
                np.asarray(
                    teste_f.fvalue
                ).squeeze()
            ),
            "p_valor_f": float(
                np.asarray(
                    teste_f.pvalue
                ).squeeze()
            ),
            "r_quadrado_primeira_etapa": (
                resultado_primeiro.rsquared
            ),
            "observacoes": int(
                resultado_primeiro.nobs
            ),
        })

        # Forma reduzida e Anderson-Rubin para H0 beta=0.
        # Em um modelo exatamente identificado,
        # testar o instrumento na regressão reduzida de y
        # equivale ao AR para beta0=0.
        X_reduzida = sm.add_constant(
            amostra[
                ["instrumento_kanzig"]
                + exogenas
            ],
            has_constant="add"
        )

        resultado_reduzido = sm.OLS(
            y,
            X_reduzida
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": max(1, h + 1)
            }
        )

        teste_ar = (
            resultado_reduzido
            .f_test(
                "instrumento_kanzig = 0"
            )
        )

        forma_reduzida.append({
            "variavel": nome_variavel,
            "h": h,
            "coeficiente_forma_reduzida": (
                resultado_reduzido.params[
                    "instrumento_kanzig"
                ]
            ),
            "erro_padrao_forma_reduzida": (
                resultado_reduzido.bse[
                    "instrumento_kanzig"
                ]
            ),
            "p_valor_forma_reduzida": (
                resultado_reduzido.pvalues[
                    "instrumento_kanzig"
                ]
            ),
            "anderson_rubin_f_beta_igual_zero": float(
                np.asarray(
                    teste_ar.fvalue
                ).squeeze()
            ),
            "anderson_rubin_p_beta_igual_zero": float(
                np.asarray(
                    teste_ar.pvalue
                ).squeeze()
            ),
            "observacoes": int(
                resultado_reduzido.nobs
            ),
        })

        dados = dados.drop(
            columns=colunas_futuras
        )

pd.DataFrame(
    segunda_etapa
).to_excel(
    PASTA_SAIDA
    / "segunda_etapa_lp_iv_detalhada.xlsx",
    index=False
)

pd.DataFrame(
    primeira_etapa
).to_excel(
    PASTA_SAIDA
    / "primeira_etapa_detalhada_por_variavel_h.xlsx",
    index=False
)

pd.DataFrame(
    forma_reduzida
).to_excel(
    PASTA_SAIDA
    / "forma_reduzida_anderson_rubin.xlsx",
    index=False
)

pd.DataFrame(
    primeira_etapa
)[
    pd.DataFrame(
        primeira_etapa
    )["h"].isin(
        HORIZONTES_ARTIGO
    )
].to_excel(
    PASTA_SAIDA
    / "tabela_artigo_primeira_etapa_exata.xlsx",
    index=False
)

print("")
print("Modelo 2B concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
