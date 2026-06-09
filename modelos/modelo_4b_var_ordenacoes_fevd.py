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
# MODELO 4B — VAR COM ORDENAÇÕES ALTERNATIVAS E FEVD AUDITADA
# ============================================================
#
# Ordenações:
# A: petróleo, câmbio, combustível, inflação
# B: petróleo, combustível, câmbio, inflação
# C: câmbio, petróleo, combustível, inflação
#
# As bandas bootstrap são calculadas para a ordem principal.
# As ordens alternativas são usadas para comparar sinal,
# magnitude e persistência sem esconder a hipótese recursiva.
# ============================================================

from statsmodels.tsa.api import VAR

PASTA_SAIDA = (
    PASTA_RESULTADOS
    / "modelo_4b_var_ordenacoes"
)
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

MAX_LAGS = 12
HORIZONTE_IRF = 12
REPLICACOES_BOOTSTRAP = 500

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

base["delta_selic"] = (
    pd.to_numeric(
        base["Selic.1"],
        errors="coerce"
    ).diff()
)

base["delta_expectativa"] = (
    pd.to_numeric(
        base["Expectativa_inflacao"],
        errors="coerce"
    ).diff()
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

combustiveis = {
    "gasolina_refinaria": (
        "Var_GasolinaABrasil_media"
    ),
    "gasolina_consumidor": "Var_Gasolina",
    "etanol": "Var_Etanol",
    "diesel": "Var_Oleo_diesel",
}

inflacoes = {
    "ipca_geral": "Var_IPCA_Geral",
    "ipca_transportes": "Var_IPCA_Trans",
}

ordenacoes = {
    "A_principal": [
        "petroleo",
        "cambio",
        "combustivel",
        "inflacao",
    ],
    "B_combustivel_antes_cambio": [
        "petroleo",
        "combustivel",
        "cambio",
        "inflacao",
    ],
    "C_cambio_antes_petroleo": [
        "cambio",
        "petroleo",
        "combustivel",
        "inflacao",
    ],
}

diagnosticos = []
irfs_ordenacoes = []
fevd_precisao = []
comparacao_ordenacoes = []

for nome_combustivel, coluna_combustivel in combustiveis.items():

    for nome_inflacao, coluna_inflacao in inflacoes.items():

        mapa_colunas = {
            "petroleo": "dlog_petroleo_usd",
            "cambio": "dlog_cambio",
            "combustivel": coluna_combustivel,
            "inflacao": coluna_inflacao,
        }

        colunas_exogenas = [
            "dlog_atividade",
            "delta_selic",
            "delta_expectativa",
        ] + colunas_dummies

        for nome_ordem, ordem in ordenacoes.items():

            print("")
            print("=" * 75)
            print(
                f"VAR: {nome_combustivel} "
                f"| {nome_inflacao} "
                f"| {nome_ordem}"
            )
            print("=" * 75)

            colunas_endogenas = [
                mapa_colunas[item]
                for item in ordem
            ]

            dados = base[
                ["Data"]
                + colunas_endogenas
                + colunas_exogenas
            ].replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            Y = dados[
                colunas_endogenas
            ].copy()

            X = dados[
                colunas_exogenas
            ].copy()

            seletor = VAR(
                Y,
                exog=X
            )

            selecao = seletor.select_order(
                maxlags=MAX_LAGS
            )

            lag_bic = (
                selecao
                .selected_orders
                .get("bic")
            )

            if (
                lag_bic is None
                or lag_bic < 1
            ):

                lag_bic = 1

            resultado = None

            for lag_teste in range(
                int(lag_bic),
                MAX_LAGS + 1
            ):

                tentativa = seletor.fit(
                    lag_teste
                )

                try:

                    port = (
                        tentativa
                        .test_whiteness(
                            nlags=max(
                                12,
                                lag_teste + 5
                            ),
                            adjusted=True
                        )
                    )

                    p_port = float(
                        port.pvalue
                    )

                except Exception:

                    p_port = np.nan

                if (
                    np.isfinite(p_port)
                    and p_port >= 0.05
                ):

                    resultado = tentativa
                    break

            if resultado is None:

                resultado = seletor.fit(
                    int(lag_bic)
                )

            lag_final = resultado.k_ar

            try:

                port = (
                    resultado
                    .test_whiteness(
                        nlags=max(
                            12,
                            lag_final + 5
                        ),
                        adjusted=True
                    )
                )

                port_p = float(
                    port.pvalue
                )

            except Exception:

                port_p = np.nan

            diagnosticos.append({
                "combustivel": nome_combustivel,
                "inflacao": nome_inflacao,
                "ordenacao": nome_ordem,
                "ordem_texto": " -> ".join(ordem),
                "lag_bic": int(lag_bic),
                "lag_final": int(lag_final),
                "estavel": bool(
                    resultado.is_stable(
                        verbose=False
                    )
                ),
                "portmanteau_p": port_p,
                "observacoes": int(
                    resultado.nobs
                ),
            })

            irf = resultado.irf(
                HORIZONTE_IRF
            )

            irf_ortogonal = (
                irf.orth_irfs
            )

            indice_petroleo = (
                ordem.index("petroleo")
            )

            indice_combustivel = (
                ordem.index("combustivel")
            )

            indice_inflacao = (
                ordem.index("inflacao")
            )

            for indice_resposta, nome_resposta in [
                (
                    indice_combustivel,
                    nome_combustivel
                ),
                (
                    indice_inflacao,
                    nome_inflacao
                ),
            ]:

                pontual = (
                    irf_ortogonal[
                        :,
                        indice_resposta,
                        indice_petroleo
                    ]
                )

                acumulada = np.cumsum(
                    pontual
                )

                for h in range(
                    len(acumulada)
                ):

                    irfs_ordenacoes.append({
                        "combustivel": nome_combustivel,
                        "inflacao": nome_inflacao,
                        "ordenacao": nome_ordem,
                        "resposta": nome_resposta,
                        "h": h,
                        "irf_pontual": pontual[h],
                        "irf_acumulada": acumulada[h],
                    })

                if nome_resposta == nome_combustivel:

                    linha_comparacao = {
                        "combustivel": nome_combustivel,
                        "inflacao": nome_inflacao,
                        "ordenacao": nome_ordem,
                    }

                    for h in HORIZONTES_ARTIGO:

                        linha_comparacao[
                            f"irf_combustivel_h{h}"
                        ] = acumulada[h]

                    comparacao_ordenacoes.append(
                        linha_comparacao
                    )

            # FEVD com precisão de 10 casas
            fevd = resultado.fevd(13)

            for h in [
                1,
                3,
                6,
                12
            ]:

                indice_h = min(
                    h,
                    fevd.decomp.shape[1] - 1
                )

                participacoes = (
                    fevd.decomp[
                        indice_inflacao,
                        indice_h,
                        :
                    ]
                )

                linha_fevd = {
                    "combustivel": nome_combustivel,
                    "inflacao": nome_inflacao,
                    "ordenacao": nome_ordem,
                    "h": h,
                }

                for indice, item in enumerate(ordem):

                    linha_fevd[
                        f"participacao_{item}"
                    ] = float(
                        participacoes[indice]
                    )

                fevd_precisao.append(
                    linha_fevd
                )

            # Bootstrap apenas na ordem principal
            if nome_ordem == "A_principal":

                residuos = np.asarray(
                    resultado.resid
                )

                coeficientes_ar = np.asarray(
                    resultado.coefs
                )

                coeficientes_exog = np.asarray(
                    resultado.coefs_exog
                )

                y_original = np.asarray(Y)
                x_original = np.asarray(X)

                gerador = (
                    np.random.default_rng(
                        12345
                    )
                )

                simulacoes = []

                for repeticao in range(
                    REPLICACOES_BOOTSTRAP
                ):

                    indices = (
                        gerador.integers(
                            0,
                            len(residuos),
                            size=len(residuos)
                        )
                    )

                    residuos_boot = (
                        residuos[indices]
                    )

                    y_boot = (
                        y_original.copy()
                    )

                    for t in range(
                        lag_final,
                        len(y_boot)
                    ):

                        estimado = np.zeros(
                            y_boot.shape[1]
                        )

                        for lag_ar in range(
                            1,
                            lag_final + 1
                        ):

                            estimado += (
                                coeficientes_ar[
                                    lag_ar - 1
                                ]
                                @ y_boot[
                                    t - lag_ar
                                ]
                            )

                        x_constante = (
                            np.concatenate(
                                (
                                    np.array([1.0]),
                                    x_original[t]
                                )
                            )
                        )

                        estimado += (
                            coeficientes_exog
                            @ x_constante
                        )

                        residuo_indice = min(
                            t - lag_final,
                            len(residuos_boot) - 1
                        )

                        y_boot[t] = (
                            estimado
                            + residuos_boot[
                                residuo_indice
                            ]
                        )

                    try:

                        resultado_boot = VAR(
                            y_boot,
                            exog=x_original
                        ).fit(
                            lag_final
                        )

                        simulacoes.append(
                            resultado_boot
                            .irf(
                                HORIZONTE_IRF
                            )
                            .orth_irfs
                        )

                    except Exception:

                        continue

                simulacoes = np.asarray(
                    simulacoes
                )

                if len(simulacoes) >= 100:

                    for indice_resposta, nome_resposta in [
                        (
                            indice_combustivel,
                            nome_combustivel
                        ),
                        (
                            indice_inflacao,
                            nome_inflacao
                        ),
                    ]:

                        draws = (
                            simulacoes[
                                :,
                                :,
                                indice_resposta,
                                indice_petroleo
                            ]
                        )

                        draws_acumulados = (
                            np.cumsum(
                                draws,
                                axis=1
                            )
                        )

                        inferior = np.quantile(
                            draws_acumulados,
                            0.05,
                            axis=0
                        )

                        superior = np.quantile(
                            draws_acumulados,
                            0.95,
                            axis=0
                        )

                        ponto = np.cumsum(
                            irf_ortogonal[
                                :,
                                indice_resposta,
                                indice_petroleo
                            ]
                        )

                        plt.figure(
                            figsize=(9, 5)
                        )

                        plt.plot(
                            range(
                                HORIZONTE_IRF + 1
                            ),
                            ponto,
                            marker="o"
                        )

                        plt.fill_between(
                            range(
                                HORIZONTE_IRF + 1
                            ),
                            inferior,
                            superior,
                            alpha=0.20
                        )

                        plt.axhline(
                            0,
                            linewidth=1
                        )

                        plt.xlabel(
                            "Horizonte, em meses"
                        )

                        plt.ylabel(
                            "Resposta acumulada"
                        )

                        plt.title(
                            f"VAR principal: petróleo -> "
                            f"{nome_resposta}"
                        )

                        plt.grid(alpha=0.25)
                        plt.tight_layout()

                        plt.savefig(
                            PASTA_SAIDA
                            / (
                                f"irf_principal_"
                                f"{nome_combustivel}_"
                                f"{nome_inflacao}_"
                                f"{nome_resposta}.png"
                            ),
                            dpi=300
                        )

                        plt.close()

pd.DataFrame(
    diagnosticos
).to_excel(
    PASTA_SAIDA
    / "diagnosticos_ordenacoes_var.xlsx",
    index=False
)

pd.DataFrame(
    irfs_ordenacoes
).to_excel(
    PASTA_SAIDA
    / "irfs_todas_ordenacoes.xlsx",
    index=False
)

pd.DataFrame(
    comparacao_ordenacoes
).to_excel(
    PASTA_SAIDA
    / "comparacao_ordenacoes_horizontes.xlsx",
    index=False
)

tabela_fevd = pd.DataFrame(
    fevd_precisao
)

tabela_fevd.to_excel(
    PASTA_SAIDA
    / "fevd_precisao_10_casas.xlsx",
    index=False,
    float_format="%.10f"
)

# Auditoria automática da igualdade h=6 versus h=12
auditoria_fevd = []

for (
    combustivel,
    inflacao,
    ordenacao
), grupo in tabela_fevd.groupby([
    "combustivel",
    "inflacao",
    "ordenacao",
]):

    linha_6 = grupo[
        grupo["h"] == 6
    ]

    linha_12 = grupo[
        grupo["h"] == 12
    ]

    if (
        not linha_6.empty
        and not linha_12.empty
    ):

        colunas_participacao = [
            coluna
            for coluna in grupo.columns
            if coluna.startswith(
                "participacao_"
            )
        ]

        maior_diferenca = max(
            abs(
                float(
                    linha_6.iloc[0][coluna]
                )
                - float(
                    linha_12.iloc[0][coluna]
                )
            )
            for coluna in colunas_participacao
        )

        auditoria_fevd.append({
            "combustivel": combustivel,
            "inflacao": inflacao,
            "ordenacao": ordenacao,
            "maior_diferenca_h6_h12": maior_diferenca,
            "iguais_ate_2_casas": (
                maior_diferenca < 0.005
            ),
            "iguais_ate_6_casas": (
                maior_diferenca < 0.0000005
            ),
            "interpretacao": (
                "Convergência praticamente completa"
                if maior_diferenca < 0.0000005
                else (
                    "Coincidência apenas após arredondamento"
                    if maior_diferenca < 0.005
                    else "Ainda há mudança relevante"
                )
            ),
        })

pd.DataFrame(
    auditoria_fevd
).to_excel(
    PASTA_SAIDA
    / "auditoria_convergencia_fevd.xlsx",
    index=False
)

print("")
print("Modelo 4B concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
