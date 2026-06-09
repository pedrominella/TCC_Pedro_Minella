
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


from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss

try:
    from arch.unitroot import PhillipsPerron, DFGLS
    TEM_ARCH = True
except Exception:
    TEM_ARCH = False

# ============================================================
# MODELO 4 — VAR DE ROBUSTEZ
# ============================================================

PASTA_SAIDA = PASTA_RESULTADOS / "modelo_4_var_robustez_completo"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

MAX_LAGS = 12
HORIZONTE_IRF = 24

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

base["delta_selic"] = pd.to_numeric(
    base["Selic.1"],
    errors="coerce"
).diff()

base["delta_expectativa"] = pd.to_numeric(
    base["Expectativa_inflacao"],
    errors="coerce"
).diff()

base["mes"] = base["Data"].dt.month

dummies_mes = pd.get_dummies(
    base["mes"],
    prefix="mes",
    drop_first=True,
    dtype=float
)

base = pd.concat([base, dummies_mes], axis=1)
colunas_dummies = list(dummies_mes.columns)

variaveis_teste = {
    "dlog_petroleo_usd": "dlog_petroleo_usd",
    "dlog_cambio": "dlog_cambio",
    "dlog_atividade": "dlog_atividade",
    "delta_selic": "delta_selic",
    "delta_expectativa": "delta_expectativa",
    "gasolina_refinaria": "Var_GasolinaABrasil_media",
    "gasolina_consumidor": "Var_Gasolina",
    "etanol": "Var_Etanol",
    "diesel": "Var_Oleo_diesel",
    "ipca_geral": "Var_IPCA_Geral",
    "ipca_transportes": "Var_IPCA_Trans",
}

testes_estacionariedade = []

for nome, coluna in variaveis_teste.items():

    serie = pd.to_numeric(
        base[coluna],
        errors="coerce"
    ).dropna()

    try:

        adf = adfuller(
            serie,
            autolag="AIC",
            regression="c"
        )

        adf_estatistica = adf[0]
        adf_p_valor = adf[1]

    except Exception:

        adf_estatistica = np.nan
        adf_p_valor = np.nan

    try:

        kpss_resultado = kpss(
            serie,
            regression="c",
            nlags="auto"
        )

        kpss_estatistica = kpss_resultado[0]
        kpss_p_valor = kpss_resultado[1]

    except Exception:

        kpss_estatistica = np.nan
        kpss_p_valor = np.nan

    if TEM_ARCH:

        try:
            pp = PhillipsPerron(serie)
            pp_estatistica = pp.stat
            pp_p_valor = pp.pvalue
        except Exception:
            pp_estatistica = np.nan
            pp_p_valor = np.nan

        try:
            dfgls = DFGLS(serie)
            dfgls_estatistica = dfgls.stat
            dfgls_p_valor = dfgls.pvalue
        except Exception:
            dfgls_estatistica = np.nan
            dfgls_p_valor = np.nan

    else:

        pp_estatistica = np.nan
        pp_p_valor = np.nan
        dfgls_estatistica = np.nan
        dfgls_p_valor = np.nan

    testes_estacionariedade.append({
        "variavel": nome,
        "observacoes": len(serie),
        "adf_estatistica": adf_estatistica,
        "adf_p_valor": adf_p_valor,
        "kpss_estatistica": kpss_estatistica,
        "kpss_p_valor": kpss_p_valor,
        "pp_estatistica": pp_estatistica,
        "pp_p_valor": pp_p_valor,
        "dfgls_estatistica": dfgls_estatistica,
        "dfgls_p_valor": dfgls_p_valor,
    })

pd.DataFrame(testes_estacionariedade).to_excel(
    PASTA_SAIDA / "testes_estacionariedade.xlsx",
    index=False
)

combustiveis = {
    "gasolina_refinaria": "Var_GasolinaABrasil_media",
    "gasolina_consumidor": "Var_Gasolina",
    "etanol": "Var_Etanol",
    "diesel": "Var_Oleo_diesel",
}

inflacoes = {
    "ipca_geral": "Var_IPCA_Geral",
    "ipca_transportes": "Var_IPCA_Trans",
}

diagnosticos = []
irfs = []
fevds = []

for nome_combustivel, coluna_combustivel in combustiveis.items():

    for nome_inflacao, coluna_inflacao in inflacoes.items():

        nome_modelo = (
            f"{nome_combustivel}_{nome_inflacao}"
        )

        print("")
        print("=" * 75)
        print(f"VAR: {nome_modelo}")
        print("=" * 75)

        colunas_endogenas = [
            "dlog_petroleo_usd",
            "dlog_cambio",
            coluna_combustivel,
            coluna_inflacao,
        ]

        colunas_exogenas = [
            "dlog_atividade",
            "delta_selic",
            "delta_expectativa",
        ] + colunas_dummies

        dados_modelo = base[
            ["Data"]
            + colunas_endogenas
            + colunas_exogenas
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        Y = dados_modelo[colunas_endogenas].copy()
        X = dados_modelo[colunas_exogenas].copy()

        seletor = VAR(Y, exog=X)

        selecao = seletor.select_order(
            maxlags=MAX_LAGS
        )

        lag_bic = selecao.selected_orders.get("bic")

        if lag_bic is None or lag_bic < 1:

            lag_bic = 1

        resultado_final = None

        for lag_teste in range(
            int(lag_bic),
            MAX_LAGS + 1
        ):

            if lag_teste < 1:

                continue

            resultado_teste = seletor.fit(
                lag_teste
            )

            try:

                portmanteau = resultado_teste.test_whiteness(
                    nlags=max(12, lag_teste + 5),
                    adjusted=True
                )

                p_portmanteau = portmanteau.pvalue

            except Exception:

                p_portmanteau = np.nan

            if (
                np.isfinite(p_portmanteau)
                and p_portmanteau >= 0.05
            ):

                resultado_final = resultado_teste
                break

        if resultado_final is None:

            resultado_final = seletor.fit(
                int(max(1, lag_bic))
            )

        resultado = resultado_final
        lag_final = resultado.k_ar

        try:

            portmanteau = resultado.test_whiteness(
                nlags=max(12, lag_final + 5),
                adjusted=True
            )

            portmanteau_estatistica = (
                portmanteau.test_statistic
            )

            portmanteau_p_valor = (
                portmanteau.pvalue
            )

        except Exception:

            portmanteau_estatistica = np.nan
            portmanteau_p_valor = np.nan

        try:

            normalidade = resultado.test_normality()

            normalidade_estatistica = (
                normalidade.test_statistic
            )

            normalidade_p_valor = (
                normalidade.pvalue
            )

        except Exception:

            normalidade_estatistica = np.nan
            normalidade_p_valor = np.nan

        try:

            granger_petroleo_combustivel = (
                resultado.test_causality(
                    caused=2,
                    causing=[0],
                    kind="wald"
                )
            )

            p_granger_petroleo_combustivel = (
                granger_petroleo_combustivel.pvalue
            )

        except Exception:

            p_granger_petroleo_combustivel = np.nan

        try:

            granger_combustivel_ipca = (
                resultado.test_causality(
                    caused=3,
                    causing=[2],
                    kind="wald"
                )
            )

            p_granger_combustivel_ipca = (
                granger_combustivel_ipca.pvalue
            )

        except Exception:

            p_granger_combustivel_ipca = np.nan

        try:

            granger_petroleo_ipca = (
                resultado.test_causality(
                    caused=3,
                    causing=[0],
                    kind="wald"
                )
            )

            p_granger_petroleo_ipca = (
                granger_petroleo_ipca.pvalue
            )

        except Exception:

            p_granger_petroleo_ipca = np.nan

        diagnosticos.append({
            "modelo": nome_modelo,
            "combustivel": nome_combustivel,
            "inflacao": nome_inflacao,
            "inicio": dados_modelo["Data"].min(),
            "fim": dados_modelo["Data"].max(),
            "observacoes": int(resultado.nobs),
            "lag_bic_inicial": int(lag_bic),
            "lag_final": int(lag_final),
            "estavel": bool(
                resultado.is_stable(
                    verbose=False
                )
            ),
            "portmanteau_estatistica": portmanteau_estatistica,
            "portmanteau_p_valor": portmanteau_p_valor,
            "normalidade_estatistica": normalidade_estatistica,
            "normalidade_p_valor": normalidade_p_valor,
            "granger_petroleo_para_combustivel_p": (
                p_granger_petroleo_combustivel
            ),
            "granger_combustivel_para_ipca_p": (
                p_granger_combustivel_ipca
            ),
            "granger_petroleo_para_ipca_p": (
                p_granger_petroleo_ipca
            ),
        })

        irf = resultado.irf(HORIZONTE_IRF)
        irf_ortogonal = irf.orth_irfs

        # ------------------------------------------------------------
        # BOOTSTRAP RESIDUAL CORRIGIDO PARA AS IRFs ACUMULADAS
        # ------------------------------------------------------------

        numero_replicacoes = 1000
        gerador = np.random.default_rng(12345)

        residuos = np.asarray(resultado.resid)
        coeficientes_ar = np.asarray(resultado.coefs)
        coeficientes_exog = np.asarray(resultado.coefs_exog)

        y_original = np.asarray(Y)
        x_original = np.asarray(X)

        simulacoes_irf = []

        for repeticao in range(numero_replicacoes):

            indices_residuos = gerador.integers(
                0,
                len(residuos),
                size=len(residuos)
            )

            residuos_sorteados = residuos[indices_residuos]

            y_boot = y_original.copy()

            inicio = lag_final

            for t in range(inicio, len(y_boot)):

                valor_estimado = np.zeros(
                    y_boot.shape[1]
                )

                for lag_ar in range(1, lag_final + 1):

                    valor_estimado = (
                        valor_estimado
                        + coeficientes_ar[lag_ar - 1]
                        @ y_boot[t - lag_ar]
                    )

                if coeficientes_exog.size > 0:

                    # coefs_exog inclui a constante e as variáveis exógenas.
                    x_t_com_constante = np.concatenate(
                        (
                            np.array([1.0]),
                            x_original[t]
                        )
                    )

                    valor_estimado = (
                        valor_estimado
                        + coeficientes_exog
                        @ x_t_com_constante
                    )

                indice_residuo = min(
                    t - inicio,
                    len(residuos_sorteados) - 1
                )

                y_boot[t] = (
                    valor_estimado
                    + residuos_sorteados[indice_residuo]
                )

            try:

                resultado_boot = VAR(
                    y_boot,
                    exog=x_original
                ).fit(lag_final)

                irf_boot = resultado_boot.irf(
                    HORIZONTE_IRF
                ).orth_irfs

                simulacoes_irf.append(irf_boot)

            except Exception:

                continue

        simulacoes_irf = np.asarray(simulacoes_irf)

        if simulacoes_irf.shape[0] < 200:

            raise RuntimeError(
                f"Poucas replicações válidas no bootstrap: "
                f"{simulacoes_irf.shape[0]}"
            )

        for indice_resposta, nome_resposta in [
            (2, nome_combustivel),
            (3, nome_inflacao),
        ]:

            resposta_pontual = (
                irf_ortogonal[
                    :,
                    indice_resposta,
                    0
                ]
            )

            resposta_acumulada = np.cumsum(
                resposta_pontual
            )

            simulacoes_resposta = (
                simulacoes_irf[
                    :,
                    :,
                    indice_resposta,
                    0
                ]
            )

            simulacoes_acumuladas = np.cumsum(
                simulacoes_resposta,
                axis=1
            )

            limite_inferior = np.quantile(
                simulacoes_acumuladas,
                0.05,
                axis=0
            )

            limite_superior = np.quantile(
                simulacoes_acumuladas,
                0.95,
                axis=0
            )

            for h in range(
                len(resposta_acumulada)
            ):

                irfs.append({
                    "modelo": nome_modelo,
                    "resposta": nome_resposta,
                    "h": h,
                    "irf_pontual": resposta_pontual[h],
                    "irf_acumulada": resposta_acumulada[h],
                    "limite_inferior_90": limite_inferior[h],
                    "limite_superior_90": limite_superior[h],
                })

            plt.figure(figsize=(9, 5))

            plt.plot(
                range(len(resposta_acumulada)),
                resposta_acumulada,
                marker="o"
            )

            plt.fill_between(
                range(len(resposta_acumulada)),
                limite_inferior,
                limite_superior,
                alpha=0.20
            )

            plt.axhline(0, linewidth=1)
            plt.xlabel("Horizonte, em meses")
            plt.ylabel("Resposta acumulada")
            plt.title(
                f"VAR: petróleo em dólares -> {nome_resposta}"
            )
            plt.grid(alpha=0.25)
            plt.tight_layout()

            plt.savefig(
                PASTA_SAIDA
                / f"irf_{nome_modelo}_{nome_resposta}.png",
                dpi=300
            )

            plt.close()

        try:

            fevd = resultado.fevd(13)

            for h in HORIZONTES_ARTIGO:

                indice_h = min(
                    h,
                    fevd.decomp.shape[1] - 1
                )

                participacoes = fevd.decomp[
                    3,
                    indice_h,
                    :
                ]

                fevds.append({
                    "modelo": nome_modelo,
                    "h": h,
                    "petroleo_usd": participacoes[0],
                    "cambio": participacoes[1],
                    "combustivel": participacoes[2],
                    "proprio_ipca": participacoes[3],
                })

        except Exception:

            pass

pd.DataFrame(diagnosticos).to_excel(
    PASTA_SAIDA / "diagnosticos_var.xlsx",
    index=False
)

pd.DataFrame(irfs).to_excel(
    PASTA_SAIDA / "irfs_acumuladas_var.xlsx",
    index=False
)

pd.DataFrame(fevds).to_excel(
    PASTA_SAIDA / "fevd_ipca_var.xlsx",
    index=False
)

tabela_artigo_var = pd.DataFrame(irfs)

tabela_artigo_var = tabela_artigo_var[
    tabela_artigo_var["h"].isin(
        HORIZONTES_ARTIGO
    )
].copy()

tabela_artigo_var.to_excel(
    PASTA_SAIDA / "tabela_resumida_artigo_var.xlsx",
    index=False
)

print("")
print("Modelo 4 concluído.")
print(f"Resultados em: {PASTA_SAIDA}")
