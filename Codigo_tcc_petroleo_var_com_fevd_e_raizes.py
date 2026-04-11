
# ============================================================
# TCC - Petróleo, combustíveis e IPCA brasileiro
# Autor: Pedro Franck Minella (adaptado para a sua base)
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # permite salvar figuras sem abrir janela
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tools.sm_exceptions import InterpolationWarning

# ------------------------------------------------------------
# CONFIGURAÇÕES
# ------------------------------------------------------------
FILE_PATH = r"IPCA.xlsx"   # ajuste se necessário
SHEET_NAME = "Sheet1"
START_DATE = "2003-01-01"
MAX_LAGS = 12
IRF_HORIZON = 24

# Se quiser, troque para um caminho absoluto, por exemplo:
# FILE_PATH = r"C:\Users\pedro\OneDrive\Documentos\TCC\IPCA.xlsx"

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", InterpolationWarning)


# ------------------------------------------------------------
# PASTAS DE SAÍDA
# ------------------------------------------------------------
BASE_OUT = Path("output_tcc_var")
DIR_GRAFICOS = BASE_OUT / "graficos"
DIR_GRAFICOS_NIVEL = DIR_GRAFICOS / "01_nivel"
DIR_GRAFICOS_LOG = DIR_GRAFICOS / "02_log"
DIR_GRAFICOS_DLN = DIR_GRAFICOS / "03_dln"
DIR_RESID = DIR_GRAFICOS / "04_residuos"
DIR_IRF = DIR_GRAFICOS / "05_irf"
DIR_FEVD = DIR_GRAFICOS / "06_fevd"
DIR_RAIZES = DIR_GRAFICOS / "07_raizes_unitarias"
DIR_TABELAS = BASE_OUT / "tabelas"
DIR_MODELOS = BASE_OUT / "modelos"

for p in [
    BASE_OUT, DIR_GRAFICOS, DIR_GRAFICOS_NIVEL, DIR_GRAFICOS_LOG, DIR_GRAFICOS_DLN,
    DIR_RESID, DIR_IRF, DIR_FEVD, DIR_RAIZES, DIR_TABELAS, DIR_MODELOS
]:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------
def reconstruir_indice_por_variacao(series_pct, base=100.0):
    """
    Reconstrói um índice em nível a partir de uma série de variação mensal em %.
    Ex.: se a série for 1.2, -0.4, 0.8, cria um índice base 100.
    """
    s = pd.to_numeric(series_pct, errors="coerce")
    return base * (1 + s / 100.0).cumprod()


def garantir_numerico(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def carregar_e_preparar_dados(filepath, sheet_name="Sheet1", start_date="2003-01-01"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # datas
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data").copy()

    # amostra a partir de 2003 (início do IBC-Br)
    df = df[df["Data"] >= pd.Timestamp(start_date)].reset_index(drop=True).copy()

    # colunas numéricas principais
    cols_numericas = [
        "IPCA_Geral_nivel", "Var_IPCA_Brasil", "IPCA_Trans_nivel",
        "Cambio", "Preco_Barril", "Atividade",
        "GasolinaABrasil_media_nivel", "Gasolina_nivel", "Etanol_nivel", "Oleo_diesel_nivel"
    ]
    df = garantir_numerico(df, cols_numericas)

    # --------------------------------------------------------
    # PADRONIZAÇÃO DAS SÉRIES EM NÍVEL / ÍNDICE
    # --------------------------------------------------------
       # --------------------------------------------------------
    # PADRONIZAÇÃO DAS SÉRIES EM NÍVEL / ÍNDICE
    # --------------------------------------------------------
    # Usando diretamente as colunas em nível que já existem na planilha
    df["LV_IPCA_Brasil"] = df["IPCA_Geral_nivel"]
    df["LV_Cambio"] = df["Cambio"]
    df["LV_Preco_Barril"] = df["Preco_Barril"]
    df["LV_Atividade"] = df["Atividade"]
    df["LV_IPCA_Trans"] = df["IPCA_Trans_nivel"]
    df["LV_GasolinaA"] = df["GasolinaABrasil_media_nivel"]
    df["LV_Gasolina"] = df["Gasolina_nivel"]
    df["LV_Etanol"] = df["Etanol_nivel"]
    df["LV_Oleo_diesel"] = df["Oleo_diesel_nivel"]

    # Alias útil
    df["LV_Diesel"] = df["LV_Oleo_diesel"] 

    # Lista final de séries padronizadas em nível
    level_cols = [
        "LV_IPCA_Brasil",
        "LV_Cambio",
        "LV_Preco_Barril",
        "LV_Atividade",
        "LV_IPCA_Trans",
        "LV_GasolinaA",
        "LV_Gasolina",
        "LV_Etanol",
        "LV_Oleo_diesel",
    ]

    # Checagem de positividade (necessária para log)
    for c in level_cols:
        if (df[c] <= 0).any():
            raise ValueError(
                f"A coluna {c} possui valores <= 0. Não é possível tirar log sem tratamento prévio."
            )

    # logs, diferenças simples e diferenças logarítmicas
    for c in level_cols:
        nome = c.replace("LV_", "")
        df[f"LN_{nome}"] = np.log(df[c])
        df[f"D_{nome}"] = df[c].diff()
        df[f"DLN_{nome}"] = df[f"LN_{nome}"].diff()

    # --------------------------------------------------------
    # DUMMIES SAZONAIS E COVID
    # --------------------------------------------------------
    # Dummies mensais: janeiro fica como categoria base
    dummies_mensais = pd.get_dummies(df["Data"].dt.month, prefix="M", drop_first=True, dtype=int)
    dummies_mensais.index = df.index

    # Covid: aqui criei duas dummies
    # 1) choque inicial mais forte
    df["D_COVID_SHOCK"] = (
        (df["Data"] >= "2020-03-01") & (df["Data"] <= "2020-05-01")
    ).astype(int)

    # 2) período Covid mais amplo
    df["D_COVID_PERIOD"] = (
        (df["Data"] >= "2020-03-01") & (df["Data"] <= "2021-12-01")
    ).astype(int)

    exog = pd.concat(
        [
            df[["DLN_Atividade", "D_COVID_SHOCK", "D_COVID_PERIOD"]],
            dummies_mensais
        ],
        axis=1
    )

    return df, level_cols, exog


def plotar_serie(data, y, titulo, ylabel, caminho_saida):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data, y, linewidth=1.8)
    ax.set_title(titulo, fontsize=13)
    ax.set_xlabel("Data")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)


def gerar_graficos_descritivos(df, dict_series):
    """
    Gera gráficos em:
    1) nível
    2) log
    3) primeira diferença logarítmica (dln)
    """
    for nome_legivel, nome_base in dict_series.items():
        # nível
        plotar_serie(
            df["Data"], df[f"LV_{nome_base}"],
            f"{nome_legivel} - nível",
            nome_legivel,
            DIR_GRAFICOS_NIVEL / f"{nome_base}_nivel.png"
        )

        # log
        plotar_serie(
            df["Data"], df[f"LN_{nome_base}"],
            f"{nome_legivel} - logaritmo",
            f"ln({nome_legivel})",
            DIR_GRAFICOS_LOG / f"{nome_base}_log.png"
        )

        # dln
        plotar_serie(
            df["Data"], df[f"DLN_{nome_base}"],
            f"{nome_legivel} - primeira diferença do log",
            f"dln({nome_legivel})",
            DIR_GRAFICOS_DLN / f"{nome_base}_dln.png"
        )


def rodar_testes_unit_root(series, nome_variavel, transformacao, adf_reg="c", kpss_reg="c"):
    """
    ADF: H0 = raiz unitária
    KPSS: H0 = estacionária
    """
    s = pd.to_numeric(series, errors="coerce").dropna()

    out = {
        "variavel": nome_variavel,
        "transformacao": transformacao,
        "n_obs": len(s),
        "adf_stat": np.nan,
        "adf_pvalue": np.nan,
        "adf_lags": np.nan,
        "kpss_stat": np.nan,
        "kpss_pvalue": np.nan,
        "kpss_lags": np.nan
    }

    if len(s) < 20:
        return out

    try:
        adf_res = adfuller(s, regression=adf_reg, autolag="AIC")
        out["adf_stat"] = adf_res[0]
        out["adf_pvalue"] = adf_res[1]
        out["adf_lags"] = adf_res[2]
    except Exception:
        pass

    try:
        kpss_res = kpss(s, regression=kpss_reg, nlags="auto")
        out["kpss_stat"] = kpss_res[0]
        out["kpss_pvalue"] = kpss_res[1]
        out["kpss_lags"] = kpss_res[2]
    except Exception:
        pass

    return out


def gerar_tabelas_estacionariedade(df, dict_series):
    """
    Gera duas tabelas:
    1) nível vs primeira diferença de nível
    2) log vs primeira diferença do log
    """
    resultados_level_diff = []
    resultados_log_dlog = []

    for nome_legivel, nome_base in dict_series.items():
        # Nível: para séries em nível, faz sentido usar tendência no nível
        resultados_level_diff.append(
            rodar_testes_unit_root(
                df[f"LV_{nome_base}"], nome_legivel, "nivel",
                adf_reg="ct", kpss_reg="ct"
            )
        )
        resultados_level_diff.append(
            rodar_testes_unit_root(
                df[f"D_{nome_base}"], nome_legivel, "primeira_diferenca_nivel",
                adf_reg="c", kpss_reg="c"
            )
        )

        # Log e dlog
        resultados_log_dlog.append(
            rodar_testes_unit_root(
                df[f"LN_{nome_base}"], nome_legivel, "log_nivel",
                adf_reg="ct", kpss_reg="ct"
            )
        )
        resultados_log_dlog.append(
            rodar_testes_unit_root(
                df[f"DLN_{nome_base}"], nome_legivel, "primeira_diferenca_log",
                adf_reg="c", kpss_reg="c"
            )
        )

    tab1 = pd.DataFrame(resultados_level_diff)
    tab2 = pd.DataFrame(resultados_log_dlog)

    tab1.to_excel(DIR_TABELAS / "testes_estacionariedade_nivel_e_diff.xlsx", index=False)
    tab2.to_excel(DIR_TABELAS / "testes_estacionariedade_log_e_dlog.xlsx", index=False)

    return tab1, tab2


def tabela_johansen(level_df, det_order=0, k_ar_diff=1):
    joh = coint_johansen(level_df, det_order=det_order, k_ar_diff=k_ar_diff)

    rows = []
    for r in range(level_df.shape[1]):
        rows.append({
            "r": r,
            "trace_stat": joh.trace_stat[r],
            "trace_cv_90": joh.trace_stat_crit_vals[r, 0],
            "trace_cv_95": joh.trace_stat_crit_vals[r, 1],
            "trace_cv_99": joh.trace_stat_crit_vals[r, 2],
            "maxeig_stat": joh.max_eig_stat[r],
            "maxeig_cv_90": joh.max_eig_stat_crit_vals[r, 0],
            "maxeig_cv_95": joh.max_eig_stat_crit_vals[r, 1],
            "maxeig_cv_99": joh.max_eig_stat_crit_vals[r, 2],
        })
    return pd.DataFrame(rows)


def testes_bg_por_equacao(var_res, nlags=12):
    """
    BG é rodado equação por equação, usando a mesma matriz de regressores
    do VAR para cada equação.
    """
    X = pd.DataFrame(var_res.endog_lagged, columns=var_res.exog_names)
    Y = var_res.model.endog[var_res.k_ar:, :]

    resultados = []
    for i, eq in enumerate(var_res.names):
        ols_eq = sm.OLS(Y[:, i], X).fit()
        lm_stat, lm_pvalue, f_stat, f_pvalue = acorr_breusch_godfrey(ols_eq, nlags=nlags)
        resultados.append({
            "equacao": eq,
            "bg_lm_stat": lm_stat,
            "bg_lm_pvalue": lm_pvalue,
            "bg_f_stat": f_stat,
            "bg_f_pvalue": f_pvalue,
        })

    return pd.DataFrame(resultados)


def plotar_residuos_var(var_res, nome_modelo):
    resid = pd.DataFrame(var_res.resid, columns=var_res.names)

    n = resid.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for i, col in enumerate(resid.columns):
        axes[i].plot(resid.index, resid[col], linewidth=1.2)
        axes[i].axhline(0, linestyle="--", linewidth=1)
        axes[i].set_title(f"Resíduo - {col}")
        axes[i].grid(alpha=0.3)

    fig.suptitle(f"Resíduos do VAR - {nome_modelo}", fontsize=13)
    fig.tight_layout()
    fig.savefig(DIR_RESID / f"residuos_{nome_modelo}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plotar_fevd(var_res, nome_modelo, horizon=24):
    """
    Gera gráficos de FEVD para cada variável endógena.
    Mostra, ao longo do horizonte, a fração da variância do erro de previsão
    explicada por choques em cada variável do sistema.
    """
    fevd = var_res.fevd(horizon)
    decomposicao = fevd.decomp  # [nvars, horizonte, nvars]
    nomes = var_res.names
    h = np.arange(1, decomposicao.shape[1] + 1)

    for i, resposta in enumerate(nomes):
        fig, ax = plt.subplots(figsize=(10, 5))
        for j, impulso in enumerate(nomes):
            ax.plot(h, decomposicao[i, :, j], linewidth=2, label=impulso)

        ax.set_title(f"FEVD - {resposta} ({nome_modelo})", fontsize=13)
        ax.set_xlabel("Horizonte")
        ax.set_ylabel("Proporção da variância")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(DIR_FEVD / f"fevd_{nome_modelo}_{resposta}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)



def plotar_circulo_raizes_unitarias(var_res, nome_modelo):
    """
    Plota o círculo unitário e as raízes inversas do polinômio característico.
    Em statsmodels, resultado.roots traz as raízes; para o gráfico de estabilidade,
    é mais intuitivo plotar os inversos delas e verificar se ficam dentro do círculo unitário.
    """
    roots = np.asarray(var_res.roots)
    inv_roots = 1 / roots

    theta = np.linspace(0, 2 * np.pi, 500)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_circle, y_circle, linestyle="--", linewidth=1.5, label="Círculo unitário")
    ax.scatter(inv_roots.real, inv_roots.imag, s=60, alpha=0.9)

    for i, z in enumerate(inv_roots, start=1):
        ax.annotate(str(i), (z.real, z.imag), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Círculo unitário e raízes inversas - {nome_modelo}", fontsize=13)
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginária")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    limite = max(1.1, np.max(np.abs(inv_roots)) * 1.15)
    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)

    fig.tight_layout()
    fig.savefig(DIR_RAIZES / f"raizes_unitarias_{nome_modelo}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plotar_irf_resposta(var_res, nome_modelo, response_var, horizon=24):
    """
    Gera IRFs ortogonalizadas (Cholesky) para a variável resposta.
    Como a ordem do VAR é:
    [dln_oil, dln_cambio, dln_combustivel, dln_ipca_resposta],
    a identificação de Cholesky respeita exatamente essa ordem.
    """
    irf = var_res.irf(horizon)
    irfs = irf.orth_irfs  # [horizonte+1, nvars, nvars]

    nomes = var_res.names
    idx_resp = nomes.index(response_var)

    impulses = nomes[:-1]  # oil, cambio, combustivel
    h = np.arange(irfs.shape[0])

    fig, axes = plt.subplots(1, len(impulses), figsize=(5 * len(impulses), 4), sharey=True)
    if len(impulses) == 1:
        axes = [axes]

    for ax, imp in zip(axes, impulses):
        idx_imp = nomes.index(imp)
        ax.plot(h, irfs[:, idx_resp, idx_imp], linewidth=2)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"Resposta de {response_var}\na choque em {imp}")
        ax.set_xlabel("Horizonte")
        ax.grid(alpha=0.3)

    fig.suptitle(f"IRFs ortogonalizadas (Cholesky) - {nome_modelo}", fontsize=13)
    fig.tight_layout()
    fig.savefig(DIR_IRF / f"irf_{nome_modelo}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def selecionar_lag_var(modelo_var, maxlags=12):
    order = modelo_var.select_order(maxlags=maxlags)

    # prioriza AIC; se não vier, cai para BIC
    lag = order.selected_orders.get("aic", None)

    if lag is None or pd.isna(lag) or lag < 1:
        lag = order.selected_orders.get("bic", None)

    if lag is None or pd.isna(lag) or lag < 1:
        lag = 1

    return int(lag), order


def rodar_var_completo(df, exog):
    """
    Roda 8 VARs:
    4 combustíveis x 2 respostas (IPCA geral e IPCA transporte)
    Ordem de Cholesky:
        dln_oil
        dln_cambio
        dln_combustivel
        dln_ipca_resposta
    Exógenas:
        dln_atividade
        dummies mensais
        dummies covid
    """
    responses = {
        "IPCA_Brasil": "LV_IPCA_Brasil",
        "IPCA_Transporte": "LV_IPCA_Trans",
    }

    combustiveis = {
        "GasolinaA": "LV_GasolinaA",
        "Gasolina": "LV_Gasolina",
        "Etanol": "LV_Etanol",
        "Oleo_diesel": "LV_Oleo_diesel",
    }

    tabela_lags = []
    tabela_granger = []
    tabela_diag = []
    tabela_joh_total = []

    for nome_resp, col_resp in responses.items():
        for nome_comb, col_comb in combustiveis.items():

            nome_modelo = f"{nome_comb}_{nome_resp}"

            endog_cols = [
                "DLN_Preco_Barril",
                "DLN_Cambio",
                f"DLN_{col_comb.replace('LV_', '')}",
                f"DLN_{col_resp.replace('LV_', '')}",
            ]

            level_cols_model = [
                "LV_Preco_Barril",
                "LV_Cambio",
                col_comb,
                col_resp,
            ]

            base_modelo = pd.concat(
                [df[["Data"] + endog_cols + level_cols_model], exog],
                axis=1
            ).dropna().copy()

            endog = base_modelo[endog_cols]
            exog_model = base_modelo[exog.columns]

            # VAR com exógenas
            modelo = VAR(endog, exog=exog_model)
            lag_escolhido, ordem = selecionar_lag_var(modelo, maxlags=MAX_LAGS)
            resultado = modelo.fit(lag_escolhido)

            # salvar resumo textual completo
            with open(DIR_MODELOS / f"resumo_{nome_modelo}.txt", "w", encoding="utf-8") as f:
                f.write(f"===== RESUMO DO MODELO: {nome_modelo} =====\n\n")
                f.write(str(resultado.summary()))
                f.write("\n\n===== CRITÉRIOS DE SELEÇÃO DE DEFASAGEM =====\n\n")
                f.write(str(ordem.summary()))
                f.write("\n\n===== RAÍZES DO VAR =====\n\n")
                f.write(str(resultado.roots))
                f.write("\n\nModelo estável? ")
                f.write(str(resultado.is_stable(verbose=False)))

            # tabela de lags e informação do modelo
            tabela_lags.append({
                "modelo": nome_modelo,
                "lag_aic_escolhido": lag_escolhido,
                "n_obs": int(resultado.nobs),
                "AIC": resultado.aic,
                "BIC": resultado.bic,
                "HQIC": resultado.hqic,
                "estavel": resultado.is_stable(verbose=False)
            })

            # -----------------------------
            # Teste de Johansen (em níveis)
            # -----------------------------
            # k_ar_diff do Johansen costuma ser p-1
            k_ar_diff_joh = max(lag_escolhido - 1, 0)
            joh_tab = tabela_johansen(
                base_modelo[level_cols_model],
                det_order=0,      # constante
                k_ar_diff=k_ar_diff_joh
            )
            joh_tab["modelo"] = nome_modelo
            tabela_joh_total.append(joh_tab)

            # -----------------------------
            # Testes F (Granger causality)
            # -----------------------------
            target = endog_cols[-1]
            hipoteses = {
                f"{endog_cols[0]} -> {target}": [endog_cols[0]],
                f"{endog_cols[1]} -> {target}": [endog_cols[1]],
                f"{endog_cols[2]} -> {target}": [endog_cols[2]],
                f"{endog_cols[0]} + {endog_cols[2]} -> {target}": [endog_cols[0], endog_cols[2]],
            }

            for nome_hip, causing in hipoteses.items():
                teste_f = resultado.test_causality(target, causing, kind="f")
                tabela_granger.append({
                    "modelo": nome_modelo,
                    "hipotese_nula": nome_hip.replace(" -> ", " não causam Granger em "),
                    "estatistica_F": teste_f.test_statistic,
                    "pvalue": teste_f.pvalue,
                    "gl": str(teste_f.df)
                })

            # -----------------------------
            # Diagnósticos dos resíduos
            # -----------------------------
            # Observação:
            # - test_whiteness = Portmanteau (tipo Ljung-Box multivariado)
            # - BG = Breusch-Godfrey por equação
            # - normality = normalidade multivariada
            whiteness = resultado.test_whiteness(nlags=12)
            normality = resultado.test_normality()
            bg_tab = testes_bg_por_equacao(resultado, nlags=12)

            for _, row in bg_tab.iterrows():
                tabela_diag.append({
                    "modelo": nome_modelo,
                    "equacao": row["equacao"],
                    "portmanteau_ljungbox_pvalue_sistema": whiteness.pvalue,
                    "normalidade_residuos_pvalue_sistema": normality.pvalue,
                    "bg_lm_pvalue_equacao": row["bg_lm_pvalue"],
                    "bg_f_pvalue_equacao": row["bg_f_pvalue"],
                })

            # -----------------------------
            # Gráficos dos resíduos
            # -----------------------------
            plotar_residuos_var(resultado, nome_modelo)

            # -----------------------------
            # IRFs ortogonalizadas (Cholesky)
            # -----------------------------
            plotar_irf_resposta(resultado, nome_modelo, response_var=target, horizon=IRF_HORIZON)

            # -----------------------------
            # FEVD
            # -----------------------------
            plotar_fevd(resultado, nome_modelo, horizon=IRF_HORIZON)

            # -----------------------------
            # Círculo unitário / raízes inversas
            # -----------------------------
            plotar_circulo_raizes_unitarias(resultado, nome_modelo)

    tab_lags = pd.DataFrame(tabela_lags)
    tab_granger = pd.DataFrame(tabela_granger)
    tab_diag = pd.DataFrame(tabela_diag)
    tab_joh = pd.concat(tabela_joh_total, ignore_index=True)

    tab_lags.to_excel(DIR_TABELAS / "00_resumo_lags_e_estabilidade.xlsx", index=False)
    tab_granger.to_excel(DIR_TABELAS / "01_testes_F_granger.xlsx", index=False)
    tab_diag.to_excel(DIR_TABELAS / "02_diagnosticos_residuos.xlsx", index=False)
    tab_joh.to_excel(DIR_TABELAS / "03_teste_johansen.xlsx", index=False)

    return tab_lags, tab_granger, tab_diag, tab_joh


# ------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Lendo base e preparando variáveis...")
    df, level_cols, exog = carregar_e_preparar_dados(
        filepath=FILE_PATH,
        sheet_name=SHEET_NAME,
        start_date=START_DATE
    )

    # Mapa das séries pedidas por você
    series_plot = {
        "IPCA Brasil": "IPCA_Brasil",
        "Câmbio": "Cambio",
        "Preço do barril": "Preco_Barril",
        "Atividade (IBC-Br)": "Atividade",
        "IPCA Transporte": "IPCA_Trans",
        "Gasolina A (refinaria Petrobras)": "GasolinaA",
        "Gasolina (consumidor)": "Gasolina",
        "Etanol": "Etanol",
        "Óleo diesel": "Oleo_diesel",
    }

    print("Gerando gráficos em nível, log e primeira diferença do log...")
    gerar_graficos_descritivos(df, series_plot)

    print("Rodando ADF e KPSS...")
    tab1, tab2 = gerar_tabelas_estacionariedade(df, series_plot)

    print("Rodando os 8 VARs, Johansen, F-Granger e diagnósticos...")
    tab_lags, tab_granger, tab_diag, tab_joh = rodar_var_completo(df, exog)

    print("\nConcluído.")
    print(f"Saídas salvas em: {BASE_OUT.resolve()}")