# ============================================================
# TCC - Petróleo, combustíveis e IPCA brasileiro
# Versão VECM
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.vector_ar.vecm import VECM, select_order, coint_johansen
from statsmodels.tools.sm_exceptions import InterpolationWarning

# ------------------------------------------------------------
# CONFIGURAÇÕES
# ------------------------------------------------------------
FILE_PATH = str(Path(__file__).resolve().parent / "IPCA.xlsx")
SHEET_NAME = "Sheet1"
START_DATE = "2003-01-01"
MAX_LAGS = 12
IRF_HORIZON = 24
SIGNIF_JOHANSEN = 0.05
DETERMINISTIC = "ci"   # constante dentro da relação de cointegração
DET_ORDER_JOHANSEN = 0  # 0 = constante no teste de Johansen

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", InterpolationWarning)

# ------------------------------------------------------------
# PASTAS DE SAÍDA
# ------------------------------------------------------------
BASE_OUT = Path("output_tcc_vecm_Modelo2")
DIR_GRAFICOS = BASE_OUT / "graficos"
DIR_RESID = DIR_GRAFICOS / "01_residuos"
DIR_IRF = DIR_GRAFICOS / "02_irf"
DIR_TABELAS = BASE_OUT / "tabelas"
DIR_MODELOS = BASE_OUT / "modelos"
DIR_PARAMS = BASE_OUT / "parametros"

for p in [BASE_OUT, DIR_GRAFICOS, DIR_RESID, DIR_IRF, DIR_TABELAS, DIR_MODELOS, DIR_PARAMS]:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------
def reconstruir_indice_por_variacao(series_pct, base=100.0):
    s = pd.to_numeric(series_pct, errors="coerce")
    return base * (1 + s / 100.0).cumprod()


def garantir_numerico(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def inferir_rank_trace(johansen_result, signif=0.05):
    sig_to_col = {0.10: 0, 0.05: 1, 0.01: 2}
    col = sig_to_col.get(signif, 1)
    rank = 0
    for r in range(len(johansen_result.trace_stat)):
        if johansen_result.trace_stat[r] > johansen_result.trace_stat_crit_vals[r, col]:
            rank = r + 1
        else:
            break
    return int(rank)


def tabela_johansen(johansen_result):
    rows = []
    for r in range(len(johansen_result.trace_stat)):
        rows.append({
            "r": r,
            "trace_stat": johansen_result.trace_stat[r],
            "trace_cv_90": johansen_result.trace_stat_crit_vals[r, 0],
            "trace_cv_95": johansen_result.trace_stat_crit_vals[r, 1],
            "trace_cv_99": johansen_result.trace_stat_crit_vals[r, 2],
            "maxeig_stat": johansen_result.max_eig_stat[r],
            "maxeig_cv_90": johansen_result.max_eig_stat_crit_vals[r, 0],
            "maxeig_cv_95": johansen_result.max_eig_stat_crit_vals[r, 1],
            "maxeig_cv_99": johansen_result.max_eig_stat_crit_vals[r, 2],
        })
    return pd.DataFrame(rows)


def carregar_e_preparar_dados(filepath, sheet_name="Sheet1", start_date="2003-01-01"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data").copy()
    df = df[df["Data"] >= pd.Timestamp(start_date)].reset_index(drop=True).copy()

    cols_numericas = [
        "IPCA_Brasil", "Var_IPCA_Brasil", "Var_IPCA_Trans", "Var_IPCA_trans",
        "Cambio", "Preco_Barril", "Atividade", "GasolinaABrasil_media",
        "Gasolina", "Etanol", "Oleo_diesel"
    ]
    df = garantir_numerico(df, cols_numericas)

    trans_col = None
    if "Var_IPCA_Trans" in df.columns and df["Var_IPCA_Trans"].notna().sum() > 0:
        trans_col = "Var_IPCA_Trans"
    elif "Var_IPCA_trans" in df.columns and df["Var_IPCA_trans"].notna().sum() > 0:
        trans_col = "Var_IPCA_trans"
    else:
        raise ValueError("Não encontrei a coluna de variação do IPCA transporte na planilha.")

    # séries em nível / índice
    df["LV_IPCA_Brasil"] = df["IPCA_Brasil"]
    df["LV_Cambio"] = df["Cambio"]
    df["LV_Preco_Barril"] = df["Preco_Barril"]
    df["LV_Atividade"] = df["Atividade"]
    df["LV_GasolinaA"] = df["GasolinaABrasil_media"]

    df["LV_IPCA_Trans"] = reconstruir_indice_por_variacao(df[trans_col])
    df["LV_Gasolina"] = reconstruir_indice_por_variacao(df["Gasolina"])
    df["LV_Etanol"] = reconstruir_indice_por_variacao(df["Etanol"])
    df["LV_Oleo_diesel"] = reconstruir_indice_por_variacao(df["Oleo_diesel"])

    level_cols = [
        "LV_IPCA_Brasil", "LV_Cambio", "LV_Preco_Barril", "LV_Atividade",
        "LV_IPCA_Trans", "LV_GasolinaA", "LV_Gasolina", "LV_Etanol", "LV_Oleo_diesel"
    ]

    for c in level_cols:
        if (df[c] <= 0).any():
            raise ValueError(f"A coluna {c} possui valores <= 0. Não é possível usar log.")

    for c in level_cols:
        nome = c.replace("LV_", "")
        df[f"LN_{nome}"] = np.log(df[c])
        df[f"DLN_{nome}"] = df[f"LN_{nome}"].diff()

    # exógenas fora da relação de cointegração
    dummies_mensais = pd.get_dummies(df["Data"].dt.month, prefix="M", drop_first=True, dtype=float)
    dummies_mensais.index = df.index

    df["D_COVID_SHOCK"] = ((df["Data"] >= "2020-03-01") & (df["Data"] <= "2020-05-01")).astype(float)
    df["D_COVID_PERIOD"] = ((df["Data"] >= "2020-03-01") & (df["Data"] <= "2021-12-01")).astype(float)

    exog = pd.concat(
        [
            df[["DLN_Atividade", "D_COVID_SHOCK", "D_COVID_PERIOD"]],
            dummies_mensais,
        ],
        axis=1,
    )

    return df, exog


def escolher_k_ar_diff(endog, exog, maxlags=12):
    order = select_order(endog, maxlags=maxlags, deterministic=DETERMINISTIC, exog=exog)
    selected = order.selected_orders

    # Em VECM, priorizo BIC por parcimônia; se falhar, HQIC e AIC.
    for criterio in ["bic", "hqic", "aic", "fpe"]:
        lag = selected.get(criterio, None)
        if lag is not None and pd.notna(lag) and int(lag) >= 1:
            return int(lag), selected, criterio

    return 1, selected, "fallback_1"


def escolher_rank_e_lag_johansen(endog_log_levels, k_inicial, signif=0.05):
    candidatos = [k_inicial, 1, 2, 3]
    usados = []

    for k in candidatos:
        if k in usados or k < 1:
            continue
        usados.append(k)
        joh = coint_johansen(endog_log_levels, det_order=DET_ORDER_JOHANSEN, k_ar_diff=k)
        rank = inferir_rank_trace(joh, signif=signif)
        if rank > 0:
            return k, rank, joh

    # se não encontrar cointegração, devolve o teste do k inicial
    joh_final = coint_johansen(endog_log_levels, det_order=DET_ORDER_JOHANSEN, k_ar_diff=max(k_inicial, 1))
    rank_final = inferir_rank_trace(joh_final, signif=signif)
    return max(k_inicial, 1), rank_final, joh_final


def plotar_residuos_vecm(res, nome_modelo):
    resid = pd.DataFrame(res.resid, columns=res.names)
    n = resid.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(resid.columns):
        axes[i].plot(resid.index, resid[col], linewidth=1.2)
        axes[i].axhline(0, linestyle="--", linewidth=1)
        axes[i].set_title(f"Resíduo - {col}")
        axes[i].grid(alpha=0.3)

    fig.suptitle(f"Resíduos do VECM - {nome_modelo}", fontsize=13)
    fig.tight_layout()
    fig.savefig(DIR_RESID / f"residuos_{nome_modelo}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plotar_irf_resposta_vecm(res, nome_modelo, response_var, horizon=24):
    irf = res.irf(periods=horizon)
    irfs = irf.orth_irfs

    nomes = list(res.names)
    idx_resp = nomes.index(response_var)
    impulses = nomes[:-1]
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

    fig.suptitle(f"IRFs ortogonalizadas - VECM - {nome_modelo}", fontsize=13)
    fig.tight_layout()
    fig.savefig(DIR_IRF / f"irf_{nome_modelo}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def salvar_parametros_vecm(res, nome_modelo):
    rank = res.alpha.shape[1]
    ect_cols = [f"ECT_{i+1}" for i in range(rank)]

    alpha = pd.DataFrame(res.alpha, index=res.names, columns=ect_cols)
    alpha_p = pd.DataFrame(res.pvalues_alpha, index=res.names, columns=ect_cols)
    beta = pd.DataFrame(res.beta, index=res.names, columns=ect_cols)
    beta_p = pd.DataFrame(res.pvalues_beta, index=res.names, columns=ect_cols)

    gamma_cols = []
    neqs = len(res.names)
    k_ar_diff = res.model.k_ar_diff
    if k_ar_diff > 0:
        for lag in range(1, k_ar_diff + 1):
            for var in res.names:
                gamma_cols.append(f"L{lag}.D_{var}")
        gamma = pd.DataFrame(res.gamma, index=res.names, columns=gamma_cols)
        gamma_p = pd.DataFrame(res.pvalues_gamma, index=res.names, columns=gamma_cols)
    else:
        gamma = pd.DataFrame(index=res.names)
        gamma_p = pd.DataFrame(index=res.names)

    with pd.ExcelWriter(DIR_PARAMS / f"parametros_{nome_modelo}.xlsx") as writer:
        alpha.to_excel(writer, sheet_name="alpha")
        alpha_p.to_excel(writer, sheet_name="alpha_pvalues")
        beta.to_excel(writer, sheet_name="beta")
        beta_p.to_excel(writer, sheet_name="beta_pvalues")
        gamma.to_excel(writer, sheet_name="gamma")
        gamma_p.to_excel(writer, sheet_name="gamma_pvalues")


def rodar_vecm_completo(df, exog):
    responses = {
        "IPCA_Brasil": "IPCA_Brasil",
        "IPCA_Transporte": "IPCA_Trans",
    }

    combustiveis = {
        "GasolinaA": "GasolinaA",
        "Gasolina": "Gasolina",
        "Etanol": "Etanol",
        "Oleo_diesel": "Oleo_diesel",
    }

    tabela_modelos = []
    tabela_granger = []
    tabela_diag = []
    tabela_joh_total = []
    tabela_status = []

    for nome_resp, resp_base in responses.items():
        for nome_comb, comb_base in combustiveis.items():
            nome_modelo = f"{nome_comb}_{nome_resp}"

            endog_cols = [
                "LN_Preco_Barril",
                "LN_Cambio",
                f"LN_{comb_base}",
                f"LN_{resp_base}",
            ]

            base_modelo = pd.concat([df[["Data"] + endog_cols], exog], axis=1).dropna().copy()
            base_modelo = base_modelo.set_index("Data")
            endog = base_modelo[endog_cols]
            exog_model = base_modelo[exog.columns]

            k_inicial, selected_orders, criterio_lag = escolher_k_ar_diff(endog, exog_model, maxlags=MAX_LAGS)
            k_usado, rank, joh = escolher_rank_e_lag_johansen(endog, k_inicial, signif=SIGNIF_JOHANSEN)

            joh_tab = tabela_johansen(joh)
            joh_tab["modelo"] = nome_modelo
            joh_tab["k_ar_diff_usado"] = k_usado
            tabela_joh_total.append(joh_tab)

            if rank == 0:
                tabela_status.append({
                    "modelo": nome_modelo,
                    "status": "sem_cointegracao_em_log_niveis",
                    "k_inicial_select_order": k_inicial,
                    "criterio_lag": criterio_lag,
                    "k_usado_johansen": k_usado,
                    "rank_trace_5pct": rank,
                })
                continue

            vecm = VECM(
                endog=endog,
                exog=exog_model,
                k_ar_diff=k_usado,
                coint_rank=rank,
                deterministic=DETERMINISTIC,
            )
            res = vecm.fit()

            with open(DIR_MODELOS / f"resumo_{nome_modelo}.txt", "w", encoding="utf-8") as f:
                f.write(f"===== RESUMO DO VECM: {nome_modelo} =====\n\n")
                f.write(str(res.summary()))
                f.write("\n\n===== LAG SELECTION (select_order) =====\n\n")
                f.write(str(selected_orders))
                f.write(f"\n\nCritério usado para k_ar_diff: {criterio_lag}")
                f.write(f"\nk_ar_diff selecionado inicialmente: {k_inicial}")
                f.write(f"\nk_ar_diff usado no VECM: {k_usado}")
                f.write(f"\nRank de cointegração (trace, 5%): {rank}")
                f.write("\n\n===== BETA (vetores de cointegração) =====\n")
                f.write(f"\n{pd.DataFrame(res.beta, index=res.names).to_string()}\n")
                f.write("\n===== ALPHA (velocidade de ajuste) =====\n")
                f.write(f"\n{pd.DataFrame(res.alpha, index=res.names).to_string()}\n")

            salvar_parametros_vecm(res, nome_modelo)

            tabela_modelos.append({
                "modelo": nome_modelo,
                "k_ar_diff": k_usado,
                "rank_trace_5pct": rank,
                "n_obs": int(res.nobs),
                "llf": float(res.llf),
                "criterio_lag": criterio_lag,
                "k_inicial_select_order": k_inicial,
                "k_usado_johansen": k_usado,
                "status": "estimado",
            })

            target = endog_cols[-1]
            hipoteses = {
                f"{endog_cols[0]} -> {target}": [endog_cols[0]],
                f"{endog_cols[1]} -> {target}": [endog_cols[1]],
                f"{endog_cols[2]} -> {target}": [endog_cols[2]],
                f"{endog_cols[0]} + {endog_cols[2]} -> {target}": [endog_cols[0], endog_cols[2]],
            }

            for nome_hip, causing in hipoteses.items():
                teste = res.test_granger_causality(caused=target, causing=causing, signif=0.05)
                tabela_granger.append({
                    "modelo": nome_modelo,
                    "hipotese_nula": nome_hip.replace(" -> ", " não causam Granger em "),
                    "estatistica": teste.test_statistic,
                    "pvalue": teste.pvalue,
                    "gl": str(teste.df),
                })

            whiteness = res.test_whiteness(nlags=12)
            normality = res.test_normality()

            tabela_diag.append({
                "modelo": nome_modelo,
                "portmanteau_whiteness_pvalue": whiteness.pvalue,
                "normalidade_residuos_pvalue": normality.pvalue,
                "rank_trace_5pct": rank,
                "k_ar_diff": k_usado,
            })

            plotar_residuos_vecm(res, nome_modelo)
            plotar_irf_resposta_vecm(res, nome_modelo, response_var=target, horizon=IRF_HORIZON)

    tab_modelos = pd.DataFrame(tabela_modelos)
    tab_granger = pd.DataFrame(tabela_granger)
    tab_diag = pd.DataFrame(tabela_diag)
    tab_joh = pd.concat(tabela_joh_total, ignore_index=True) if tabela_joh_total else pd.DataFrame()
    tab_status = pd.DataFrame(tabela_status)

    tab_modelos.to_excel(DIR_TABELAS / "00_resumo_modelos_vecm.xlsx", index=False)
    tab_granger.to_excel(DIR_TABELAS / "01_testes_granger_vecm.xlsx", index=False)
    tab_diag.to_excel(DIR_TABELAS / "02_diagnosticos_residuos_vecm.xlsx", index=False)
    tab_joh.to_excel(DIR_TABELAS / "03_teste_johansen_log_niveis.xlsx", index=False)
    tab_status.to_excel(DIR_TABELAS / "04_modelos_pulados_sem_cointegracao.xlsx", index=False)

    return tab_modelos, tab_granger, tab_diag, tab_joh, tab_status


if __name__ == "__main__":
    print("Lendo base...")
    df, exog = carregar_e_preparar_dados(FILE_PATH, SHEET_NAME, START_DATE)

    print("Rodando VECMs...")
    tab_modelos, tab_granger, tab_diag, tab_joh, tab_status = rodar_vecm_completo(df, exog)

    print("\nConcluído.")
    print(f"Saídas salvas em: {BASE_OUT.resolve()}")
    if not tab_status.empty:
        print("\nModelos sem cointegração em log-níveis e, por isso, pulados:")
        print(tab_status.to_string(index=False))
