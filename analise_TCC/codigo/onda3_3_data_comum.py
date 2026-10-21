# -*- coding: utf-8 -*-
"""
UMA DATA SO, ESTIMADA PELOS DADOS, COM AS DUMMIES DE PANDEMIA MANTIDAS

As dummies de pandemia ficam nos controles de TODAS as especificacoes -- nenhum mes e
descartado da amostra. Sobre essa base:

(1) ESCOLHER a data. Em vez de uma data por variavel, estima-se uma QUEBRA COMUM ao
    sistema: para cada data candidata tau soma-se a estatistica F da igualdade de repasse
    das variaveis do conjunto, e toma-se o argmax. E o analogo multivariado do sup-Wald
    (na linha de Bai, Lumsdaine e Stock, 1998). Valor critico por wild block bootstrap com
    o MESMO sorteio aplicado a todas as series, para preservar a dependencia entre elas.

(2) ESTIMAR tudo na data escolhida: LP por regime para as seis variaveis, h = 0 a 12,
    lado a lado com o corte de out/2016.

CORRECOES DE 27/08 (ver LEIA-ME, secao "Auditoria")
---------------------------------------------------
(F1) INFERENCIA POS-SELECAO. Este era o problema mais serio do conjunto. A data e
     escolhida MAXIMIZANDO a estatistica F da igualdade de repasse em h=0; reportar
     depois o p-valor convencional de Wald DESSA MESMA hipotese, NESSA MESMA data, e
     circular. Medido na versao anterior: entre as 188 datas candidatas, mar/2017
     estava entre as 13 de menor p para IPCA Geral e entre as 2 menores para diesel --
     por construcao, nao por evidencia. A frase do LEIA-ME "todos os p-valores caem, a
     data estimada e uniformemente mais nitida que a imposta" media apenas que o argmax
     e o argmax.

     Agora ha um p-valor AJUSTADO POR SELECAO: sob H0 de ausencia de quebra, cada
     replica do bootstrap refaz A BUSCA INTEIRA, encontra seu proprio argmax tau*, e a
     estatistica da variavel e avaliada NAQUELE tau*. A distribuicao resultante e a
     distribuicao nula de "a estatistica na data que os dados escolheram", que e a
     pergunta certa. O p convencional continua na planilha, com o nome
     `p_naive_pos_selecao`, explicitamente rotulado como invalido para essa comparacao.

(F2) A grade de datas candidatas vinha de `list(ams.values())[0]`, com `ams` iterado a
     partir de um `set` de strings -- ordem nao deterministica entre execucoes. Agora as
     variaveis sao percorridas em ordem fixa e a grade e a intersecao explicita das datas.

(F3) A busca usava 1{t >= tau} e a estimacao aplicava `.shift(1)`, deslocando o regime um
     mes em relacao a data escolhida. Resolvido no modulo (C1).

(F4) Sorteio do bootstrap indexado por DATA, e nao por posicao, para que as replicas
     fiquem alinhadas entre variaveis E entre horizontes (as amostras tem tamanhos
     diferentes). Residuo corrigido por alavancagem.

Uso: python onda3_3_data_comum.py escolher | estimar AAAA-MM
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
import lp_base_nova as L

B = 299
BLOCO = 12
SEMENTE = 20260829
APARA = 0.15
H_ESCOLHA = 0
H_PERFIS = [0, 3, 6]          # horizontes com p ajustado por selecao
# Data do anuncio da PPI corrigida em 01/10/2026: a Petrobras anunciou a politica em 14/10/2016 e os
# novos precos valeram a partir de 15/10/2016 (Agencia Brasil, 14/10/2016). Nos dados mensais, out/2016.
DATA_PPI = pd.Timestamp("2016-10-01")
PRINCIPAIS = ["gasolina_consumidor", "diesel", "ipca_transportes", "ipca_geral"]
CONJUNTOS = {"4 principais": PRINCIPAIS,
             "5 com refinaria": PRINCIPAIS + ["gasolina_refinaria"],
             "so refinaria": ["gasolina_refinaria"]}
# (F2) ordem fixa, sem depender de iteracao de `set`
TODAS = sorted({n for v in CONJUNTOS.values() for n in v})


def sem_colunas_vazias(X):
    """Descarta colunas sem variacao MAS preserva a constante.

    `X.loc[:, X.std() > 0]` parece inocente e nao e: a coluna de uns criada por
    `add_constant` tem desvio-padrao zero e era eliminada junto, deixando a regressao
    SEM INTERCEPTO. Com as indicadoras mensais em drop_first, tirar o intercepto muda
    todos os coeficientes -- de 0,02 a 0,12 nos coeficientes de regime deste arquivo.
    O filtro existe para descartar dummies de pandemia vazias numa subamostra; e so
    isso que ele deve fazer.
    """
    manter = (X.std() > 0) | (X.columns == "const")
    return X.loc[:, manter]


def montar(nome, h, b, dum, pand):
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    regs = []
    for lag in range(1, L.LAGS + 1):
        regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in L.CONTROLES]
    regs += dum + pand(d, h)
    d["__y"] = L.y_acum(d, nome, h)
    am = d[["Data", "__y", "choque_petroleo"] + regs].replace(
        [np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    W = sm.add_constant(am[regs], has_constant="add")
    W = sem_colunas_vazias(W)
    return am, np.asarray(W, float), regs


def _perfil_F(am, W, cand, etas_df):
    """F(tau) observado e das replicas, para toda a grade de datas candidatas."""
    y = np.asarray(am["__y"], float)
    z = np.asarray(am["choque_petroleo"], float)
    datas = pd.DatetimeIndex(am["Data"])
    n = len(y)
    X0 = np.column_stack([z, W]); P0 = np.linalg.pinv(X0)
    ajus = X0 @ (P0 @ y)
    lev = np.clip(np.einsum("ij,ji->i", X0, P0), 0.0, 0.999)
    res = (y - ajus) / np.sqrt(1.0 - lev)                       # (F4)
    E = etas_df.reindex(datas).to_numpy()                       # T x B, alinhado por data
    Y = ajus[:, None] + res[:, None] * E

    s0o = float(((y - ajus) ** 2).sum())
    R0 = Y - X0 @ (P0 @ Y); s0b = (R0 * R0).sum(axis=0)
    fo = np.empty(len(cand)); fb = np.empty((B, len(cand)))
    for j, tau in enumerate(cand):
        pos = (datas >= tau).astype(float)
        Xt = np.column_stack([z * (1 - pos), z * pos, pos, W])
        Pt = np.linalg.pinv(Xt); k = Xt.shape[1]
        r = y - Xt @ (Pt @ y); s1o = float(r @ r)
        fo[j] = ((s0o - s1o) / 2) / (s1o / (n - k))
        R = Y - Xt @ (Pt @ Y); s1b = (R * R).sum(axis=0)
        fb[:, j] = ((s0b - s1b) / 2) / (s1b / (n - k))
    return fo, fb


def escolher():
    b, dum, pand = L.carregar_pandemia("corrigida")
    datas_ref = pd.DatetimeIndex(sorted(b["Data"].unique()))
    rng = np.random.default_rng(SEMENTE)
    nb = int(np.ceil(len(datas_ref) / BLOCO))
    etas_df = pd.DataFrame(
        np.column_stack([np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:len(datas_ref)]
                         for _ in range(B)]), index=datas_ref)

    # (F2) grade comum = intersecao explicita das datas, em ordem fixa
    ams = {(nome, h): montar(nome, h, b, dum, pand) for nome in TODAS for h in H_PERFIS}
    comuns = sorted(set.intersection(*[set(am["Data"]) for (am, _, _) in ams.values()]))
    n0 = len(comuns)
    cand = comuns[int(APARA * n0):int((1 - APARA) * n0)]

    Fobs, Fboot = {}, {}
    for (nome, h), (am, W, _) in ams.items():
        Fobs[(nome, h)], Fboot[(nome, h)] = _perfil_F(am, W, cand, etas_df)
        print(f"  perfil calculado: {nome} h={h}")

    linhas, perfis, ajust = [], [], []
    for rot, nomes in CONJUNTOS.items():
        soma = sum(Fobs[(n_, H_ESCOLHA)] for n_ in nomes)
        somab = sum(Fboot[(n_, H_ESCOLHA)] for n_ in nomes)
        j = int(np.argmax(soma)); tau = cand[j]; estat = float(soma[j])
        maxb = somab.max(axis=1)
        jstar = somab.argmax(axis=1)          # (F1) argmax de CADA replica
        crit = float(np.quantile(maxb, 0.95))
        dentro = [d_ for d_, v in zip(cand, soma) if v >= crit]
        linhas.append({"conjunto": rot, "n_series": len(nomes),
                       "data_escolhida": tau, "estatistica": estat,
                       "p_bootstrap": float((maxb >= estat).mean()),
                       "critico_95": crit,
                       "estatistica_em_out2016": float(soma[cand.index(DATA_PPI)])
                       if DATA_PPI in cand else np.nan,
                       "intervalo_inicio": min(dentro) if dentro else pd.NaT,
                       "intervalo_fim": max(dentro) if dentro else pd.NaT})
        for d_, v in zip(cand, soma):
            perfis.append({"conjunto": rot, "data": d_, "soma_F": v, "critico_95": crit})

        # (F1) p ajustado por selecao, variavel a variavel
        for nome in L.VARIAVEIS:
            for h in H_PERFIS:
                if (nome, h) not in Fobs:
                    ams[(nome, h)] = montar(nome, h, b, dum, pand)
                    am, W, _ = ams[(nome, h)]
                    Fobs[(nome, h)], Fboot[(nome, h)] = _perfil_F(am, W, cand, etas_df)
                f_obs = float(Fobs[(nome, h)][j])
                nulo = Fboot[(nome, h)][np.arange(B), jstar]     # F na data que a replica escolheu
                ajust.append({"conjunto": rot, "variavel": nome, "h": h,
                              "data_escolhida": tau, "F_na_data_escolhida": f_obs,
                              "p_ajustado_selecao": float((nulo >= f_obs).mean()),
                              "critico_95_ajustado": float(np.quantile(nulo, 0.95)),
                              "p_naive_qui2_2gl": float(np.exp(-f_obs))})
        print(f"  {rot:16s} data = {tau:%Y-%m}  estat = {estat:7.1f}  "
              f"p = {linhas[-1]['p_bootstrap']:.3f}")

    t = pd.DataFrame(linhas); pf = pd.DataFrame(perfis); aj = pd.DataFrame(ajust)
    with pd.ExcelWriter(SAIDA / "onda3_3_data_comum.xlsx") as w:
        t.to_excel(w, sheet_name="escolha", index=False)
        pf.to_excel(w, sheet_name="perfil", index=False)
        aj.to_excel(w, sheet_name="p_ajustado_selecao", index=False)
    print("\n=== escolha da data comum ===")
    print(t.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    print("\n=== (F1) p-valor AJUSTADO POR SELECAO na data escolhida ===")
    print("(conjunto '4 principais', que e o que define o corte adotado, mar/2017;")
    print(" sob H0 cada replica refaz a busca e a estatistica")
    print(" e avaliada no argmax DELA -- e a distribuicao nula correta)")
    v = aj[aj.conjunto == "4 principais"]   # coerencia: e o conjunto que da mar/2017
    print(v[["variavel", "h", "F_na_data_escolhida", "critico_95_ajustado",
             "p_ajustado_selecao"]].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\nrejeicoes a 5% com ajuste de selecao: "
          f"{int((v.p_ajustado_selecao < 0.05).sum())} de {len(v)}")
    return t, aj


def estimar(data_escolhida):
    tau = pd.Timestamp(data_escolhida)
    b, dum, pand = L.carregar_pandemia("corrigida")
    linhas = []
    for nome in L.VARIAVEIS:
        for rot, corte in [("escolhida", tau), ("out/2016", DATA_PPI)]:
            d = b.copy()
            d["regime_pos"] = L.regime(d.Data, corte)          # (F3)
            d["choque_pre"] = d["choque_petroleo"] * (1 - d["regime_pos"])
            d["choque_pos"] = d["choque_petroleo"] * d["regime_pos"]
            for lag in range(1, L.LAGS + 1):
                d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
            base = ["choque_pre", "choque_pos", "regime_pos"]
            for lag in range(1, L.LAGS + 1):
                base += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                        [f"{c}_lag{lag}" for c in L.CONTROLES]
            for h in range(0, L.H_MAX + 1):
                d["__y"] = L.y_acum(d, nome, h)
                regs = base + dum + pand(d, h)
                am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                X = sm.add_constant(am[regs], has_constant="add")
                X = sem_colunas_vazias(X)
                r = sm.OLS(am["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
                w = r.wald_test("choque_pre = choque_pos", scalar=True)
                # (F1.3) erro-padrao HAC de cada regime e da diferenca. A Tabela 2
                # do trabalho traz so coeficiente, asterisco e o p do Wald, o que
                # impede o leitor de avaliar se 0,005 e 0,176 diferem por magnitude
                # ou por precisao -- que e exatamente a afirmacao central do texto.
                # A variancia da diferenca usa a covariancia entre os dois, e nao a
                # soma das variancias.
                V = r.cov_params()
                ep_dif = float(np.sqrt(max(
                    V.loc["choque_pre", "choque_pre"] + V.loc["choque_pos", "choque_pos"]
                    - 2 * V.loc["choque_pre", "choque_pos"], 0.0)))
                n_pos = int(am["regime_pos"].sum())
                linhas.append({"variavel": nome, "corte": rot, "data_corte": corte, "h": h,
                               "coef_pre": r.params["choque_pre"], "p_pre": r.pvalues["choque_pre"],
                               "ep_pre": float(r.bse["choque_pre"]),
                               "coef_pos": r.params["choque_pos"], "p_pos": r.pvalues["choque_pos"],
                               "ep_pos": float(r.bse["choque_pos"]),
                               "diferenca": r.params["choque_pos"] - r.params["choque_pre"],
                               "ep_diferenca": ep_dif,
                               # (F1) nome explicito: este p NAO e valido na data estimada
                               "wald_p_naive_pos_selecao": float(w.pvalue),
                               "obs": int(r.nobs), "obs_pos": n_pos,
                               "obs_pre": int(r.nobs) - n_pos,
                               "n_parametros": int(X.shape[1])})
        print(f"  {nome}")
    t = pd.DataFrame(linhas)
    t.to_excel(SAIDA / "onda3_3_regimes_data_escolhida.xlsx", index=False)
    print(f"\n=== regime com corte em {tau:%Y-%m} contra out/2016 ===")
    v = t[t.h.isin(L.HORIZONTES)]
    print(v.pivot_table(index=["variavel", "h"], columns="corte",
                        values=["coef_pos", "wald_p_naive_pos_selecao"]).round(3).to_string())
    print("\nATENCAO: `wald_p_naive_pos_selecao` NAO e um p-valor valido na data ESTIMADA --")
    print("a data foi escolhida maximizando exatamente essa estatistica. Para a data")
    print("estimada, use `p_ajustado_selecao` da aba do arquivo onda3_3_data_comum.xlsx.")
    print("Na data IMPOSTA (out/2016), que nao foi escolhida pelos dados, ele e valido.")
    return t


def figuras():
    """Perfil da estatistica somada ao longo das datas candidatas.

    Mostra de uma vez as tres coisas que o texto precisa afirmar: onde esta o
    maximo (a data escolhida), que out/2016 tambem esta acima do valor critico
    (o corte antigo nao era erro), e que a regiao acima do critico e LARGA --
    os dados datam a quebra com folga de anos, nao de meses.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pf = pd.read_excel(SAIDA / "onda3_3_data_comum.xlsx", sheet_name="perfil")
    t = pd.read_excel(SAIDA / "onda3_3_data_comum.xlsx", sheet_name="escolha")
    pf["data"] = pd.to_datetime(pf["data"])
    cores = {"4 principais": "#0B5A56", "5 com refinaria": "#9E3D25", "so refinaria": "#5F6549"}

    fig, axes = plt.subplots(len(CONJUNTOS), 1, figsize=(7.4, 8.2), sharex=True)
    for ax, (rot, cor) in zip(axes, cores.items()):
        s = pf[pf.conjunto == rot].sort_values("data")
        r = t[t.conjunto == rot].iloc[0]
        tau = pd.Timestamp(r.data_escolhida)
        ax.plot(s.data, s.soma_F, color=cor, lw=1.7)
        ax.fill_between(s.data, r.critico_95, s.soma_F,
                        where=(s.soma_F >= r.critico_95), color=cor, alpha=0.15)
        ax.axhline(r.critico_95, color=cor, lw=0.9, ls=":")
        ax.plot([tau], [r.estatistica], "o", color=cor, ms=7, zorder=5)
        ax.annotate(f"{tau:%b/%Y}", xy=(tau, r.estatistica), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=cor, weight="bold")
        ax.axvline(DATA_PPI, color="0.25", lw=1.1, ls="--")
        if not pd.isna(r.estatistica_em_out2016):
            ax.plot([DATA_PPI], [r.estatistica_em_out2016], "s", color="0.25", ms=5, zorder=5)
        ax.set_ylabel("soma dos F", fontsize=8)
        ax.set_title(f"{rot}  --  escolhida {tau:%Y-%m}, p = {r.p_bootstrap:.3f}; "
                     f"faixa acima do critico: {pd.Timestamp(r.intervalo_inicio):%Y-%m} a "
                     f"{pd.Timestamp(r.intervalo_fim):%Y-%m}", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    axes[0].annotate("PPI, out/2016", xy=(DATA_PPI, axes[0].get_ylim()[1] * 0.96),
                     fontsize=7.5, rotation=90, ha="right", va="top", color="0.25")
    axes[-1].set_xlabel("data candidata da quebra")
    fig.suptitle("Onde os dados colocam a quebra comum de repasse\n"
                 "pontilhado = critico de 5% por bootstrap; bola = argmax; quadrado = out/2016",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "quebra_data_comum.png", dpi=150); plt.close(fig)
    print("  figura quebra_data_comum")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "estimar":
        estimar(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "figuras":
        figuras()
    else:
        escolher()
