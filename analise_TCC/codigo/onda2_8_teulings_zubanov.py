# -*- coding: utf-8 -*-
"""
ONDA 2, ITEM 2.8 -- CORRECAO DE TEULINGS E ZUBANOV: LEADS DO CHOQUE

O PROBLEMA
----------
Em local projections, ao regredir y(t+h) sobre choque(t), os choques ocorridos ENTRE
t+1 e t+h tambem afetam y(t+h). Se a serie de choque for autocorrelacionada, o
coeficiente beta(h) absorve parte desses choques futuros e deixa de medir apenas o
efeito do choque datado em t. Teulings e Zubanov (2014) mostram que isso e um vies, e
a correcao e incluir choque(t+1), ..., choque(t+h) como controles.

ISTO SE APLICA AQUI? SIM.
O choque do trabalho -- variacao logaritmica mensal do Brent padronizada -- tem
autocorrelacao de 0,287 na primeira defasagem, e o teste de Ljung-Box rejeita ausencia
de autocorrelacao com p abaixo de 0,001 em 1, 6 e 12 defasagens. Ou seja, a condicao
que gera o vies esta presente. Nao e um refinamento decorativo.

E vale registrar que Choi et al. (2018), o principal benchmark internacional do
trabalho, adota exatamente essa correcao. Alinhar-se a ela facilita a comparacao.

O QUE NAO MUDA, POR CONSTRUCAO
Em h=0 nao existe lead a incluir, entao as duas versoes sao IDENTICAS. O script
confere isso numericamente -- e a verificacao mais barata de que a implementacao esta
certa. Como os leads vao ate h e a dependente ja exige t+h, a amostra tambem nao muda:
o script confere isso tambem.

ESTE ARQUIVO NAO SUBSTITUI NADA
A especificacao de referencia continua sendo a sem leads, definida em
`lp_base_nova.MODELO_PRINCIPAL`. Aqui as duas rodam lado a lado e a comparacao e o
produto. A decisao de adotar ou nao e editorial, e o LEIA-ME registra a recomendacao.

Uso: python onda2_8_teulings_zubanov.py rodar | figuras
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
PASTA = AQUI.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
SAIDA.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)

import lp_base_nova as L
import onda2_bandas_placebo as O2

PRINCIPAIS = ["gasolina_consumidor", "diesel", "ipca_transportes", "ipca_geral"]
VERSOES = ["sem leads", "com leads"]


def _prep(b, nome):
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    for j in range(1, L.H_MAX + 1):
        d[f"choque_lead{j}"] = d["choque_petroleo"].shift(-j)
    din = []
    for lag in range(1, L.LAGS + 1):
        din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
               [f"{c}_lag{lag}" for c in L.CONTROLES]
    return d, din


def _bloco(d, nome, cab, din, alvo, dum, pand, com_leads):
    """Estima h=0..12 e devolve os ajustes e o critico uniforme."""
    montado, gs = [], {}
    for h in range(0, L.H_MAX + 1):
        d["__y"] = L.y_acum(d, nome, h)
        regs = list(cab) + din + dum + pand(d, h)
        if com_leads:
            regs = regs + [f"choque_lead{j}" for j in range(1, h + 1)]
        am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
        montado.append((h, regs, am))
        gs[h] = O2.ajustar(am, regs, h, alvo)
    comuns = set.intersection(*[set(am["Data"]) for _, _, am in montado])
    gsc = [O2.ajustar(am[am["Data"].isin(comuns)], regs, h, alvo)
           for h, regs, am in montado]
    c_unif = O2.sup_t_plugin(O2.hac_conjunta_horizontes(gsc, L.H_MAX + 1))
    return gs, c_unif


def rodar():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]

    linhas, wald = [], []
    checagens = {"h0_identico": 0.0, "amostra_igual": True}
    for nome in L.VARIAVEIS:
        d, din = _prep(b, nome)
        for modo, cab, alvos in [("cheia", ["choque_petroleo"], ["choque_petroleo"]),
                                 ("regime", ["choque_pre", "choque_pos", "regime_pos"],
                                  ["choque_pre", "choque_pos"])]:
            for alvo in alvos:
                res = {}
                for versao in VERSOES:
                    res[versao] = _bloco(d, nome, cab, din, alvo, dum, pand,
                                         com_leads=(versao == "com leads"))
                for h in range(0, L.H_MAX + 1):
                    g0, g1 = res["sem leads"][0][h], res["com leads"][0][h]
                    if h == 0:
                        checagens["h0_identico"] = max(
                            checagens["h0_identico"], abs(g0["valor"] - g1["valor"]))
                    if g0["nobs"] != g1["nobs"]:
                        checagens["amostra_igual"] = False
                    for versao, g, cu in [("sem leads", g0, res["sem leads"][1]),
                                          ("com leads", g1, res["com leads"][1])]:
                        cf, ep = g["valor"], g["ep"]
                        linhas.append({
                            "variavel": nome, "modo": modo, "alvo": alvo,
                            "versao": versao, "h": h,
                            "coeficiente": cf, "erro_padrao": ep,
                            "p_valor": float(2 * (1 - stats.norm.cdf(abs(cf / ep)))),
                            "pontual_inferior": cf - L.Z_90 * ep,
                            "pontual_superior": cf + L.Z_90 * ep,
                            "critico_uniforme": cu,
                            "uniforme_inferior": cf - cu * ep,
                            "uniforme_superior": cf + cu * ep,
                            "signif_pontual_10": bool(abs(cf / ep) > L.Z_90),
                            "signif_uniforme_10": bool(abs(cf / ep) > cu),
                            "n_leads": h if versao == "com leads" else 0,
                            "observacoes": g["nobs"]})
        # ---- Wald de igualdade entre regimes, nas duas versoes
        for h in range(0, L.H_MAX + 1):
            d["__y"] = L.y_acum(d, nome, h)
            base = ["choque_pre", "choque_pos", "regime_pos"] + din + dum + pand(d, h)
            for versao in VERSOES:
                regs = base + ([f"choque_lead{j}" for j in range(1, h + 1)]
                               if versao == "com leads" else [])
                am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                X = sm.add_constant(am[regs], has_constant="add")
                r = sm.OLS(am["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
                w = r.wald_test("choque_pre = choque_pos", scalar=True)
                wald.append({"variavel": nome, "versao": versao, "h": h,
                             "coef_pre": float(r.params["choque_pre"]),
                             "coef_pos": float(r.params["choque_pos"]),
                             "wald_p": float(w.pvalue), "observacoes": int(r.nobs)})
        print(f"  {nome}")

    t = pd.DataFrame(linhas); tw = pd.DataFrame(wald)
    piv = t.pivot_table(index=["variavel", "modo", "alvo", "h"], columns="versao",
                        values="coeficiente").reset_index()
    piv["dif"] = piv["com leads"] - piv["sem leads"]
    piv["dif_pct"] = 100 * piv["dif"] / piv["sem leads"].abs().clip(lower=0.02)
    with pd.ExcelWriter(SAIDA / "onda2_8_teulings_zubanov.xlsx") as w:
        t.to_excel(w, sheet_name="respostas", index=False)
        tw.to_excel(w, sheet_name="wald_regime", index=False)
        piv.to_excel(w, sheet_name="comparacao", index=False)

    print(f"\nVERIFICACOES")
    print(f"  h=0 identico nas duas versoes: desvio maximo {checagens['h0_identico']:.2e}"
          f"  ({'OK' if checagens['h0_identico'] < 1e-10 else 'FALHOU'})")
    print(f"  amostra igual nas duas versoes: "
          f"{'OK' if checagens['amostra_igual'] else 'FALHOU'}")

    print("\n=== AMOSTRA CHEIA: coeficiente sem e com leads ===")
    v = piv[(piv.modo == "cheia") & (piv.h.isin([0, 1, 2, 3, 6, 12]))]
    print(v[["variavel", "h", "sem leads", "com leads", "dif", "dif_pct"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n=== SIGNIFICANCIA: quantas das 78 respostas de amostra cheia ===")
    r = t[t.modo == "cheia"]
    for versao in VERSOES:
        s = r[r.versao == versao]
        print(f"  {versao:10s}  pontual {int(s.signif_pontual_10.sum()):2d}/78 | "
              f"uniforme {int(s.signif_uniforme_10.sum()):2d}/78 | "
              f"c_uniforme medio {s.groupby('variavel').critico_uniforme.first().mean():.3f}")

    print("\n=== REGIME: coeficiente posterior e Wald de igualdade ===")
    p2 = tw[tw.h.isin([0, 1, 3, 6])].pivot_table(
        index=["variavel", "h"], columns="versao", values=["coef_pos", "wald_p"])
    print(p2.round(3).to_string())

    print("\nhorizontes com Wald abaixo de 10%, por versao:")
    print(tw.groupby("versao").apply(lambda g: int((g.wald_p < 0.10).sum())).to_string())
    return t, tw, piv


def figuras():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.read_excel(SAIDA / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
    tw = pd.read_excel(SAIDA / "onda2_8_teulings_zubanov.xlsx", sheet_name="wald_regime")
    C = {"sem leads": "#0B5A56", "com leads": "#9E3D25"}

    # ---------- figura 1: amostra cheia, as duas versoes sobrepostas ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    for ax, nome in zip(axes.ravel(), PRINCIPAIS):
        for versao, cor in C.items():
            s = t[(t.modo == "cheia") & (t.variavel == nome)
                  & (t.versao == versao)].sort_values("h")
            ax.plot(s.h, s.coeficiente, color=cor, lw=2, label=versao)
            ax.fill_between(s.h, s.pontual_inferior, s.pontual_superior,
                            color=cor, alpha=0.13)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(nome, fontsize=9); ax.set_xlabel("meses"); ax.tick_params(labelsize=7.5)
    axes[0, 0].set_ylabel("%"); axes[1, 0].set_ylabel("%")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Correção de Teulings-Zubanov: resposta acumulada de amostra cheia\n"
                 "faixas = banda pontual de 90%", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "tz_amostra_cheia.png", dpi=150); plt.close(fig)
    print("  figura tz_amostra_cheia")

    # ---------- figura 2: regime, coeficiente posterior e Wald ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6))
    for ax, nome in zip(axes.ravel(), PRINCIPAIS):
        for versao, cor in C.items():
            s = t[(t.modo == "regime") & (t.alvo == "choque_pos")
                  & (t.variavel == nome) & (t.versao == versao)].sort_values("h")
            ax.plot(s.h, s.coeficiente, color=cor, lw=2, label=f"pós, {versao}")
            ax.fill_between(s.h, s.pontual_inferior, s.pontual_superior,
                            color=cor, alpha=0.13)
            p = t[(t.modo == "regime") & (t.alvo == "choque_pre")
                  & (t.variavel == nome) & (t.versao == versao)].sort_values("h")
            ax.plot(p.h, p.coeficiente, color=cor, lw=1.2, ls="--", alpha=0.75,
                    label=f"pré, {versao}")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(nome, fontsize=9); ax.set_xlabel("meses"); ax.tick_params(labelsize=7.5)
    axes[0, 0].set_ylabel("%"); axes[1, 0].set_ylabel("%")
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Correção de Teulings-Zubanov: resposta por regime (corte mar/2017)\n"
                 "linha cheia = regime posterior, tracejada = anterior", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "tz_por_regime.png", dpi=150); plt.close(fig)
    print("  figura tz_por_regime")

    # ---------- figura 3: o que muda, em uma folha ----------
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    piv = pd.read_excel(SAIDA / "onda2_8_teulings_zubanov.xlsx", sheet_name="comparacao")
    pc = piv[piv.modo == "cheia"]
    for nome in PRINCIPAIS:
        s = pc[pc.variavel == nome].sort_values("h")
        ax[0].plot(s.h, s.dif_pct, lw=1.7, marker="o", ms=3, label=nome)
    ax[0].axhline(0, color="0.3", lw=0.9)
    ax[0].set_title("Mudança percentual no coeficiente\n(com leads sobre sem leads)", fontsize=9)
    ax[0].set_xlabel("meses"); ax[0].set_ylabel("%"); ax[0].legend(fontsize=6.5)

    for versao, cor in C.items():
        s = tw[tw.versao == versao].groupby("h").apply(
            lambda g: (g.wald_p < 0.10).sum()).reset_index(name="n")
        ax[1].plot(s.h, s.n, color=cor, lw=2, marker="o", ms=4, label=versao)
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set_ylim(-0.3, 6.3)
    ax[1].set_title("Variáveis com Wald de regime abaixo de 10%\n(de 6, por horizonte)", fontsize=9)
    ax[1].set_xlabel("meses"); ax[1].legend(fontsize=7.5)
    fig.suptitle("O que a correção muda", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "tz_resumo.png", dpi=150); plt.close(fig)
    print("  figura tz_resumo")


if __name__ == "__main__":
    {"rodar": rodar, "figuras": figuras}[sys.argv[1] if len(sys.argv) > 1 else "rodar"]()
