# -*- coding: utf-8 -*-
"""
PROJEÇÕES LOCAIS COM VARIÁVEL INSTRUMENTAL (LP-IV), 14/10/2026

Continuação de iv_reestimacao.py, a pedido do autor: rodar os modelos com os instrumentos e mostrar os
resultados com gráficos. Nada aqui altera o texto do TCC.

Modelos
  (1) margem de refino do diesel -> preço do diesel acumulado de t-1 a t+h -> desfechos
      y(t,h) = β_r(h)·D(t,h) + controles, regime r antes ou depois de mar/2017
      β_pós(h): pontos logarítmicos do desfecho por ponto logarítmico de alta do diesel, quando a alta
      vem da margem de refino internacional (elasticidade acumulada)
      desfechos: IPCA Geral, IPCA Transportes, IPCA sem combustíveis (cenário central de pesos),
      não comercializáveis e comercializáveis
  (2) margem de refino da gasolina -> gasolina ao consumidor -> mesmos desfechos (instrumento fraco)
  (3) surpresa de oferta de Känzig -> choque do petróleo -> IPCA Geral, amostra completa e por regime,
      com o OLS da Tabela 2 para comparação

Especificação: a do modelo principal (defasagens do desfecho, do preço instrumentado, do choque, do
instrumento e dos controles X; sazonais; indicadoras de pandemia da janela de h; indicadora de regime;
choque contemporâneo com coeficiente por regime nos modelos de combustível), p = 5 no diesel e p = 2 nos
demais, como escolhido por AIC + 1 em iv_reestimacao.py.

Estimador: exatamente identificado, β = forma reduzida / primeiro estágio, com covariância HAC
(Newey-West, banda automática, nunca menor que h+1) conjunta das duas equações:
  intervalo convencional de 90% pelo método delta
  conjunto de Anderson-Rubin de 90%, robusto a instrumento fraco (onda2_3_forma_reduzida_ar.py)
  intervalo percentil de 90% por bootstrap selvagem em blocos de 12 meses, 999 réplicas, mesmos pesos
  nas duas equações

Saídas: resultados/iv_lp_modelos.xlsx; figuras/iv_F_por_horizonte.png, iv_lp_diesel.png,
        iv_lp_gasolina.png, iv_lp_kanzig.png
Uso: python iv_lp_modelos.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lp_base_nova as L
import iv_reestimacao as I
from onda2_3_forma_reduzida_ar import hac_conjunta, intervalo_ar
from onda2_7_segunda_volta import preparar

RES = AQUI.parent / "resultados"; FIGS = AQUI.parent / "figuras"
B, H = 999, 6
ESCURO, VERDE, TERRA, OLIVA, CINZA = "#1f2328", "#0B5A56", "#9E3D25", "#5F6549", "0.55"
NOMES = {"ipca_geral": "IPCA Geral", "ipca_transportes": "IPCA Transportes",
         "ipca_ex_comb_central": "IPCA sem combustíveis", "ipca_naocomercializaveis": "Não comercializáveis",
         "ipca_comercializaveis": "Comercializáveis"}
DESFECHOS = list(NOMES)


def bases():
    bk, bm, dum, pand = I.bases()
    bm, _ = preparar(bm)
    return bk, bm, dum, pand


def desenho(b, dum, pand, z, x, y, p, h, regime, ini=I.PMAX):
    d = b.copy()
    d["__x"] = d["choque_petroleo"] if x == "choque" else L.y_acum(d, x, h)
    d["__y"] = L.y_acum(d, y, h)
    lagvars = [f"dlog_{y}"] + ([] if x == "choque" else [f"dlog_{x}"]) + ["choque_petroleo", z] + list(L.CONTROLES)
    cols = []
    for v in dict.fromkeys(lagvars):
        for l in range(1, p + 1):
            d[f"{v}__L{l}"] = d[v].shift(l); cols.append(f"{v}__L{l}")
    contemp = [] if x == "choque" else ["choque_pre_c", "choque_pos_c"]
    exog = cols + contemp + dum + pand(d, h)
    if regime:
        d["z_pre"] = d[z] * (1 - d["regime_pos"]); d["z_pos"] = d[z] * d["regime_pos"]
        d["x_pre"] = d["__x"] * (1 - d["regime_pos"]); d["x_pos"] = d["__x"] * d["regime_pos"]
        inst, endo = ["z_pre", "z_pos"], ["x_pre", "x_pos"]; exog = exog + ["regime_pos"]
    else:
        d["z_all"] = d[z]; d["x_all"] = d["__x"]; inst, endo = ["z_all"], ["x_all"]
    am = d.iloc[ini:][["Data", "__x", "__y"] + inst + endo + exog].replace([np.inf, -np.inf], np.nan).dropna()
    return am, inst, endo, exog


def lpiv(am, inst, endo, exog, alvo, h, rng):
    X = np.column_stack([np.ones(len(am)), am[inst + exog].to_numpy(float)])
    P = np.linalg.pinv(X); T, K = X.shape; j = 1 + inst.index(alvo)
    y, x = am["__y"].to_numpy(float), am["__x"].to_numpy(float)
    by, bx = P @ y, P @ x
    uy, ux = y - X @ by, x - X @ bx
    bw = I.banda_nw(T, h)
    V, _ = hac_conjunta(X, uy, ux, bw)
    Vyy, Vxx, Vyx = V[j, j], V[K + j, K + j], V[j, K + j]
    gy, gx = by[j], bx[j]
    beta = gy / gx
    ep = np.sqrt(max((Vyy - 2 * beta * Vyx + beta ** 2 * Vxx) / gx ** 2, 0.0))
    lo, hi, forma = intervalo_ar(gy, gx, Vyy, Vyx, Vxx)
    aj_y, aj_x = X @ by, X @ bx
    bb = np.empty(B)
    for k in range(B):
        w = I.pesos_blocos(T, rng)
        bb[k] = (P[j] @ (aj_y + uy * w)) / (P[j] @ (aj_x + ux * w))
    # OLS do desfecho sobre o preço instrumentado, mesmos controles, para comparar
    Xo = np.column_stack([np.ones(len(am)), am[endo + exog].to_numpy(float)])
    bo = np.linalg.pinv(Xo) @ y
    return dict(h=h, F=gx ** 2 / Vxx, primeiro_estagio=gx, forma_reduzida=gy, p_forma_reduzida=float(
        2 * (1 - __import__("scipy").stats.norm.cdf(abs(gy) / np.sqrt(Vyy)))),
        beta=beta, ep=ep, ic_inf=beta - L.Z_90 * ep, ic_sup=beta + L.Z_90 * ep,
        ar_inf=lo, ar_sup=hi, ar_forma=forma,
        boot_inf=float(np.percentile(bb, 5)), boot_sup=float(np.percentile(bb, 95)),
        beta_ols=float(bo[1 + endo.index(alvo.replace("z_", "x_"))]), n=T, banda=bw)


def rodar():
    bk, bm, dum, pand = bases()
    rng = np.random.default_rng(I.SEMENTE)
    linhas = []
    for rot, z, x, p in [("margem_diesel", "margem_diesel", "diesel", 5),
                         ("margem_gasolina", "margem_gasolina", "gasolina_consumidor", 2)]:
        for y in DESFECHOS:
            for h in range(H + 1):
                am, inst, endo, exog = desenho(bm, dum, pand, z, x, y, p, h, True)
                for alvo in inst:
                    linhas.append({"instrumento": rot, "desfecho": y, "regime": alvo[2:],
                                   **lpiv(am, inst, endo, exog, alvo, h, rng)})
            print(f"  {rot:16s} {y}")
    for regime in (False, True):
        for h in range(H + 1):
            am, inst, endo, exog = desenho(bk, dum, pand, "instrumento_surpresa", "choque", "ipca_geral", 2, h, regime)
            for alvo in inst:
                linhas.append({"instrumento": "surpresa_kanzig", "desfecho": "ipca_geral", "regime": alvo[2:],
                               **lpiv(am, inst, endo, exog, alvo, h, rng)})
    print("  surpresa_kanzig  ipca_geral")
    t = pd.DataFrame(linhas)
    t.to_excel(RES / "iv_lp_modelos.xlsx", index=False)
    return t


# ------------------------------------------------------------------ gráficos
def virgula(v, _=None):
    return ("%g" % round(v, 3)).replace(".", ",").replace("-", "−")


def estilo(a, titulo):
    for lado in ("top", "right"):
        a.spines[lado].set_visible(False)
    a.axhline(0, color="0.35", lw=0.8)
    a.set_xticks(range(0, H + 1))
    a.tick_params(labelsize=7.5)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(virgula))
    a.set_title(titulo, fontsize=9, loc="left")
    a.set_xlabel("meses após o choque", fontsize=8)


def fig_F(t):
    casos = [("surpresa_kanzig", "Surpresa de oferta de Känzig\n(instrumenta o choque do petróleo)"),
             ("margem_gasolina", "Margem de refino da gasolina\n(instrumenta a gasolina ao consumidor)"),
             ("margem_diesel", "Margem de refino do diesel\n(instrumenta o diesel)")]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for a, (c, tit) in zip(ax, casos):
        s = t[(t.instrumento == c) & (t.desfecho == "ipca_geral")]
        for reg, cor, rot in [("pre", OLIVA, "antes de mar/2017"), ("pos", TERRA, "depois de mar/2017"),
                              ("all", ESCURO, "amostra completa")]:
            q = s[s.regime == reg].sort_values("h")
            if len(q):
                a.plot(q.h, q.F, color=cor, lw=2, marker="o", ms=3.5, label=rot)
        a.axhline(10, color="0.3", lw=1, ls="--")
        a.text(H, 10.6, "F = 10", fontsize=7, ha="right", color="0.3")
        estilo(a, tit)
    ax[0].set_ylabel("F robusto do primeiro estágio", fontsize=8)
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "iv_F_por_horizonte.png", dpi=150); plt.close(fig)


def fig_lp(t, inst, arq, titulo_geral):
    fig, ax = plt.subplots(1, 5, figsize=(13.5, 3.5))
    for a, y in zip(ax, DESFECHOS):
        q = t[(t.instrumento == inst) & (t.desfecho == y) & (t.regime == "pos")].sort_values("h")
        hs = q.h.to_numpy()
        for h_, f in zip(hs, q.F):
            if f < 10:
                a.axvspan(h_ - 0.45, h_ + 0.45, color="0.9", lw=0)
        ar_inf = np.where(np.isfinite(q.ar_inf), q.ar_inf, np.nan)
        ar_sup = np.where(np.isfinite(q.ar_sup), q.ar_sup, np.nan)
        lim = q.ar_forma == "limitado"
        a.vlines(hs[lim] + 0.12, ar_inf[lim], ar_sup[lim], color=TERRA, lw=1.4, label="Anderson-Rubin 90%")
        a.vlines(hs - 0.12, q.ic_inf, q.ic_sup, color=VERDE, lw=1.4, label="convencional 90%")
        a.plot(hs, q.beta, color=VERDE, marker="o", ms=4, lw=1.6, label="LP-IV")
        a.plot(hs, q.beta_ols, color=CINZA, ls="--", lw=1.2, label="MQO")
        for h_, fo in zip(hs, q.ar_forma):
            if fo != "limitado":
                a.text(h_ + 0.12, 0, "AR\nilim.", fontsize=6, color=TERRA, ha="center", va="bottom")
        estilo(a, NOMES[y])
        bot, topo = np.nanmin(q.ic_inf), np.nanmax(q.ic_sup)
        a.set_ylim(min(bot, 0) - 0.02, topo + 0.02)
    ax[0].set_ylabel("efeito por ponto log de alta do preço", fontsize=8)
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle(titulo_geral, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIGS / arq, dpi=150); plt.close(fig)


def fig_kanzig(t):
    ols = pd.read_excel(RES / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
    ols = ols[(ols.versao == "sem leads") & (ols.variavel == "ipca_geral")]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for a, reg, tit, alvo_ols in [(ax[0], "all", "Amostra completa", "choque_petroleo"),
                                  (ax[1], "pos", "Depois de mar/2017", "choque_pos")]:
        q = t[(t.instrumento == "surpresa_kanzig") & (t.regime == reg)].sort_values("h")
        o = ols[ols.alvo == alvo_ols].sort_values("h"); o = o[o.h <= H]
        hs = q.h.to_numpy()
        for h_, lo, hi, fo in zip(hs, q.ar_inf, q.ar_sup, q.ar_forma):
            if fo == "limitado":
                a.vlines(h_ + 0.12, lo, hi, color=TERRA, lw=1.4)
            else:
                a.text(h_ + 0.12, 0, "AR\nilim.", fontsize=6, color=TERRA, ha="center")
        a.vlines(hs - 0.12, q.ic_inf, q.ic_sup, color=VERDE, lw=1.4)
        a.plot(hs, q.beta, color=VERDE, marker="o", ms=4, lw=1.6, label="LP-IV, surpresa de Känzig")
        a.plot(o.h, o.coeficiente, color=ESCURO, lw=2, label="MQO (Tabela 2)")
        a.fill_between(o.h, o.pontual_inferior, o.pontual_superior, color=ESCURO, alpha=0.1, lw=0)
        estilo(a, f"{tit}: IPCA Geral\nF do primeiro estágio entre {virgula(q.F.min())} e {virgula(q.F.max())}")
    ax[0].set_ylabel("resposta acumulada a um choque de 1 desvio-padrão", fontsize=8)
    ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS / "iv_lp_kanzig.png", dpi=150); plt.close(fig)


def fig_lp_tcc(t, inst, arq):
    """Versão para o corpo do TCC: duas linhas, sem título geral (a legenda do Word faz esse papel)."""
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.6))
    for a, y in zip(axs.ravel(), DESFECHOS):
        q = t[(t.instrumento == inst) & (t.desfecho == y) & (t.regime == "pos")].sort_values("h")
        hs = q.h.to_numpy()
        for h_, f in zip(hs, q.F):
            if f < 10:
                a.axvspan(h_ - 0.45, h_ + 0.45, color="0.9", lw=0)
        lim = (q.ar_forma == "limitado").to_numpy()
        a.vlines(hs[lim] + 0.12, q.ar_inf[lim], q.ar_sup[lim], color=TERRA, lw=1.4, label="Anderson-Rubin, 90%")
        a.vlines(hs - 0.12, q.ic_inf, q.ic_sup, color=VERDE, lw=1.4, label="convencional, 90%")
        a.plot(hs, q.beta, color=VERDE, marker="o", ms=4, lw=1.6, label="LP-IV")
        a.plot(hs, q.beta_ols, color=CINZA, ls="--", lw=1.2, label="mínimos quadrados")
        estilo(a, NOMES[y])
    axs[0, 0].set_ylabel("pontos log por ponto log de alta", fontsize=8)
    axs[1, 0].set_ylabel("pontos log por ponto log de alta", fontsize=8)
    leg = axs[1, 2]; leg.axis("off")
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    leg.legend(h_, l_, fontsize=8, frameon=False, loc="center")
    fig.tight_layout(); fig.savefig(FIGS / arq, dpi=150); plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "figuras":
    t = pd.read_excel(RES / "iv_lp_modelos.xlsx")
    fig_lp_tcc(t, "margem_diesel", "iv_lp_diesel_tcc.png")
    print("ok")
elif __name__ == "__main__":
    t = rodar()
    fig_lp_tcc(t, "margem_diesel", "iv_lp_diesel_tcc.png")
    fig_F(t)
    fig_lp(t, "margem_diesel", "iv_lp_diesel.png",
           "LP-IV com a margem de refino do diesel, depois de mar/2017: efeito de 1 ponto log de alta do diesel "
           "(fundo cinza: F do primeiro estágio abaixo de 10)")
    fig_lp(t, "margem_gasolina", "iv_lp_gasolina.png",
           "LP-IV com a margem de refino da gasolina, depois de mar/2017: efeito de 1 ponto log de alta da "
           "gasolina (fundo cinza: F abaixo de 10)")
    fig_kanzig(t)
    print("ok")
