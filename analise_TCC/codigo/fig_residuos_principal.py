# -*- coding: utf-8 -*-
"""
DIAGNOSTICO DE RESIDUOS DO MODELO PRINCIPAL

Especificacao por regime (lp_base_nova.MODELO_PRINCIPAL), variavel padrao
ipca_geral. Quatro paineis:

  A  serie residual do horizonte de impacto, separada pelos dois regimes
  B  autocorrelograma em h = 0, 3, 6 e 12. Em projecao local o residuo do
     horizonte h e MA(h) POR CONSTRUCAO, porque horizontes vizinhos compartilham
     meses da janela acumulada. A estrutura que aparece ali nao e defeito do
     modelo: e a razao de a inferencia usar HAC com banda h+1.
  C  histograma contra a normal, com Jarque-Bera

Nos dez meses de pandemia o residuo de h = 0 e exatamente zero: naquele horizonte
a indicadora de janela 1{t <= s <= t+h} colapsa em uma indicadora por mes, que
ajusta a observacao perfeitamente. Sao 10 observacoes que nao informam o
coeficiente de impacto, e o painel A mostra isso de forma literal.

Imprime ainda Breusch-Godfrey, Breusch-Pagan, Jarque-Bera e desvio residual por
regime, em cada horizonte reportado.

Uso: python fig_residuos_principal.py [variavel]
"""
import sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L

PASTA = Path(__file__).resolve().parent.parent
FIGS = PASTA / "figuras"; FIGS.mkdir(parents=True, exist_ok=True)

ANTES, DEPOIS = "#5F6549", "#9E3D25"
HS = [0, 3, 6, 12]
CORES_H = {0: "#1D3F6E", 3: "#2E7FB8", 6: "#BA7517", 12: "#9E3D25"}


def ajusta(b, dum, pand, nome, h):
    """Estima a equacao do modelo principal no horizonte h e devolve residuos."""
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    regs = ["choque_pre", "choque_pos", "regime_pos"]
    for lag in range(1, L.LAGS + 1):
        regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                [f"{c}_lag{lag}" for c in L.CONTROLES]
    regs += dum + pand(d, h)
    d["__y"] = L.y_acum(d, nome, h)
    am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(am[regs], has_constant="add")
    r = sm.OLS(am["__y"], X).fit()
    return am.assign(res=r.resid.values, aj=r.fittedvalues.values), r, X


def acf(u, k):
    u = np.asarray(u, float) - np.mean(u)
    d = np.dot(u, u)
    return [np.dot(u[l:], u[:-l]) / d for l in range(1, k + 1)]


def main(nome="ipca_geral"):
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]

    aj = {h: ajusta(b, dum, pand, nome, h) for h in HS}
    am0, r0, X0 = aj[0]

    fig = plt.figure(figsize=(11.0, 7.9))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.22)

    # ---------------- painel A
    a = fig.add_subplot(gs[0, :])
    for rot, sel, cor in [("antes de mar/2017", am0.regime_pos == 0, ANTES),
                          ("a partir de mar/2017", am0.regime_pos == 1, DEPOIS)]:
        s = am0[sel]
        a.plot(s.Data, s.res, color=cor, lw=1.0, marker="o", ms=2.0, label=rot)
    a.axhline(0, color="0.3", lw=0.9)
    a.axvline(pd.Timestamp(L.DATA_CORTE_FINAL), color="0.35", lw=0.9, ls=":")
    a.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-01"),
              color="0.85", zorder=0)
    s2 = am0.res.std()
    for k in (2, -2):
        a.axhline(k * s2, color="0.45", lw=0.8, ls="--")
    a.margins(y=0.16)
    a.set_ylabel("resíduo, pontos percentuais", fontsize=9)
    a.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    a.tick_params(labelsize=8)
    a.set_title("A. Resíduo do horizonte de impacto, por regime", fontsize=10, loc="left", pad=20)
    a.text(0, 1.015, "tracejado: dois desvios. Na faixa cinza o resíduo é zero por construção, "
           "pois em h = 0 cada mês de pandemia tem sua própria indicadora",
           transform=a.transAxes, fontsize=8, color="0.35", va="bottom")
    for i in am0.res.abs().nlargest(2).index:
        a.annotate(f"{am0.loc[i, 'Data']:%m/%Y}", (am0.loc[i, "Data"], am0.loc[i, "res"]),
                   fontsize=7.5, color="0.25",
                   xytext=(5, 3 if am0.loc[i, "res"] > 0 else -11), textcoords="offset points")
    for sp in ["top", "right"]: a.spines[sp].set_visible(False)

    # ---------------- painel B
    b_ = fig.add_subplot(gs[1, 0])
    K = 14
    larg = 0.20
    for i, h in enumerate(HS):
        u = aj[h][0].res.values
        desl = (i - 1.5) * larg
        b_.bar(np.arange(1, K + 1) + desl, acf(u, K), width=larg,
               color=CORES_H[h], label=f"h = {h}")
    lim = 1.96 / np.sqrt(len(am0))
    b_.axhline(lim, color="0.4", ls="--", lw=0.8)
    b_.axhline(-lim, color="0.4", ls="--", lw=0.8)
    b_.axhline(0, color="0.3", lw=0.9)
    b_.set_xlabel("defasagem, em meses", fontsize=9)
    b_.set_ylabel("autocorrelação", fontsize=9)
    b_.set_xticks(range(1, K + 1)); b_.tick_params(labelsize=7.5)
    b_.legend(fontsize=7.5, ncol=4, loc="upper right", framealpha=0.9)
    b_.set_title("B. Autocorrelação residual por horizonte", fontsize=10, loc="left", pad=18)
    b_.text(0, 1.015, "a dependência até a defasagem h vem da sobreposição das janelas",
            transform=b_.transAxes, fontsize=8, color="0.35", va="bottom")
    for sp in ["top", "right"]: b_.spines[sp].set_visible(False)

    # ---------------- painel C
    c = fig.add_subplot(gs[1, 1])
    u = am0.res.values
    c.hist(u, bins=28, density=True, color="#C9CFC0", edgecolor="0.45", lw=0.5)
    xs = np.linspace(u.min(), u.max(), 300)
    c.plot(xs, stats.norm.pdf(xs, u.mean(), u.std(ddof=1)), color=DEPOIS, lw=1.6,
           label="normal de referência")
    jb, pjb = stats.jarque_bera(u)[:2]
    c.set_xlabel("resíduo em h = 0, pontos percentuais", fontsize=9)
    c.set_ylabel("densidade", fontsize=9)
    c.tick_params(labelsize=8); c.legend(fontsize=7.5, framealpha=0.9, loc="upper right")
    c.set_title("C. Distribuição do resíduo de impacto", fontsize=10, loc="left", pad=18)
    c.text(0, 1.015, "a normalidade é rejeitada, mas a inferência HAC não a exige",
           transform=c.transAxes, fontsize=8, color="0.35", va="bottom")
    cap = (f"Jarque-Bera {jb:.1f} (p = {pjb:.3f})\n"
           f"assimetria {stats.skew(u):+.2f}\n"
           f"curtose {stats.kurtosis(u, fisher=False):.2f}")
    c.text(0.035, 0.95, cap, transform=c.transAxes, fontsize=7.5, color="0.25",
           va="top",
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.6))
    for sp in ["top", "right"]: c.spines[sp].set_visible(False)

    fig.tight_layout()
    arq = FIGS / f"residuos_principal_{nome}.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {arq.name}")

    # ---------------- diagnosticos
    print(f"\n  DIAGNOSTICOS  {nome}  (especificacao por regime, corte {L.DATA_CORTE_FINAL})")
    print(f"  {'h':>3} {'n':>4} {'K':>4} {'desvio':>7} {'antes':>7} {'depois':>7} "
          f"{'BG(h+1)':>9} {'BP':>7} {'JB':>7}")
    for h in HS:
        am, r, X = aj[h]
        u = am.res.values
        bg = acorr_breusch_godfrey(r, nlags=max(1, h + 1))[1]
        bp = het_breuschpagan(u, X.values)[1]
        jb = stats.jarque_bera(u)[1]
        print(f"  {h:>3} {len(u):>4} {X.shape[1]:>4} {u.std(ddof=1):>7.3f} "
              f"{am.loc[am.regime_pos == 0, 'res'].std(ddof=1):>7.3f} "
              f"{am.loc[am.regime_pos == 1, 'res'].std(ddof=1):>7.3f} "
              f"{bg:>9.3f} {bp:>7.3f} {jb:>7.3f}")
    print("  BG: Breusch-Godfrey com h+1 defasagens, p-valor. BP: Breusch-Pagan. "
          "JB: Jarque-Bera.")
    print("  Em h > 0 a rejeicao de BG e esperada por construcao da projecao local;")
    print("  o horizonte de impacto e o unico em que o teste e informativo.")

    m = am0.res.abs().nlargest(5)
    print("\n  cinco maiores residuos em h = 0:")
    for i in m.index:
        print(f"    {am0.loc[i, 'Data']:%Y-%m}  {am0.loc[i, 'res']:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ipca_geral")
