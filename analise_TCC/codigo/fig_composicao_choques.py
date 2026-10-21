# -*- coding: utf-8 -*-
"""
COMPOSICAO DOS CHOQUES POR REGIME

Responde a uma explicacao alternativa para o contraste entre regimes: se o
repasse fosse assimetrico entre altas e quedas, e o regime posterior tivesse
concentrado choques de alta, a diferenca poderia refletir composicao da amostra
e nao mudanca de regra.

  Painel A  o choque mes a mes, com a cor marcando o regime. Serve para ver que
            nao ha aglomeracao de choques positivos depois do corte.
  Painel B  proporcao de meses de alta e de baixa em cada regime, e magnitude
            media de cada tipo.

Uso: python fig_composicao_choques.py
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L

FIGS = Path(__file__).resolve().parent.parent / "figuras"
ANTES, DEPOIS = "#5F6549", "#9E3D25"


def main():
    b, _ = L.carregar("corrigida")
    b["pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    d = b[["Data", "choque_petroleo", "pos"]].dropna()
    corte = pd.Timestamp(L.DATA_CORTE_FINAL)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.2),
                           gridspec_kw={"width_ratios": [1.85, 1]})

    # ── painel A
    a = ax[0]
    for rg, cor, rot in [(0, ANTES, "regime anterior"), (1, DEPOIS, "regime posterior")]:
        s = d[d.pos == rg]
        a.bar(s.Data, s.choque_petroleo, width=24, color=cor, label=rot)
    a.axhline(0, color="0.3", lw=0.9)
    a.axvline(corte, color="0.35", lw=1.0, ls=":")
    a.set_ylabel("choque, desvios-padrão", fontsize=9)
    a.tick_params(labelsize=8)
    a.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=2)
    a.set_title("A. O choque do petróleo, mês a mês", fontsize=10.5, loc="left", pad=16)
    a.text(0, 1.015, "a linha pontilhada marca março de 2017", transform=a.transAxes,
           fontsize=8, color="0.35", va="bottom")
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)

    # ── painel B
    b_ = ax[1]
    x = np.arange(2); larg = 0.34
    props, medias, ns = [], [], []
    for rg in (0, 1):
        s = d.loc[d.pos == rg, "choque_petroleo"]
        n = len(s); alta = (s > 0).sum()
        props.append((100 * alta / n, 100 * (s < 0).sum() / n))
        medias.append((s[s > 0].mean(), s[s < 0].mean()))
        ns.append(n)
    for j, (rot, cor, hat) in enumerate([("meses de alta", "0.55", ""),
                                         ("meses de baixa", "0.80", "//")]):
        vals = [props[0][j], props[1][j]]
        bars = b_.bar(x + (j - 0.5) * larg, vals, width=larg, color=cor,
                      hatch=hat, edgecolor="white", linewidth=0.7, label=rot)
        for xi, v in zip(x + (j - 0.5) * larg, vals):
            b_.text(xi, v + 1.5, "%.1f%%" % v, ha="center", fontsize=9.5, color="0.2")
    b_.set_xticks(x)
    b_.set_xticklabels(["regime anterior\n(n = %d)" % ns[0],
                        "regime posterior\n(n = %d)" % ns[1]], fontsize=9)
    b_.set_ylabel("% dos meses", fontsize=9)
    b_.set_ylim(0, 78)
    b_.tick_params(labelsize=8)
    b_.legend(fontsize=8, framealpha=0.9, loc="upper right")
    b_.set_title("B. Composição de altas e baixas", fontsize=10.5, loc="left", pad=16)
    b_.text(0, 1.015, "o regime posterior tem proporcionalmente MENOS choques de alta",
            transform=b_.transAxes, fontsize=8, color=DEPOIS, va="bottom")
    for sp in ["top", "right"]:
        b_.spines[sp].set_visible(False)

    fig.tight_layout()
    arq = FIGS / "composicao_choques.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ", arq.name)
    for rg, rot in [(0, "anterior"), (1, "posterior")]:
        print("   regime %-10s n=%3d | altas %.1f%% | média das altas %+.3f | "
              "média das baixas %+.3f"
              % (rot, ns[rg], props[rg][0], medias[rg][0], medias[rg][1]))


if __name__ == "__main__":
    main()
