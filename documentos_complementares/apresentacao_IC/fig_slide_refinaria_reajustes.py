# -*- coding: utf-8 -*-
"""
Figura do slide da linha do tempo (apresentacao IC, 07/10/2026): variacao mensal da gasolina A
na refinaria, por regime. Mesma serie, mesmo corte e mesmo limiar de "preco parado" (|variacao| < 0,05%)
de analise_TCC/codigo/onda3_4_diagnostico_refinaria.py, que gera os numeros citados no TCC
(16,6% dos meses parados antes, 0 depois; mediana 0,172% antes, 4,265% depois).
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "analise_TCC" / "codigo"))
import lp_base_nova as L

LIMIAR_PARADO = 0.05
PRE, POS, FUNDO, TEXTO = "#5F6549", "#9E3D25", "#f7f4ee", "#1f2328"

b, _, _ = L.carregar_pandemia("corrigida")
b["pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
v = b["dlog_gasolina_refinaria"]
ok = v.notna()
b, v = b[ok], v[ok]

est = {}
for rot, m in [("pre", b.pos == 0), ("pos", b.pos == 1)]:
    x = v[m]
    est[rot] = (100 * float((x.abs() < LIMIAR_PARADO).mean()), float(x.abs().median()))
print({k: (round(a, 1), round(c, 3)) for k, (a, c) in est.items()})

fig, ax = plt.subplots(figsize=(16.0, 3.6), dpi=200)
fig.patch.set_facecolor(FUNDO)
ax.set_facecolor(FUNDO)
cores = [POS if p else PRE for p in b.pos]
ax.bar(pd.to_datetime(b.Data), v, width=25, color=cores)
ax.axhline(0, color="0.55", lw=0.8)
corte = pd.Timestamp(L.DATA_CORTE_FINAL)
ax.axvline(corte, color="0.25", lw=1.1, ls="--")
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.set_ylabel("% ao mês", fontsize=16, color=TEXTO)
ax.tick_params(labelsize=16, colors=TEXTO)
ax.set_xlim(pd.Timestamp("2002-11-01"), pd.Timestamp("2025-11-01"))
ax.set_ylim(-40, 62)
ax.set_yticks([-40, -20, 0, 20])
ax.text(0.015, 0.95, f"antes: preço parado em {est['pre'][0]:.1f}% dos meses\nvariação mediana de {est['pre'][1]:.2f}%".replace(".", ","),
        transform=ax.transAxes, va="top", fontsize=17, color=PRE, fontweight="bold")
ax.text(0.985, 0.95, f"depois: preço parado em {est['pos'][0]:.0f}% dos meses\nvariação mediana de {est['pos'][1]:.2f}%".replace(".", ","),
        transform=ax.transAxes, va="top", ha="right", fontsize=17, color=POS, fontweight="bold")
ax.set_title("Gasolina A na refinaria: variação mensal do preço, sem tributos federais", fontsize=18, fontweight="bold",
             color=TEXTO, loc="left")
fig.tight_layout()
saida = Path(__file__).resolve().parent / "graficos" / "slide03_refinaria_reajustes.png"
fig.savefig(saida, facecolor=FUNDO)
print("salvo", saida)
