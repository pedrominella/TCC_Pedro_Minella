# -*- coding: utf-8 -*-
"""
Figura do slide "O resultado à luz da literatura" (apresentação IC, 08/10/2026).
Mesmos números da Tabela 8 do TCC (seção 4.9), em pontos percentuais por alta de 10% no petróleo;
coeficientes do TCC multiplicados por 10/10,7. Fontes conferidas: ver
analise_TCC/codigo/aplicar_comparacao_literatura_docx.py.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANTES, DEPOIS, CINZA, LIT, FUNDO, TEXTO = "#5F6549", "#9E3D25", "#8C8C8C", "#2E4A62", "#f7f4ee", "#1f2328"

paineis = [
    ("No mês do choque", [
        ("Brasil antes (TCC)", 0.005, None, ANTES, "0,005"),
        ("Brasil 2000-15 (Choi)", -0.08, None, LIT, "−0,08"),
        ("Média de 71 países (Choi)", 0.10, None, LIT, "≈0,1"),
        ("Brasil depois (TCC)", 0.16, None, DEPOIS, "0,16"),
        ("EUA (Choi)", 0.25, None, LIT, "0,25"),
    ]),
    ("Em um ano ou no pico", [
        ("Brasil antes, 12 meses (TCC)", -0.06, None, ANTES, "−0,06"),
        ("Brasil, amostra toda, 12 meses", 0.10, None, CINZA, "0,10"),
        ("Brasil depois, 12 meses (TCC)", 0.25, None, DEPOIS, "0,25"),
        ("Brasil depois, pico (TCC)", 0.33, None, DEPOIS, "0,33"),
        ("72 países, 1 ano (Choi)", 0.40, None, LIT, "0,4"),
        ("EUA, pico (Känzig)", 0.40, None, LIT, "0,4"),
    ]),
    ("Canal dos combustíveis", [
        ("Brasil antes, 3 meses (TCC)", 0.003, None, ANTES, "0,003"),
        ("Brasil depois, 3 meses (TCC)", 0.23, None, DEPOIS, "0,23"),
        ("BCB, pesos de 2021", 0.20, 0.27, LIT, "0,20 a 0,27"),
        ("BCB, pesos de 2022", 0.31, 0.43, LIT, "0,31 a 0,43"),
    ]),
]

fig, axs = plt.subplots(1, 3, figsize=(16.8, 5.4), dpi=200)
fig.patch.set_facecolor(FUNDO)
for ax, (titulo, linhas) in zip(axs, paineis):
    ax.set_facecolor(FUNDO)
    n = len(linhas)
    for k, (rot, v, v2, cor, txt) in enumerate(linhas):
        y = n - 1 - k
        if v2 is None:
            ax.plot([0, v], [y, y], color=cor, lw=3, solid_capstyle="butt")
            ax.plot(v, y, "o", color=cor, ms=11)
            fim = v
        else:
            ax.plot([v, v2], [y, y], color=cor, lw=11, solid_capstyle="butt", alpha=0.85)
            fim = v2
        ax.text(max(fim, 0) + 0.035, y, txt, va="center", ha="left", fontsize=13, color=cor, fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([l[0] for l in linhas][::-1], fontsize=12.5, color=TEXTO)
    ax.axvline(0, color="0.45", lw=0.9)
    ax.set_xlim(-0.14, 0.92 if titulo.startswith("Canal") else 0.70)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_xticklabels(["0", "0,2", "0,4"], fontsize=11)
    ax.set_title(titulo, fontsize=16, fontweight="bold", color=TEXTO, loc="left")
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)
fig.text(0.5, 0.01, "pontos percentuais por alta de 10% no petróleo", ha="center", fontsize=12, color="0.35")
fig.tight_layout(rect=(0, 0.04, 0.985, 1), w_pad=2.5)
saida = Path(__file__).resolve().parent / "graficos" / "slide_literatura.png"
fig.savefig(saida, facecolor=FUNDO)
print("salvo", saida)
