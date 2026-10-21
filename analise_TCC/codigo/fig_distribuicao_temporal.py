# -*- coding: utf-8 -*-
"""
FIGURAS 5 E 6 DO TCC: A DISTRIBUIÇÃO TEMPORAL DO REPASSE, EM DUAS FIGURAS (14/10/2026)

Antes era uma figura só (onda2_6_resposta_nao_acumulada.figuras), com as barras mensais e,
num segundo eixo, uma linha tracejada com a resposta acumulada. O autor achou confusa e pediu
duas figuras. Na separação corrigiu-se também a linha: ela vinha da amostra inteira
(onda2_bandas_placebo_final.xlsx, aba bandas, tipo resposta, 0,057 no impacto do IPCA Geral),
e não do regime posterior que a nota da figura anunciava.

  distribuicao_temporal_mensal.png     Figura 5: inflação do mês t+h, regime posterior, IC de 90%
                                       (resultados/onda2_6_resposta_nao_acumulada.xlsx, por_regime)
  distribuicao_temporal_acumulada.png  Figura 6: resposta acumulada do nível de preços, regime
                                       posterior, faixa de 90%, a mesma da Tabela 2
                                       (resultados/onda2_8_teulings_zubanov.xlsx, respostas,
                                       modo regime, versão sem leads)

Uso: python fig_distribuicao_temporal.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lp_base_nova as L

RES = AQUI.parent / "resultados"
FIGS = AQUI.parent / "figuras"
SERIES = [("gasolina_consumidor", "Gasolina ao consumidor"), ("diesel", "Óleo diesel"),
          ("ipca_transportes", "IPCA Transportes"), ("ipca_geral", "IPCA Geral")]
VERDE, TERRA = "#0B5A56", "#9E3D25"

na = pd.read_excel(RES / "onda2_6_resposta_nao_acumulada.xlsx", sheet_name="por_regime")
na = na[na.alvo == "choque_pos"]
ac = pd.read_excel(RES / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
ac = ac[(ac.modo == "regime") & (ac.versao == "sem leads") & (ac.alvo == "choque_pos")]

# conferências: barras = as da figura antiga; linha = Tabela 2
g_na = na[na.variavel == "ipca_geral"].set_index("h").coeficiente
g_ac = ac[ac.variavel == "ipca_geral"].set_index("h").coeficiente
assert [round(g_na[0], 3), round(g_na[1], 3)] == [0.176, 0.159]
assert [round(g_ac[h], 3) for h in (0, 1, 3, 6, 12)] == [0.176, 0.356, 0.314, 0.189, 0.269]


def virgula(v, _=None):
    return ("%g" % round(v, 2)).replace(".", ",").replace("-", "−")


def eixos():
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0))
    for ax in axes.ravel():
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(labelsize=7.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(virgula))
        ax.set_xticks(range(0, 13, 2))
    return fig, axes


# Figura 5: inflação de cada mês, em linha com faixa de 90% (pedido do autor em 14/10/2026; antes barras)
fig, axes = eixos()
for ax, (nome, rot) in zip(axes.ravel(), SERIES):
    s = na[na.variavel == nome].sort_values("h")
    ax.fill_between(s.h, s.coeficiente - L.Z_90 * s.erro_padrao, s.coeficiente + L.Z_90 * s.erro_padrao,
                    color=VERDE, alpha=0.16, lw=0)
    ax.plot(s.h, s.coeficiente, color=VERDE, lw=2.0, marker="o", ms=3.5)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_title(rot, fontsize=9.5, loc="left")
    ax.set_xlabel("meses após o choque", fontsize=8.5)
axes[0, 0].set_ylabel("inflação do mês, p.p.", fontsize=8.5)
axes[1, 0].set_ylabel("inflação do mês, p.p.", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGS / "distribuicao_temporal_mensal.png", dpi=150)
plt.close(fig)

# Figura 6: resposta acumulada do nível de preços
fig, axes = eixos()
for ax, (nome, rot) in zip(axes.ravel(), SERIES):
    s = ac[ac.variavel == nome].sort_values("h")
    ax.fill_between(s.h, s.pontual_inferior, s.pontual_superior, color=TERRA, alpha=0.16, lw=0)
    ax.plot(s.h, s.coeficiente, color=TERRA, lw=2.0, marker="o", ms=3.5)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_title(rot, fontsize=9.5, loc="left")
    ax.set_xlabel("meses após o choque", fontsize=8.5)
axes[0, 0].set_ylabel("resposta acumulada, p.p.", fontsize=8.5)
axes[1, 0].set_ylabel("resposta acumulada, p.p.", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGS / "distribuicao_temporal_acumulada.png", dpi=150)
plt.close(fig)
print("salvas distribuicao_temporal_mensal.png e distribuicao_temporal_acumulada.png")
