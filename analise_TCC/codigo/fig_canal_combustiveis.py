# -*- coding: utf-8 -*-
"""
FIGURA 7 DO TCC (versão de 14/10/2026, a pedido do autor no comentário do Word)

O autor pediu, no gráfico da direita, só o intervalo do efeito total, sem os demais. Os painéis
trocam de lado em relação à versão anterior (canal_combustiveis_paineis.png):

  A  efeito via combustíveis antes e depois de março de 2017, com faixas de 90% por bootstrap
     (resultados/onda2_4_canal_combustiveis.xlsx, aba por_regime, mediador principal)
  B  decomposição da resposta acumulada do IPCA Geral no regime posterior: efeito total com a
     faixa de 90% da Tabela 2 (onda2_8_teulings_zubanov.xlsx, modo regime, sem leads), parte via
     combustíveis e parte não atribuída, essas duas sem faixa

Saída: figuras/canal_combustiveis_paineis_v2.png
Uso: python fig_canal_combustiveis.py
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from onda2_4_canal_combustiveis import PRINCIPAL

RES = AQUI.parent / "resultados"
FIGS = AQUI.parent / "figuras"
ESCURO, VERDE, TERRA, OLIVA = "#1f2328", "#0B5A56", "#9E3D25", "#5F6549"

t = pd.read_excel(RES / "onda2_4_canal_combustiveis.xlsx", sheet_name="por_regime")
t = t[(t.desfecho == "ipca_geral") & (t.mediador == PRINCIPAL)]
pos = t[t.choque == "choque_pos"].sort_values("h").set_index("h", drop=False)
pre = t[t.choque == "choque_pre"].sort_values("h").set_index("h", drop=False)

tab2 = pd.read_excel(RES / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
tab2 = tab2[(tab2.modo == "regime") & (tab2.versao == "sem leads") & (tab2.alvo == "choque_pos")
            & (tab2.variavel == "ipca_geral")].sort_values("h").set_index("h")

# conferências: Tabela 4 e o efeito total igual ao da Tabela 2 em todos os horizontes
v = lambda s, h, c: round(float(s.loc[h, c]), 3)
assert [v(pos, 0, "efeito_total"), v(pos, 0, "efeito_via_combustivel")] == [0.176, 0.082]
assert [v(pos, 3, "efeito_total"), v(pos, 3, "efeito_via_combustivel")] == [0.314, 0.249]
assert [v(pre, 3, "efeito_via_combustivel")] == [0.003]
assert (pos.efeito_total - tab2.coeficiente.reindex(pos.index)).abs().max() < 1e-8


def virgula(x, _=None):
    return ("%g" % round(x, 2)).replace(".", ",").replace("-", "−")


fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9))
for a in ax:
    for lado in ("top", "right"):
        a.spines[lado].set_visible(False)
    a.axhline(0, color="0.35", lw=0.8)
    a.set_xticks(range(0, 13, 2))
    a.tick_params(labelsize=7.5)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(virgula))
    a.set_xlabel("meses após o choque", fontsize=8)

a = ax[0]
for s, cor, rot in [(pre, OLIVA, "antes de mar/2017"), (pos, TERRA, "depois de mar/2017")]:
    a.fill_between(s.h, s.canal_boot_ic5, s.canal_boot_ic95, color=cor, alpha=0.15, lw=0)
    a.plot(s.h, s.efeito_via_combustivel, color=cor, lw=2, marker="o", ms=3, label=rot)
a.set_title("A. Efeito via combustíveis,\npor regime", fontsize=9, loc="left")
a.set_ylabel("resposta acumulada, p.p.", fontsize=8)
a.legend(fontsize=7.5, frameon=False, loc="upper right")

a = ax[1]
a.fill_between(tab2.index, tab2.pontual_inferior, tab2.pontual_superior, color=ESCURO, alpha=0.13, lw=0,
               label="faixa de 90% do efeito total")
a.plot(pos.h, pos.efeito_total, color=ESCURO, lw=2, marker="o", ms=3, label="efeito total")
a.plot(pos.h, pos.efeito_via_combustivel, color=VERDE, lw=2, marker="o", ms=3, label="via combustíveis")
a.plot(pos.h, pos.efeito_nao_via_combustivel, color=TERRA, lw=1.5, ls="--", label="não atribuído")
a.set_title("B. Decomposição da resposta do IPCA Geral\n(depois de mar/2017)", fontsize=9, loc="left")
a.set_ylim(top=0.72)
a.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)

fig.tight_layout()
fig.savefig(FIGS / "canal_combustiveis_paineis_v2.png", dpi=150)
plt.close(fig)
print("salva canal_combustiveis_paineis_v2.png")
