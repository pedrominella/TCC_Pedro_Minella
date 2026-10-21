# -*- coding: utf-8 -*-
"""
Segunda parte dos gráficos da apresentação IC redesenhados para leitura à distância (08/10/2026):
preços (slide 02), correlação móvel (slide 05), perfil da quebra (slide 08) e repasse em 65 países
(slide 13). Os dados vêm das mesmas funções e planilhas das figuras do TCC:
  fig_paridade_golfo.base()              preços e correlação
  resultados/onda3_3_data_comum.xlsx     perfil da estatística de quebra
  fig_repasse_internacional.carrega()    repasse por país (base de Akcura, 2025), abr/2020 a jun/2022
Os números citados nos slides são conferidos com assert antes de desenhar.
Mesma regra de tamanho de fig_slides_legiveis.py: figura em polegadas = espaço no slide / 100.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
COD = RAIZ / "analise_TCC" / "codigo"
sys.path.insert(0, str(COD))
import fig_paridade_golfo as FP
import fig_repasse_internacional as FR

GRAF = AQUI / "graficos"
RES = RAIZ / "analise_TCC" / "resultados"
FUNDO, TEXTO, TERRA, OLIVA, AZUL, CINZA, AMBAR = "#f7f4ee", "#1f2328", "#9E3D25", "#5F6549", "#185FA5", "#8C8C8C", "#BA7517"
T_TICK, T_ROT, T_TIT, T_LEG = 16, 18, 22, 17
DPI = 200


def figura(larg_px, alt_px):
    fig, ax = plt.subplots(figsize=(larg_px / 100, alt_px / 100), dpi=DPI)
    fig.patch.set_facecolor(FUNDO)
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(labelsize=T_TICK, colors=TEXTO)
    return fig, ax


def virgula(v, _=None):
    return ("%g" % round(v, 2)).replace(".", ",").replace("-", "−")


def marcos(ax, rotulos):
    for d, rot in [("2012-01-01", "contenção"), ("2016-10-01", "paridade"), ("2023-05-01", "fim da\nparidade")]:
        ax.axvline(pd.Timestamp(d), color="0.4", lw=1.2, ls=":")
        if rotulos:
            ax.annotate(rot, (pd.Timestamp(d), 0.98), xycoords=("data", "axes fraction"), xytext=(5, 0),
                        textcoords="offset points", fontsize=T_TICK - 1, va="top", ha="left", color="0.25")


def salvar(fig, nome):
    fig.savefig(GRAF / nome, facecolor=FUNDO)
    plt.close(fig)
    print("salvo", nome)


b = FP.base()

# slide 02: preços em número-índice, espaço de 1000 x 560
fig, ax = figura(1000, 560)
for col, cor, rot, lw in [("paridade_brl", AZUL, "gasolina no Golfo dos EUA, em reais", 2.2),
                          ("brent_brl", CINZA, "Brent, em reais", 1.8),
                          ("GasolinaABrasil_media_nivel", TERRA, "gasolina A na refinaria", 3.0),
                          ("Gasolina_nivel", AMBAR, "gasolina ao consumidor", 2.4)]:
    ax.plot(b.Data, FP.indice(b[col]), color=cor, lw=lw, label=rot)
marcos(ax, rotulos=True)
ax.set_ylabel("índice, jan/2003 = 100", fontsize=T_ROT, color=TEXTO)
ax.legend(fontsize=T_LEG - 2, loc="upper left", bbox_to_anchor=(0, 0.86), frameon=False)
fig.tight_layout()
salvar(fig, "slide_precos_v2.png")

# slide 05: correlação móvel de 24 meses, espaço de 1680 x 520
d_ref = 100 * np.log(pd.to_numeric(b["GasolinaABrasil_media_nivel"])).diff()
d_par = 100 * np.log(b["paridade_brl"]).diff()
d_bre = 100 * np.log(b["brent_brl"]).diff()
c_par = d_par.rolling(FP.JANELA).corr(d_ref)
m = pd.DataFrame({"Data": b.Data, "c": c_par}).dropna()
antes, depois = m[m.Data < "2016-10-01"].c.mean(), m[m.Data >= "2016-10-01"].c.mean()
assert round(antes, 2) == 0.00 and round(depois, 2) == 0.58, (antes, depois)
fig, ax = figura(1680, 520)
ax.plot(b.Data, c_par, color=TERRA, lw=3.2, label="com a gasolina do Golfo")
ax.plot(b.Data, d_bre.rolling(FP.JANELA).corr(d_ref), color=CINZA, lw=2.4, label="com o Brent")
ax.axhline(0, color="0.3", lw=1.2, ls="--")
marcos(ax, rotulos=True)
ax.set_ylabel("correlação", fontsize=T_ROT, color=TEXTO)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(virgula))
ax.set_title("Correlação móvel de 24 meses com a variação mensal do preço na refinaria",
             fontsize=T_TIT - 2, fontweight="bold", color=TEXTO, loc="left")
ax.legend(fontsize=T_LEG, loc="lower right", frameon=False)
fig.tight_layout()
salvar(fig, "slide_correlacao_v2.png")

# slide 08: perfil da quebra, quatro séries principais, espaço de 1180 x 600
d = pd.read_excel(RES / "onda3_3_data_comum.xlsx", sheet_name="perfil")
d["data"] = pd.to_datetime(d["data"])
p = d[d.conjunto == "4 principais"].sort_values("data")
cr = float(p.critico_95.iloc[0])
arg = p.loc[p.soma_F.idxmax()]
assert round(arg.soma_F, 1) == 58.1 and round(cr, 1) == 42.9 and arg.data == pd.Timestamp("2017-03-01"), (arg.soma_F, cr, arg.data)
fig, ax = figura(1180, 600)
ax.axvspan(pd.Timestamp("2016-04-01"), pd.Timestamp("2021-02-01"), color="0.86", alpha=0.7, lw=0)
ax.plot(p.data, p.soma_F, color="#0B5A56", lw=3.2)
ax.axhline(cr, color="0.35", ls=":", lw=1.8)
ax.plot(arg.data, arg.soma_F, "o", ms=13, mfc="none", mec=TERRA, mew=3)
ax.annotate("máximo: mar/2017 (58,1)", (arg.data, arg.soma_F), xytext=(-300, -5), textcoords="offset points",
            fontsize=T_LEG, color=TERRA, fontweight="bold", va="center")
ax.text(p.data.iloc[2], cr + 1, "valor crítico de 5%: 42,9", fontsize=T_TICK, color="0.3", va="bottom")
ax.text(pd.Timestamp("2016-05-01"), ax.get_ylim()[0] + 1, "datas compatíveis\ncom os dados",
        fontsize=T_TICK - 1, color="0.3", va="bottom")
ax.set_ylabel("estatística de quebra (soma dos F)", fontsize=T_ROT - 1, color=TEXTO)
ax.set_xlabel("data candidata da quebra", fontsize=T_ROT, color=TEXTO)
fig.tight_layout()
salvar(fig, "slide_quebra_v2.png")

# slide 13: repasse em 65 países, espaço de 830 x 680
t = FR.carrega()
br = float(t.loc[t.cod == "BRA", "pt"].iloc[0])
med = t.groupby("regime").pt.median()
# janela abr/2020 a jun/2022, a de Akcura (2025), desde 08/10/2026
assert FR.A == pd.Period("2020-04", "M") and FR.B == pd.Period("2022-06", "M")
assert len(t) == 65 and round(br, 3) == 0.868, (len(t), br)
assert round(med["desregulado"], 3) == 0.798 and round(med["regulado"], 3) == 0.288, med
fig, ax = figura(830, 680)
rng = np.random.default_rng(7)
for j, (rot, cor) in enumerate([("desregulado", TERRA), ("regulado", OLIVA)]):
    s = t[t.regime == rot]
    ax.scatter(j + rng.uniform(-0.17, 0.17, len(s)), s.pt, s=90, color=cor, alpha=0.6, edgecolor="white", lw=0.8, zorder=2)
    ax.hlines(med[rot], j - 0.32, j + 0.32, color=TEXTO, lw=3, zorder=4)
    ax.text(j + 0.36, med[rot], "mediana\n" + ("%.3f" % med[rot]).replace(".", ","), fontsize=T_TICK, color=TEXTO, va="center")
ax.scatter([0], [br], s=260, color=TERRA, edgecolor=TEXTO, lw=2.2, zorder=6)
ax.annotate("Brasil " + ("%.3f" % br).replace(".", ","), xy=(0, br), xytext=(0.28, br + 0.3),
            fontsize=T_LEG + 1, color=TEXTO, fontweight="bold", arrowprops=dict(arrowstyle="-", color=TEXTO, lw=1.2))
ax.axhline(1, color="0.5", lw=1.1, ls=":")
ax.text(1.55, 1.02, "repasse integral", fontsize=T_TICK - 1, color="0.4", ha="right", va="bottom")
ax.set_xticks([0, 1])
ax.set_xticklabels([f"desregulado\n(n = {int((t.regime == 'desregulado').sum())})",
                    f"regulado\n(n = {int((t.regime == 'regulado').sum())})"], fontsize=T_ROT)
ax.set_xlim(-0.5, 1.6)
ax.set_ylabel("repasse ao preço de varejo", fontsize=T_ROT, color=TEXTO)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(virgula))
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0)
fig.tight_layout()
salvar(fig, "slide_paises_v2.png")
