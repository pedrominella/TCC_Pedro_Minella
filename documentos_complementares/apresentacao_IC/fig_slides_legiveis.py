# -*- coding: utf-8 -*-
"""
Gráficos da apresentação IC redesenhados para leitura à distância (08/10/2026).

Cada figura é criada com o tamanho, em polegadas, igual ao espaço que ocupa no slide dividido
por 100 (slide de 1920 x 1080). Assim, um texto de T pontos aparece com cerca de T/0,72 pixels
no slide; o menor texto usado tem 16 pt, ou cerca de 22 px.

Nada é reestimado: os números vêm das mesmas planilhas que geram as figuras do TCC.
  respostas por regime: analise_TCC/resultados/onda2_8_teulings_zubanov.xlsx, aba respostas,
                        modo = regime, versao = 'sem leads' (como fig_resposta_por_regime.py e
                        fig_combustiveis_regime.py)
  contrafactual anual:  analise_TCC/resultados/contrafactual_anual.xlsx
  literatura:           os mesmos valores da Tabela 8 do TCC
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
RES = RAIZ / "analise_TCC" / "resultados"
GRAF = AQUI / "graficos"
ANTES, DEPOIS, CINZA, LIT, FUNDO, TEXTO = "#5F6549", "#9E3D25", "#8C8C8C", "#2E4A62", "#f7f4ee", "#1f2328"
T_TICK, T_ROT, T_TIT, T_LEG = 16, 18, 22, 17
DPI = 200
plt.rcParams["font.family"] = "DejaVu Sans"


def figura(larg_px, alt_px, ncol=1, **kw):
    fig, ax = plt.subplots(1, ncol, figsize=(larg_px / 100, alt_px / 100), dpi=DPI, **kw)
    fig.patch.set_facecolor(FUNDO)
    for a in (ax if ncol > 1 else [ax]):
        a.set_facecolor(FUNDO)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)
        a.tick_params(labelsize=T_TICK, colors=TEXTO)
    return fig, ax


def legenda_regimes(fig, faixa=True):
    h = [Line2D([], [], color=ANTES, lw=3.5, marker="o", ms=8, label="antes de mar/2017"),
         Line2D([], [], color=DEPOIS, lw=3.5, marker="o", ms=8, label="depois de mar/2017")]
    if faixa:
        h.append(Patch(color=DEPOIS, alpha=0.18, label="faixa de 90%"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=T_LEG, frameon=False)


resp = pd.read_excel(RES / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
resp = resp[(resp.modo == "regime") & (resp.versao == "sem leads")]


def painel_regime(ax, variavel, titulo, ylab=None):
    s = resp[resp.variavel == variavel]
    for alvo, cor in (("choque_pre", ANTES), ("choque_pos", DEPOIS)):
        q = s[s.alvo == alvo].sort_values("h")
        ax.fill_between(q.h, q.pontual_inferior, q.pontual_superior, color=cor, alpha=0.16, lw=0)
        ax.plot(q.h, q.coeficiente, color=cor, lw=3.5, marker="o", ms=7)
    ax.axhline(0, color="0.35", lw=1.2)
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("meses após o choque", fontsize=T_ROT, color=TEXTO)
    ax.set_title(titulo, fontsize=T_TIT, fontweight="bold", color=TEXTO, loc="left")
    if ylab:
        ax.set_ylabel(ylab, fontsize=T_ROT, color=TEXTO)
    def fmt(v, _):
        return ("%g" % round(v, 2)).replace(".", ",").replace("-", "−")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))


def salvar(fig, nome):
    arq = GRAF / nome
    fig.savefig(arq, facecolor=FUNDO)
    plt.close(fig)
    print("salvo", arq.name)


# 1. IPCA cheio e IPCA Transportes (slide 10), espaço de 1600 x 500
fig, ax = figura(1600, 500, ncol=2)
painel_regime(ax[0], "ipca_geral", "IPCA cheio", "p.p. acumulados")
painel_regime(ax[1], "ipca_transportes", "IPCA Transportes")
legenda_regimes(fig)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=3)
salvar(fig, "slide_ipca_e_transportes_v2.png")

# 2. IPCA cheio sozinho (slide 9), espaço de 760 x 440
fig, ax = figura(760, 440)
painel_regime(ax, "ipca_geral", "IPCA cheio, resposta acumulada", "p.p.")
legenda_regimes(fig, faixa=False)
fig.tight_layout(rect=(0, 0.11, 1, 1))
salvar(fig, "slide_ipca_resposta_v2.png")

# 3. Gasolina na refinaria e ao consumidor (slide 11), espaço de 1600 x 450
fig, ax = figura(1600, 450, ncol=2)
painel_regime(ax[0], "gasolina_refinaria", "Gasolina A na refinaria", "p.p. acumulados")
painel_regime(ax[1], "gasolina_consumidor", "Gasolina ao consumidor")
legenda_regimes(fig)
fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=3)
salvar(fig, "slide_refinaria_e_bomba_v2.png")

# 4. Contrafactual por ano civil (slide 12), espaço de 1600 x 530. Desde 14/10/2026 com 2020,
#    como a Figura 12 do TCC; a diferença de 2020 vai escrita dentro do painel, com a ressalva.
ca = pd.read_excel(RES / "contrafactual_anual.xlsx", sheet_name="trajetoria")
ca["Data"] = pd.to_datetime(ca["Data"])
pa = pd.read_excel(RES / "contrafactual_anual.xlsx", sheet_name="por_ano")
dif20 = pa[(pa.ano == 2020) & pa.grandeza.str.startswith("diferenca")].ponto_pp.iloc[0]
assert ("%.2f" % dif20) == "-2.31", dif20
fig, ax = figura(1600, 530, ncol=3, sharey=True)
for a, ano in zip(ax, (2020, 2021, 2022)):
    q = ca[ca.Data.dt.year == ano]
    a.fill_between(q.Data, q.contrafactual, q.observado, color=DEPOIS, alpha=0.13, lw=0)
    a.plot(q.Data, q.observado, color=DEPOIS, lw=3.5, marker="o", ms=7)
    a.plot(q.Data, q.contrafactual, color=ANTES, lw=3.5, marker="o", ms=7)
    a.axhline(0, color="0.35", lw=1.2)
    a.set_title(str(ano), fontsize=T_TIT, fontweight="bold", color=TEXTO, loc="left")
    m = list(q.Data)[::3]
    a.set_xticks(m)
    a.set_xticklabels([x.strftime("%m/%y") for x in m], fontsize=T_TICK)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: ("%g" % round(v, 2)).replace(".", ",").replace("-", "−")))
ax[0].text(0.04, 0.97, "diferença de %s p.p.\nnúmero incerto: o modelo\nfoi estimado sem a pandemia"
           % ("%.2f" % dif20).replace(".", ",").replace("-", "−"),
           transform=ax[0].transAxes, va="top", ha="left", fontsize=T_TICK, color=DEPOIS, fontweight="bold",
           linespacing=1.25)
ax[0].set_ylabel("p.p. acumulados no ano", fontsize=T_ROT, color=TEXTO)
fig.legend(handles=[Line2D([], [], color=DEPOIS, lw=3.5, marker="o", ms=8, label="com a regra de paridade (observado)"),
                    Line2D([], [], color=ANTES, lw=3.5, marker="o", ms=8, label="com a regra anterior (contrafactual)")],
           loc="lower center", ncol=2, fontsize=T_LEG, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=2)
# posição de cada painel no slide (a imagem ocupa x de 160 a 1760), para alinhar os números de cima
for a, ano in zip(ax, (2020, 2021, 2022)):
    p_ = a.get_position()
    print("   painel %d no slide: x de %.0f a %.0f" % (ano, 160 + p_.x0 * 1600, 160 + p_.x1 * 1600))
salvar(fig, "slide_contrafactual_anual_v3.png")

# 5. Comparação com a literatura (slide 14), espaço de 1680 x 560
paineis = [
    ("No mês do choque", [
        ("Brasil antes", 0.005, None, ANTES, "0,005"),
        ("Brasil 2000-15 (Choi)", -0.08, None, LIT, "−0,08"),
        ("71 países (Choi)", 0.10, None, LIT, "≈0,1"),
        ("Brasil depois", 0.16, None, DEPOIS, "0,16"),
        ("EUA (Choi)", 0.25, None, LIT, "0,25"),
    ]),
    ("Em um ano ou no pico", [
        ("Brasil antes, 12 m", -0.06, None, ANTES, "−0,06"),
        ("Amostra toda, 12 m", 0.10, None, CINZA, "0,10"),
        ("Brasil depois, 12 m", 0.25, None, DEPOIS, "0,25"),
        ("Brasil depois, pico", 0.33, None, DEPOIS, "0,33"),
        ("72 países (Choi)", 0.40, None, LIT, "0,4"),
        ("EUA, pico (Känzig)", 0.40, None, LIT, "0,4"),
    ]),
    ("Canal dos combustíveis", [
        ("Brasil antes, 3 m", 0.003, None, ANTES, "0,003"),
        ("Brasil depois, 3 m", 0.23, None, DEPOIS, "0,23"),
        ("BCB, pesos 2021", 0.20, 0.27, LIT, "0,20 a 0,27"),
        ("BCB, pesos 2022", 0.31, 0.43, LIT, "0,31 a 0,43"),
    ]),
]
fig, ax = figura(1680, 560, ncol=3, gridspec_kw={"width_ratios": [1, 1, 1.12]})
for a, (titulo, linhas) in zip(ax, paineis):
    n = len(linhas)
    for k, (rot, v, v2, cor, txt) in enumerate(linhas):
        y = n - 1 - k
        if v2 is None:
            a.plot([0, v], [y, y], color=cor, lw=4, solid_capstyle="butt")
            a.plot(v, y, "o", color=cor, ms=13)
            fim = v
        else:
            a.plot([v, v2], [y, y], color=cor, lw=14, solid_capstyle="butt", alpha=0.85)
            fim = v2
        a.text(max(fim, 0) + 0.035, y, txt, va="center", ha="left", fontsize=T_TICK + 1, color=cor, fontweight="bold")
    a.set_yticks(range(n))
    a.set_yticklabels([l[0] for l in linhas][::-1], fontsize=T_TICK + 1)
    a.axvline(0, color="0.45", lw=1.1)
    a.set_xlim(-0.12, 0.92 if titulo.startswith("Canal") else 0.72)
    a.set_ylim(-0.6, n - 0.4)
    a.set_xticks([0, 0.2, 0.4])
    a.set_xticklabels(["0", "0,2", "0,4"])
    a.spines["left"].set_visible(False)
    a.tick_params(axis="y", length=0)
fig.text(0.5, 0.012, "pontos percentuais por alta de 10% no petróleo", ha="center", fontsize=T_TICK, color="0.3")
fig.tight_layout(rect=(0, 0.05, 1, 0.9), w_pad=1.5)
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
for a, (titulo, _) in zip(ax, paineis):
    x0 = min(t.get_window_extent(r).x0 for t in a.get_yticklabels())
    x0 = inv.transform((x0, 0))[0]
    topo = a.get_position().y1
    fig.text(x0, topo + 0.03, titulo, fontsize=T_TIT, fontweight="bold", color=TEXTO, ha="left", va="bottom")
salvar(fig, "slide_literatura_v2.png")
