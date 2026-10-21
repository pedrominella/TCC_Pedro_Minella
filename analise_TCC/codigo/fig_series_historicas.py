# -*- coding: utf-8 -*-
"""
FIGURAS DE SERIE HISTORICA

Tres figuras:

  series_principais.png  compacta, 2x2, para a secao de dados da metodologia.
  series_nivel.png       todas as variaveis em NIVEL, grade 4x3, para o apendice.
  series_log.png         as variaveis que o modelo loga, em log-indice com
                         jan/2003 = 0, grade 3x3, para o apendice.

Sobre a escala logaritmica: mostra-se 100*[ln P(t) - ln P(2003-01)], que se le
em PONTOS LOGARITMICOS, nao em porcento: 126 pontos, o acumulado do IPCA Geral,
equivalem a alta real de 253%. Usa-se log porque distancias verticais iguais
representam taxas de crescimento iguais, e porque e a escala do modelo. Meta Selic e
expectativa NAO aparecem em log, porque ja sao taxas: o modelo as usa em nivel.

A linha vertical marca marco de 2017, a quebra estimada em onda3_3. A faixa
sombreada marca os meses de pandemia tratados por indicadoras (mar a dez/2020).

Uso: python fig_series_historicas.py
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

PASTA = Path(__file__).resolve().parent.parent
FIGS = PASTA / "figuras"
CORTE = pd.Timestamp(L.DATA_CORTE_FINAL)
PAND = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-01"))

AZUL, FERRUGEM, VERDE, TEAL, CINZA = "#3b74d4", "#9E3D25", "#5F6549", "#0B5A56", "#8C8C8C"


def moldura(ax, titulo, ylab=None):
    ax.axvline(CORTE, color="0.25", lw=1.0, ls="--", zorder=1)
    ax.axvspan(*PAND, color="0.85", alpha=0.55, lw=0, zorder=0)
    ax.set_title(titulo, fontsize=9)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.margins(x=0.01)


def logidx(b, col):
    """100*[ln P(t) - ln P(inicio)]: variacao acumulada em pontos logaritmicos."""
    v = 100 * np.log(pd.to_numeric(b[col], errors="coerce"))
    return v - v.dropna().iloc[0]


def carregar():
    b, _ = L.carregar("corrigida")
    return b


# ---------------------------------------------------------------- 1
def principais(b):
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.0), sharex=True)

    ax[0, 0].plot(b.Data, b.Preco_Barril, color=FERRUGEM, lw=1.4)
    moldura(ax[0, 0], "Petróleo Brent", "US$ por barril")

    for col, cor, rot in [("GasolinaABrasil_media_nivel", FERRUGEM, "gasolina na refinaria"),
                          ("Gasolina_nivel", AZUL, "gasolina ao consumidor"),
                          ("Oleo_diesel_nivel", VERDE, "diesel"),
                          ("Etanol_nivel", TEAL, "etanol")]:
        ax[0, 1].plot(b.Data, logidx(b, col), color=cor, lw=1.3, label=rot)
    moldura(ax[0, 1], "Combustíveis", "pontos log., jan/2003 = 0")
    ax[0, 1].legend(fontsize=6.8, loc="upper left", framealpha=0.9)

    for col, cor, rot in [("IPCA_Geral_nivel", AZUL, "IPCA Geral"),
                          ("IPCA_Trans_nivel", FERRUGEM, "IPCA Transportes")]:
        ax[1, 0].plot(b.Data, logidx(b, col), color=cor, lw=1.4, label=rot)
    moldura(ax[1, 0], "Índices de preços ao consumidor", "pontos log., jan/2003 = 0")
    ax[1, 0].legend(fontsize=7, loc="upper left", framealpha=0.9)

    ax[1, 1].plot(b.Data, logidx(b, "Cambio"), color=VERDE, lw=1.3, label="câmbio (pontos log.)")
    moldura(ax[1, 1], "Câmbio e política monetária", "pontos log., jan/2003 = 0")
    g = ax[1, 1].twinx()
    g.plot(b.Data, b.meta_selic_sgs432, color=FERRUGEM, lw=1.2, label="meta Selic")
    g.plot(b.Data, b.Expectativa_inflacao, color=CINZA, lw=1.2, ls="--",
           label="expectativa de inflação")
    g.set_ylabel("% ao ano", fontsize=8); g.tick_params(labelsize=7.5)
    h1, l1 = ax[1, 1].get_legend_handles_labels(); h2, l2 = g.get_legend_handles_labels()
    ax[1, 1].legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper left", framealpha=0.9)

    fig.suptitle("Séries utilizadas: janeiro de 2003 a setembro de 2025\n"
                 "escala em pontos logarítmicos, não em porcento (126 pontos = alta de 253%)",
                 fontsize=10.5)
    fig.tight_layout()
    arq = FIGS / "series_principais.png"
    fig.savefig(arq, dpi=150); plt.close(fig)
    print(f"  {arq.name}")


# ---------------------------------------------------------------- 2
PAINEIS_NIVEL = [
    ("Preco_Barril", "Petróleo Brent", "US$ por barril", FERRUGEM),
    ("Cambio", "Câmbio nominal", "R$ por US$", VERDE),
    ("Atividade", "Atividade econômica (IBC-Br)", "índice", TEAL),
    ("meta_selic_sgs432", "Meta Selic", "% ao ano", FERRUGEM),
    ("Expectativa_inflacao", "Expectativa de inflação", "% ao ano", CINZA),
    ("IPCA_Geral_nivel", "IPCA Geral", "índice", AZUL),
    ("IPCA_Trans_nivel", "IPCA Transportes", "índice", AZUL),
    ("GasolinaABrasil_media_nivel", "Gasolina A na refinaria", "índice", FERRUGEM),
    ("Gasolina_nivel", "Gasolina ao consumidor", "índice", FERRUGEM),
    ("Oleo_diesel_nivel", "Óleo diesel", "índice", VERDE),
    ("Etanol_nivel", "Etanol", "índice", TEAL),
]


def grade_nivel(b):
    fig, ax = plt.subplots(4, 3, figsize=(10.2, 10.4), sharex=True)
    axf = ax.ravel()
    for k, (col, tit, ylab, cor) in enumerate(PAINEIS_NIVEL):
        axf[k].plot(b.Data, pd.to_numeric(b[col], errors="coerce"), color=cor, lw=1.2)
        moldura(axf[k], tit, ylab)
    axf[-1].axis("off")
    fig.suptitle("Séries em nível, janeiro de 2003 a setembro de 2025\n"
                 "linha tracejada = março de 2017; faixa cinza = meses de pandemia",
                 fontsize=11)
    fig.tight_layout()
    arq = FIGS / "series_nivel.png"
    fig.savefig(arq, dpi=150); plt.close(fig)
    print(f"  {arq.name}")


# ---------------------------------------------------------------- 3
PAINEIS_LOG = [
    ("Preco_Barril", "Petróleo Brent", FERRUGEM),
    ("Cambio", "Câmbio nominal", VERDE),
    ("Atividade", "Atividade econômica (IBC-Br)", TEAL),
    ("IPCA_Geral_nivel", "IPCA Geral", AZUL),
    ("IPCA_Trans_nivel", "IPCA Transportes", AZUL),
    ("GasolinaABrasil_media_nivel", "Gasolina A na refinaria", FERRUGEM),
    ("Gasolina_nivel", "Gasolina ao consumidor", FERRUGEM),
    ("Oleo_diesel_nivel", "Óleo diesel", VERDE),
    ("Etanol_nivel", "Etanol", TEAL),
]


def grade_log(b):
    fig, ax = plt.subplots(3, 3, figsize=(10.2, 8.0), sharex=True)
    axf = ax.ravel()
    for k, (col, tit, cor) in enumerate(PAINEIS_LOG):
        axf[k].plot(b.Data, logidx(b, col), color=cor, lw=1.2)
        axf[k].axhline(0, color="0.6", lw=0.7)
        moldura(axf[k], tit, "pontos log., jan/2003 = 0" if k % 3 == 0 else None)
    fig.suptitle("Séries em escala logarítmica: 100·[ln P(t) − ln P(jan/2003)]\n"
                 "valores em PONTOS LOGARÍTMICOS, não em porcento: 126 pontos = alta real de 253%",
                 fontsize=11)
    fig.tight_layout()
    arq = FIGS / "series_log.png"
    fig.savefig(arq, dpi=150); plt.close(fig)
    print(f"  {arq.name}")


def main():
    b = carregar()
    print(f"amostra: {b.Data.min():%Y-%m} a {b.Data.max():%Y-%m}, n = {len(b)}")
    principais(b); grade_nivel(b); grade_log(b)


if __name__ == "__main__":
    main()
