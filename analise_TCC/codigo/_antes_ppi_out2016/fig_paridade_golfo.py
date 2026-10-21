# -*- coding: utf-8 -*-
"""
FIGURA: preço doméstico contra a referência internacional do produto refinado.

Três painéis, todos DESCRITIVOS. A cotação da gasolina no Golfo dos EUA entra
como termo de comparação e NÃO é variável do modelo: o choque do trabalho é o
Brent, conforme a seção 3.2.

  1. Quatro séries em número-índice, jan/2003 = 100. A paridade "seca" converte
     dólares por galão em reais por litro pelo câmbio, sem frete, seguro,
     internação ou tributos. Serve para comparar DINÂMICA, não nível.
  2. Razão entre o índice da refinaria e o índice da paridade. Acima de 1
     significa que o preço doméstico cresceu mais que a paridade desde o início
     da amostra. Não é comparação de nível em reais por litro.
  3. Correlação móvel de 24 meses entre a variação mensal do preço na refinaria
     e a da paridade, e entre a refinaria e o Brent.

O terceiro painel é o mais informativo: a correlação oscila em torno de zero até
2016 e salta para a faixa de 0,45 a 0,75 depois, sem que nada tenha sido
estimado, sem escolha de data de corte e sem hipótese de identificação.

Uso: python fig_paridade_golfo.py
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
RAIZ = PASTA.parent
FIGS = PASTA / "figuras"
GOLFO = RAIZ / "base_de_dados" / "gasolina_golfo.csv"

LITROS_POR_GALAO = 3.785411784
JANELA = 24
MARCOS = {"2012-01-01": "Início da contenção\nde preços",
          "2016-09-01": "Anúncio da paridade\nde importação",
          "2023-05-01": "Fim da paridade\nde importação"}
AZUL, CINZA, FERRO, AMBAR = "#185FA5", "#8C8C8C", "#9E3D25", "#BA7517"


def base():
    b, _ = L.carregar("corrigida")
    g = pd.read_csv(GOLFO, skiprows=4)
    g.columns = ["mes_texto", "preco_dolar_galao"]
    g["Data"] = pd.to_datetime(g["mes_texto"].str.strip(), format="%b %Y", errors="coerce")
    g["usd_litro"] = pd.to_numeric(g["preco_dolar_galao"], errors="coerce") / LITROS_POR_GALAO
    g = g.dropna(subset=["Data", "usd_litro"])
    b = b.merge(g[["Data", "usd_litro"]], on="Data", how="left")
    falta = int(b["usd_litro"].isna().sum())
    if falta:
        raise SystemExit(f"{falta} meses sem cotação do Golfo")
    b["paridade_brl"] = b["usd_litro"] * b["Cambio"]
    b["brent_brl"] = b["Preco_Barril"] * b["Cambio"]
    return b


def indice(s):
    s = pd.to_numeric(s, errors="coerce")
    return 100.0 * s / s.dropna().iloc[0]


def marcos(ax, rotulos=False, y=0.97):
    for d, rot in MARCOS.items():
        ax.axvline(pd.Timestamp(d), color="0.35", lw=0.9, ls=":", zorder=1)
        if rotulos:
            ax.annotate(rot, (pd.Timestamp(d), y), xycoords=("data", "axes fraction"),
                        fontsize=7.5, va="top", ha="left", color="0.25",
                        xytext=(4, 0), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.5))


def main():
    b = base()
    fig, ax = plt.subplots(3, 1, figsize=(10.0, 11.0), sharex=True,
                           gridspec_kw={"height_ratios": [1.25, 1, 1], "hspace": 0.28})

    # ---------------- painel 1
    for col, cor, rot, lw in [
            ("paridade_brl", AZUL, "Gasolina no Golfo dos EUA, em reais", 1.5),
            ("brent_brl", CINZA, "Brent, em reais", 1.1),
            ("GasolinaABrasil_media_nivel", FERRO, "Gasolina A na refinaria", 2.0),
            ("Gasolina_nivel", AMBAR, "Gasolina ao consumidor", 1.4)]:
        ax[0].plot(b.Data, indice(b[col]), color=cor, lw=lw, label=rot)
    marcos(ax[0], rotulos=True)
    ax[0].set_ylabel("índice, janeiro de 2003 = 100", fontsize=9)
    ax[0].legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax[0].set_title("Preço doméstico da gasolina e referência internacional do produto\n"
                    "referência: gasolina convencional na Costa do Golfo dos EUA, FOB, "
                    "convertida em reais", fontsize=10.5, loc="left")
    ax[0].tick_params(labelsize=8)

    # ---------------- painel 2
    razao = indice(b["GasolinaABrasil_media_nivel"]) / indice(b["paridade_brl"])
    ax[1].plot(b.Data, razao, color=AZUL, lw=1.5)
    ax[1].axhline(1.0, color="0.3", lw=1.0, ls="--")
    marcos(ax[1])
    ax[1].set_ylabel("razão entre índices", fontsize=9)
    ax[1].set_title("Crescimento acumulado do preço na refinaria em relação à paridade\n"
                    "acima de 1: o preço doméstico subiu mais que a paridade desde 2003. "
                    "Não é comparação de nível em reais por litro.", fontsize=10, loc="left")
    ax[1].tick_params(labelsize=8)

    # ---------------- painel 3
    d_ref = 100 * np.log(pd.to_numeric(b["GasolinaABrasil_media_nivel"])).diff()
    d_par = 100 * np.log(b["paridade_brl"]).diff()
    d_bre = 100 * np.log(b["brent_brl"]).diff()
    ax[2].plot(b.Data, d_par.rolling(JANELA).corr(d_ref), color=FERRO, lw=1.6,
               label="com a gasolina do Golfo")
    ax[2].plot(b.Data, d_bre.rolling(JANELA).corr(d_ref), color=CINZA, lw=1.3,
               label="com o Brent")
    ax[2].axhline(0.0, color="0.3", lw=1.0, ls="--")
    marcos(ax[2])
    ax[2].set_ylabel("correlação", fontsize=9)
    ax[2].set_xlabel("", fontsize=9)
    ax[2].legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax[2].set_title("Correlação móvel de 24 meses com a variação mensal do preço na refinaria",
                    fontsize=10, loc="left")
    ax[2].tick_params(labelsize=8)

    fig.tight_layout()
    arq = FIGS / "paridade_golfo.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {arq.name}")

    m = pd.DataFrame({"Data": b.Data, "corr": d_par.rolling(JANELA).corr(d_ref)}).dropna()
    ant = m[m.Data < "2016-09-01"].corr_ if False else m[m.Data < "2016-09-01"]["corr"]
    pos = m[m.Data >= "2016-09-01"]["corr"]
    print(f"  correlação média antes de set/2016: {ant.mean():+.3f}  "
          f"(mínimo {ant.min():+.3f}, máximo {ant.max():+.3f})")
    print(f"  correlação média depois          : {pos.mean():+.3f}  "
          f"(mínimo {pos.min():+.3f}, máximo {pos.max():+.3f})")
    print(f"  razão no último mês: {razao.dropna().iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
