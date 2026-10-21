# -*- coding: utf-8 -*-
"""
FIGURAS DA VERSAO DE ARTIGO: out/2016 imposto, comparado com mar/2017 estimado

  fig1_resposta_acumulada.png   resposta acumulada por regime, quatro variaveis,
                                nas duas datas, com bandas pontuais de 90%
  fig2_canal.png                a decomposicao do canal, que passa a LIDERAR o
                                artigo: parcela do efeito que atravessa os
                                combustiveis, por regime e por horizonte
  fig3_voltas.png               primeira e segunda volta, e a inversao
  fig4_comparacao_datas.png     coeficiente de impacto nas duas datas, por
                                variavel, com intervalo de 90%

As duas datas usam EXATAMENTE o mesmo codigo de estimacao. A unica diferenca e
L.DATA_CORTE_FINAL. Ver artigo_set2016.py.

Uso: python figuras_artigo.py
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent.parent
TCC = PASTA / "resultados"
ART = PASTA / "resultados_artigo"
FIGS = PASTA / "figuras_artigo"; FIGS.mkdir(exist_ok=True)

ANTES, DEPOIS = "#5F6549", "#9E3D25"
AZUL, CINZA = "#1D3F6E", "#8C8C8C"
PRINCIPAIS = {"ipca_geral": "IPCA Geral", "ipca_transportes": "IPCA Transportes",
              "gasolina_consumidor": "Gasolina ao consumidor", "diesel": "Óleo diesel"}


def acumulada(pasta):
    d = pd.read_excel(pasta / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
    return d[(d.modo == "regime") & (d.versao == "sem leads")]


# ══════════════════════════════════════════════════ figura 1
def fig_acumulada():
    a, t = acumulada(ART), acumulada(TCC)
    fig, ax = plt.subplots(2, 4, figsize=(15.0, 7.0), sharex=True)
    for col, (v, rot) in enumerate(PRINCIPAIS.items()):
        for lin, (d, data) in enumerate([(a, "out/2016 imposto"), (t, "mar/2017 estimado")]):
            b = ax[lin, col]
            for alvo, cor, lab in [("choque_pre", ANTES, "regime anterior"),
                                   ("choque_pos", DEPOIS, "regime posterior")]:
                s = d[(d.variavel == v) & (d.alvo == alvo)].sort_values("h")
                if s.empty:
                    continue
                b.plot(s.h, s.coeficiente, color=cor, lw=1.8, marker="o", ms=3, label=lab)
                b.fill_between(s.h, s.pontual_inferior, s.pontual_superior,
                               color=cor, alpha=0.15, lw=0)
            b.axhline(0, color="0.35", lw=0.9)
            b.tick_params(labelsize=8)
            if lin == 0:
                b.set_title(rot, fontsize=10.5)
            if col == 0:
                b.set_ylabel("%s\np.p. acumulados" % data, fontsize=9)
            if lin == 1:
                b.set_xlabel("meses após o choque", fontsize=9)
            for sp in ["top", "right"]:
                b.spines[sp].set_visible(False)
    ax[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("Resposta acumulada do nível de preços a um choque de um desvio-padrão no Brent, "
                 "por regime\nlinha superior: data do anúncio da PPI imposta;  "
                 "linha inferior: data estimada pelos dados.  Bandas pontuais de 90%.",
                 fontsize=11, y=1.02, ha="center")
    fig.tight_layout()
    arq = FIGS / "fig1_resposta_acumulada.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ", arq.name)


# ══════════════════════════════════════════════════ figura 2
def fig_canal():
    d = pd.read_excel(ART / "onda2_4_canal_combustiveis.xlsx", sheet_name="por_regime")
    d = d[(d.desfecho == "ipca_geral") & (d.mediador == d.mediador.iloc[0])]
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4))

    a = ax[0]
    for alvo, cor, lab in [("choque_pre", ANTES, "regime anterior"),
                           ("choque_pos", DEPOIS, "regime posterior")]:
        s = d[d.choque == alvo].sort_values("h")
        a.plot(s.h, s.efeito_total, color=cor, lw=1.9, marker="o", ms=3.4, label="%s, total" % lab)
        a.plot(s.h, s.efeito_via_combustivel, color=cor, lw=1.4, ls="--",
               label="%s, via combustíveis" % lab)
    a.axhline(0, color="0.35", lw=0.9)
    a.set_xlabel("meses após o choque", fontsize=9)
    a.set_ylabel("pontos percentuais acumulados", fontsize=9)
    a.legend(fontsize=7.5, framealpha=0.9); a.tick_params(labelsize=8)
    a.set_title("Efeito total e parcela que atravessa os combustíveis", fontsize=10.5, loc="left")

    b = ax[1]
    for alvo, cor, lab in [("choque_pre", ANTES, "regime anterior"),
                           ("choque_pos", DEPOIS, "regime posterior")]:
        s = d[d.choque == alvo].sort_values("h")
        b.plot(s.h, s.efeito_via_combustivel, color=cor, lw=1.9, marker="o", ms=3.4, label=lab)
        b.fill_between(s.h, s.efeito_via_combustivel - 1.645 * s.ep_canal,
                       s.efeito_via_combustivel + 1.645 * s.ep_canal,
                       color=cor, alpha=0.15, lw=0)
    b.axhline(0, color="0.35", lw=0.9)
    b.set_xlabel("meses após o choque", fontsize=9)
    b.set_ylabel("pontos percentuais acumulados", fontsize=9)
    b.legend(fontsize=8, framealpha=0.9); b.tick_params(labelsize=8)
    b.set_title("Canal dos combustíveis em nível, com banda de 90%", fontsize=10.5, loc="left")
    for e in ax:
        for sp in ["top", "right"]:
            e.spines[sp].set_visible(False)
    fig.suptitle("Decomposição do repasse ao IPCA Geral segundo o canal dos combustíveis, "
                 "corte em out/2016", fontsize=11, y=1.03)
    fig.tight_layout()
    arq = FIGS / "fig2_canal.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ", arq.name)
    for h in [0, 1, 3, 6]:
        for alvo in ["choque_pre", "choque_pos"]:
            s = d[(d.choque == alvo) & (d.h == h)]
            if not s.empty:
                print("     h=%-2d %-11s total %6.3f | canal %6.3f | parcela %6.1f%%"
                      % (h, alvo, s.efeito_total.iloc[0], s.efeito_via_combustivel.iloc[0],
                         100 * s.parcela_via_combustivel.iloc[0]))


# ══════════════════════════════════════════════════ figura 3
def fig_comparacao():
    a, t = acumulada(ART), acumulada(TCC)
    vs = list(PRINCIPAIS)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    x = np.arange(len(vs)); larg = 0.19
    for k, (d, data, cor) in enumerate([(a, "out/2016 imposto", AZUL),
                                        (t, "mar/2017 estimado", CINZA)]):
        for j, alvo in enumerate(["choque_pre", "choque_pos"]):
            s = d[(d.alvo == alvo) & (d.h == 0)].set_index("variavel").reindex(vs)
            desl = (k * 2 + j - 1.5) * larg
            alpha = 0.45 if alvo == "choque_pre" else 1.0
            ax.bar(x + desl, s.coeficiente, width=larg, color=cor, alpha=alpha,
                   label="%s, %s" % (data, "anterior" if j == 0 else "posterior"))
            ax.errorbar(x + desl, s.coeficiente,
                        yerr=[s.coeficiente - s.pontual_inferior,
                              s.pontual_superior - s.coeficiente],
                        fmt="none", ecolor="0.3", elinewidth=0.9, capsize=2.5)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([PRINCIPAIS[v] for v in vs], fontsize=9)
    ax.set_ylabel("resposta no impacto, p.p.", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)
    ax.set_title("Resposta no mês do choque: data imposta contra data estimada",
                 fontsize=11, loc="left", pad=26)
    ax.text(0, 1.02, "barras claras: regime anterior.  barras cheias: regime posterior.  "
            "hastes: intervalo pontual de 90%.",
            transform=ax.transAxes, fontsize=8, color="0.35", va="bottom")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    arq = FIGS / "fig3_comparacao_datas.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ", arq.name)


def main():
    print("figuras da versao de artigo:")
    fig_acumulada(); fig_canal(); fig_comparacao()


if __name__ == "__main__":
    main()
