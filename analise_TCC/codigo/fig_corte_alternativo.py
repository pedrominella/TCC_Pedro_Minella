# -*- coding: utf-8 -*-
"""
FIGURAS DA SECAO 4.7.6 E DO APENDICE F: data do anuncio contra data estimada

  corte_alternativo_impacto.png   compacta, para o corpo (secao 4.7.6). Resposta
      no mes do choque, quatro variaveis, nos dois cortes, com intervalo pontual
      de 90%. Serve para ver de relance que o padrao se reproduz.

  corte_alternativo_completo.png  detalhada, para o Apendice F. Resposta
      acumulada em doze horizontes, quatro variaveis, nos dois cortes.

Fonte dos dados, nada e reestimado aqui:
  mar/2017 (estimado, especificacao principal) -> resultados/
  out/2016 (anuncio da PPI, imposto)           -> resultados_artigo/
As duas rodam EXATAMENTE o mesmo codigo; a unica diferenca e a constante
L.DATA_CORTE_FINAL. Ver artigo_set2016.py.

Uso: python fig_corte_alternativo.py
"""
import warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent.parent
EST = PASTA / "resultados"            # mar/2017, estimado
IMP = PASTA / "resultados_artigo"     # out/2016, imposto
FIGS = PASTA / "figuras"

CORES = {"choque_pre": "#5F6549", "choque_pos": "#9E3D25"}
PRINCIPAIS = {"ipca_geral": "IPCA Geral", "ipca_transportes": "IPCA Transportes",
              "gasolina_consumidor": "Gasolina ao consumidor", "diesel": "Óleo diesel"}


def ler(pasta):
    d = pd.read_excel(pasta / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
    return d[(d.modo == "regime") & (d.versao == "sem leads")]


# ══════════════════════════════════════════════ corpo, secao 4.7.6
def compacta():
    imp, est = ler(IMP), ler(EST)
    vs = list(PRINCIPAIS)
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    x = np.arange(len(vs)); larg = 0.19
    for k, (d, data, hachura) in enumerate([(imp, "out/2016, imposto", ""),
                                            (est, "mar/2017, estimado", "///")]):
        for j, alvo in enumerate(["choque_pre", "choque_pos"]):
            s = d[(d.alvo == alvo) & (d.h == 0)].set_index("variavel").reindex(vs)
            desl = (k * 2 + j - 1.5) * larg
            ax.bar(x + desl, s.coeficiente, width=larg, color=CORES[alvo],
                   hatch=hachura, edgecolor="white", linewidth=0.6,
                   label="%s, regime %s" % (data, "anterior" if j == 0 else "posterior"))
            ax.errorbar(x + desl, s.coeficiente,
                        yerr=[s.coeficiente - s.pontual_inferior,
                              s.pontual_superior - s.coeficiente],
                        fmt="none", ecolor="0.25", elinewidth=0.9, capsize=2.5)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([PRINCIPAIS[v] for v in vs], fontsize=9)
    ax.set_ylabel("resposta no mês do choque, p.p.", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, ncol=2, framealpha=0.92, loc="upper left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    arq = FIGS / "corte_alternativo_impacto.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ", arq.name)


# ══════════════════════════════════════════════ Apendice F
def completa():
    imp, est = ler(IMP), ler(EST)
    fig, ax = plt.subplots(2, 4, figsize=(15.0, 7.0), sharex=True)
    for col, (v, rot) in enumerate(PRINCIPAIS.items()):
        for lin, (d, data) in enumerate([(imp, "out/2016, imposto"),
                                         (est, "mar/2017, estimado")]):
            b = ax[lin, col]
            for alvo, lab in [("choque_pre", "regime anterior"),
                              ("choque_pos", "regime posterior")]:
                s = d[(d.variavel == v) & (d.alvo == alvo)].sort_values("h")
                if s.empty:
                    continue
                b.plot(s.h, s.coeficiente, color=CORES[alvo], lw=1.8, marker="o",
                       ms=3, label=lab)
                b.fill_between(s.h, s.pontual_inferior, s.pontual_superior,
                               color=CORES[alvo], alpha=0.15, lw=0)
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
    ax[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    arq = FIGS / "corte_alternativo_completo.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ", arq.name)


def main():
    print("figuras do corte alternativo:")
    compacta(); completa()
    imp, est = ler(IMP), ler(EST)
    print("\n  conferencia, resposta no impacto:")
    for v in PRINCIPAIS:
        a = imp[(imp.variavel == v) & (imp.alvo == "choque_pos") & (imp.h == 0)].coeficiente.iloc[0]
        b = est[(est.variavel == v) & (est.alvo == "choque_pos") & (est.h == 0)].coeficiente.iloc[0]
        print("    %-22s out/2016 %6.3f | mar/2017 %6.3f" % (v, a, b))


if __name__ == "__main__":
    main()
