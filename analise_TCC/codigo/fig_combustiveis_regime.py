# -*- coding: utf-8 -*-
"""
FIGURA: resposta dos quatro combustiveis ao choque do petroleo, por regime.

O primeiro elo da cadeia que o trabalho descreve. As figuras de regime existentes
cobrem IPCA Geral e Transportes, que sao o SEGUNDO elo; os precos de combustivel
apareciam so em tabela. Esta figura reune os quatro num painel 2x2, com a mesma
construcao e as mesmas cores das figuras de regime.

Fonte: resultados/onda2_8_teulings_zubanov.xlsx, aba respostas, modo=regime e
versao='sem leads', que e a especificacao de referencia. Nada e reestimado aqui.

Uso: python fig_combustiveis_regime.py
"""
import sys, warnings
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"

ORDEM = [("gasolina_refinaria", "Gasolina A na refinaria"),
         ("gasolina_consumidor", "Gasolina ao consumidor"),
         ("diesel", "Óleo diesel"),
         ("etanol", "Etanol")]
CORES = {"choque_pre": "#5F6549", "choque_pos": "#9E3D25"}
ROTULO = {"choque_pre": "antes de mar/2017", "choque_pos": "depois de mar/2017"}


def main():
    t = pd.read_excel(SAIDA / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
    t = t[(t.modo == "regime") & (t.versao == "sem leads")]

    fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey=True)
    for k, (chave, rot) in enumerate(ORDEM):
        a = ax[k // 2, k % 2]
        s = t[t.variavel == chave]
        for alvo in ["choque_pre", "choque_pos"]:
            q = s[s.alvo == alvo].sort_values("h")
            a.plot(q.h, q.coeficiente, color=CORES[alvo], lw=2, marker="o", ms=3,
                   label=ROTULO[alvo])
            a.fill_between(q.h, q.pontual_inferior, q.pontual_superior,
                           color=CORES[alvo], alpha=0.14, lw=0)
        a.axhline(0, color="0.35", lw=0.9)
        a.set_title(rot, fontsize=9.5)
        a.set_xticks(range(0, 13, 2)); a.tick_params(labelsize=7.5)
        if k // 2 == 1:
            a.set_xlabel("meses após o choque", fontsize=8.5)
        if k % 2 == 0:
            a.set_ylabel("pontos percentuais", fontsize=8.5)
    ax[0, 0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Resposta acumulada dos preços de combustíveis a um choque de um "
                 "desvio-padrão\nno preço do petróleo, por regime de política de preços "
                 "(faixas = banda pontual de 90%)", fontsize=10.5)
    fig.tight_layout()
    arq = FIGS / "regime_combustiveis_painel.png"
    fig.savefig(arq, dpi=150); plt.close(fig)
    print(f"  {arq.name}")


if __name__ == "__main__":
    main()
