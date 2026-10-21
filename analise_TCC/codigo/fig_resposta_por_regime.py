# -*- coding: utf-8 -*-
"""
FIGURAS DE COMPARACAO: funcao resposta e funcao resposta acumulada, por regime

Quatro figuras, uma por variavel principal. Cada uma com dois paineis:

  ESQUERDA -- FUNCAO RESPOSTA, nao acumulada.
      y(t+h) - y(t+h-1), ou seja a resposta da inflacao DO MES t+h.
      Responde "em que mes o efeito aparece".

  DIREITA -- FUNCAO RESPOSTA ACUMULADA.
      100*[ln P(t+h) - ln P(t-1)], a resposta do NIVEL de precos acumulada
      de t ate t+h. Responde "quanto o nivel esta mais alto depois de h meses".

Em cada painel, o regime anterior e o posterior a marco de 2017, com bandas
pontuais de 90%.

As duas sao a mesma informacao: a acumulada em h e a soma das nao acumuladas ate h.
Em h=0 as duas coincidem por definicao -- o script confere isso e imprime o desvio.

Fontes (nada e reestimado aqui, so lido):
  nao acumulada -> resultados/onda2_6_resposta_nao_acumulada.xlsx, aba por_regime
  acumulada     -> resultados/onda2_8_teulings_zubanov.xlsx, aba respostas,
                   modo=regime e versao='sem leads' (a especificacao de referencia)

Uso: python fig_resposta_por_regime.py
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"

PRINCIPAIS = {
    "ipca_geral": "IPCA Geral",
    "ipca_transportes": "IPCA Transportes",
    "gasolina_consumidor": "Gasolina ao consumidor",
    "diesel": "Diesel",
}
CORES = {"choque_pre": "#5F6549", "choque_pos": "#9E3D25"}
ROTULO = {"choque_pre": "antes de mar/2017", "choque_pos": "depois de mar/2017"}


def carregar():
    na = pd.read_excel(SAIDA / "onda2_6_resposta_nao_acumulada.xlsx", sheet_name="por_regime")
    ac = pd.read_excel(SAIDA / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
    ac = ac[(ac.modo == "regime") & (ac.versao == "sem leads")].copy()
    return na, ac


def painel(ax, dados, titulo, ylab=None):
    for alvo in ["choque_pre", "choque_pos"]:
        s = dados[dados.alvo == alvo].sort_values("h")
        ax.plot(s.h, s.coeficiente, color=CORES[alvo], lw=2, marker="o", ms=3.2,
                label=ROTULO[alvo])
        ax.fill_between(s.h, s.pontual_inferior, s.pontual_superior,
                        color=CORES[alvo], alpha=0.15, lw=0)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xlabel("meses após o choque", fontsize=8.5)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_xticks(range(0, 13, 2))


def main():
    na, ac = carregar()
    pior = 0.0
    for chave, rotulo in PRINCIPAIS.items():
        dn = na[na.variavel == chave]
        da = ac[ac.variavel == chave]
        # conferencia: em h=0 as duas definicoes coincidem
        for alvo in ["choque_pre", "choque_pos"]:
            v0 = float(dn[(dn.alvo == alvo) & (dn.h == 0)].coeficiente.iloc[0])
            v1 = float(da[(da.alvo == alvo) & (da.h == 0)].coeficiente.iloc[0])
            pior = max(pior, abs(v0 - v1))

        fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))
        painel(ax[0], dn, "Função resposta (não acumulada)\nresposta da inflação do mês t+h",
               ylab="pontos percentuais")
        painel(ax[1], da, "Função resposta acumulada\nresposta do nível de preços até t+h")
        ax[0].legend(fontsize=7.5, loc="best")
        fig.suptitle(f"{rotulo}: choque de um desvio-padrão no preço do petróleo\n"
                     f"faixas = banda pontual de 90%", fontsize=10.5)
        fig.tight_layout()
        arq = FIGS / f"regime_resposta_{chave}.png"
        fig.savefig(arq, dpi=150); plt.close(fig)
        print(f"  {arq.name}")

    print(f"\nconferencia h=0 (acumulada deve igualar a nao acumulada): "
          f"desvio maximo {pior:.2e}  ({'OK' if pior < 1e-9 else 'ATENCAO'})")


if __name__ == "__main__":
    main()
