# -*- coding: utf-8 -*-
"""
ONDA 4.5 -- COMPARACAO GRAFICA COM O MODELO PRINCIPAL

Tres figuras:

  comparacao_regimes_<var>.png   modelo principal (2 regimes) ao lado do de 3
                                 regimes, para mostrar que o "pos" do principal
                                 mistura PPI com pos-PPI e por isso subestima.

  comparacao_coeficiente_ppi.png o coeficiente da PPI contra o coeficiente
                                 agrupado do modelo principal, horizonte a
                                 horizonte, com a diferenca sombreada.

  comparacao_ols_iv.png          efeito do combustivel sobre precos que excluem
                                 combustivel: OLS (contabil) contra 2SLS
                                 instrumentado pela paridade (causal).

Uso: python onda4_5_comparacao_figuras.py
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"

C_PRE, C_PPI, C_POS = "#5F6549", "#9E3D25", "#0B5A56"
C_AGRUPADO = "#8C8C8C"
PRINCIPAIS = ["ipca_geral", "ipca_transportes", "gasolina_consumidor", "diesel"]
ROTULO = {"ipca_geral": "IPCA Geral", "ipca_transportes": "IPCA Transportes",
          "gasolina_consumidor": "Gasolina ao consumidor", "diesel": "Diesel"}


def carregar():
    t3 = pd.read_excel(SAIDA / "onda4_1_tres_regimes.xlsx", sheet_name="respostas")
    t2 = pd.read_excel(SAIDA / "onda2_8_teulings_zubanov.xlsx", sheet_name="respostas")
    t2 = t2[(t2.modo == "regime") & (t2.versao == "sem leads")].copy()
    t2["regime"] = t2.alvo.map({"choque_pre": "pré (modelo principal)",
                                "choque_pos": "pós 2017 (agrupado)"})
    return t2, t3


def fig_regimes(t2, t3):
    for nome in PRINCIPAIS:
        a = t2[t2.variavel == nome]
        b = t3[t3.variavel == nome]
        fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9), sharey=True)

        for alvo, cor, rot in [("choque_pre", C_PRE, "pré-2017"),
                               ("choque_pos", C_AGRUPADO, "pós-2017 (agrupado)")]:
            q = a[a.alvo == alvo].sort_values("h")
            ax[0].plot(q.h, q.coeficiente, color=cor, lw=2, marker="o", ms=3, label=rot)
            ax[0].fill_between(q.h, q.pontual_inferior, q.pontual_superior,
                               color=cor, alpha=0.14, lw=0)
        ax[0].set_title("Modelo principal: dois regimes", fontsize=9.5)

        for reg, cor, rot in [("pre-PPI", C_PRE, "pré-PPI"), ("PPI", C_PPI, "PPI"),
                              ("pos-PPI", C_POS, "pós-PPI")]:
            q = b[b.regime == reg].sort_values("h")
            ax[1].plot(q.h, q.coeficiente, color=cor, lw=2, marker="o", ms=3, label=rot)
            ax[1].fill_between(q.h, q.pontual_inferior, q.pontual_superior,
                               color=cor, alpha=0.14, lw=0)
        ax[1].set_title("Três regimes: a PPI isolada", fontsize=9.5)

        for k in [0, 1]:
            ax[k].axhline(0, color="0.35", lw=0.9)
            ax[k].set_xlabel("meses após o choque", fontsize=8.5)
            ax[k].set_xticks(range(0, 13, 2)); ax[k].tick_params(labelsize=7.5)
            ax[k].legend(fontsize=7.5)
        ax[0].set_ylabel("pontos percentuais", fontsize=8.5)
        fig.suptitle(f"{ROTULO[nome]}: o que o regime agrupado esconde\n"
                     f"faixas = banda pontual de 90%", fontsize=10.5)
        fig.tight_layout()
        arq = FIGS / f"comparacao_regimes_{nome}.png"
        fig.savefig(arq, dpi=150); plt.close(fig)
        print(f"  {arq.name}")


def fig_coeficiente(t2, t3):
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))
    for k, nome in enumerate(["ipca_geral", "ipca_transportes"]):
        ag = t2[(t2.variavel == nome) & (t2.alvo == "choque_pos")].sort_values("h")
        pp = t3[(t3.variavel == nome) & (t3.regime == "PPI")].sort_values("h")
        ax[k].fill_between(pp.h.values, ag.coeficiente.values, pp.coeficiente.values,
                           color=C_PPI, alpha=0.16, lw=0,
                           label="subestimação do modelo agrupado")
        ax[k].plot(ag.h, ag.coeficiente, color=C_AGRUPADO, lw=2, marker="s", ms=3,
                   label="pós-2017 agrupado")
        ax[k].plot(pp.h, pp.coeficiente, color=C_PPI, lw=2.2, marker="o", ms=3.4,
                   label="PPI isolada (2017 a 2023)")
        ax[k].axhline(0, color="0.35", lw=0.9)
        ax[k].set_title(ROTULO[nome], fontsize=9.5)
        ax[k].set_xlabel("meses após o choque", fontsize=8.5)
        ax[k].set_xticks(range(0, 13, 2)); ax[k].tick_params(labelsize=7.5)
        ax[k].legend(fontsize=7.5)
    ax[0].set_ylabel("pontos percentuais", fontsize=8.5)
    fig.suptitle("O efeito da PPI é maior do que o modelo agrupado sugere",
                 fontsize=10.5)
    fig.tight_layout()
    arq = FIGS / "comparacao_coeficiente_ppi.png"
    fig.savefig(arq, dpi=150); plt.close(fig)
    print(f"  {arq.name}")


def fig_ols_iv():
    t = pd.read_excel(SAIDA / "onda4_4_paridade_iv.xlsx", sheet_name="segundo_estagio")
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9), sharey=True)
    for k, desf in enumerate(["ipca_ex_comb_central", "ipca_naocomercializaveis"]):
        s = t[(t.desfecho == desf) & (~t.com_brent)].sort_values("h")
        ax[k].plot(s.h, s.c_ols, color=C_AGRUPADO, lw=2, marker="s", ms=3.4,
                   label="OLS (leitura contábil)")
        ax[k].fill_between(s.h, s.c_ols - L.Z_90 * s.ep_ols,
                           s.c_ols + L.Z_90 * s.ep_ols,
                           color=C_AGRUPADO, alpha=0.16, lw=0)
        ax[k].plot(s.h, s.c_2sls, color=C_PPI, lw=2.2, marker="o", ms=3.4,
                   label="2SLS, paridade do Golfo")
        ax[k].fill_between(s.h, s.c_2sls - L.Z_90 * s.ep_2sls,
                           s.c_2sls + L.Z_90 * s.ep_2sls,
                           color=C_PPI, alpha=0.16, lw=0)
        ax[k].axhline(0, color="0.35", lw=0.9)
        ax[k].set_title({"ipca_ex_comb_central": "IPCA ex-combustíveis",
                         "ipca_naocomercializaveis": "IPCA não-comercializáveis"}[desf],
                        fontsize=9.5)
        ax[k].set_xlabel("meses após o choque", fontsize=8.5)
        ax[k].tick_params(labelsize=7.5); ax[k].legend(fontsize=7.5)
    ax[0].set_ylabel("efeito de 1 p.p. na gasolina", fontsize=8.5)
    fig.suptitle("Efeito do combustível sobre preços que o excluem: "
                 "OLS contra variável instrumental\nfaixas = banda pontual de 90%",
                 fontsize=10.5)
    fig.tight_layout()
    arq = FIGS / "comparacao_ols_iv.png"
    fig.savefig(arq, dpi=150); plt.close(fig)
    print(f"  {arq.name}")


def main():
    print(__doc__)
    t2, t3 = carregar()
    fig_regimes(t2, t3)
    fig_coeficiente(t2, t3)
    fig_ols_iv()


if __name__ == "__main__":
    main()
