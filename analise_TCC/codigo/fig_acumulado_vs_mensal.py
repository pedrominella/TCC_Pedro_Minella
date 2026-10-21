# -*- coding: utf-8 -*-
"""
O QUE A VARIAVEL DEPENDENTE MEDE: ACUMULADO CONTRA MENSAL

Existe confusao recorrente entre dois objetos que o trabalho estima:

  ACUMULADO (Tabela 2)   y = 100*[ln P(t+h) - ln P(t-1)]
                         inflacao somada de t ate t+h. Em h=1 sao DOIS meses.

  MENSAL   (Tabela 3)    y = 100*[ln P(t+h) - ln P(t+h-1)]
                         inflacao SO do mes t+h. Em h=1 e so o mes seguinte.

Os dois tem o MESMO lado direito, com 43 regressores, todos datados em t ou
antes. O que muda e ate onde a inflacao acumula do lado esquerdo.

Painel A  esquema: quais meses cada dependente cobre
Painel B  as duas respostas estimadas, regime posterior, IPCA Geral
Painel C  a acumulada e a soma corrida das mensais

Uso: python fig_acumulado_vs_mensal.py
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

ART = AQUI.parent
RES, FIGS = ART / "resultados", ART / "figuras"
OLIVA, TERRA, ESCURO, CINZA = "#5F6549", "#9E3D25", "#272A24", "#8C8C8C"


def main():
    ac = pd.read_excel(RES / "onda3_3_regimes_data_escolhida.xlsx")
    ac = ac[ac.data_corte.astype(str).str.startswith("2017-03")]
    ac = ac[ac.variavel == "ipca_geral"].sort_values("h")
    na = pd.read_excel(RES / "onda2_6_resposta_nao_acumulada.xlsx", sheet_name="por_regime")
    na = na[(na.variavel == "ipca_geral") & (na.alvo == "choque_pos")].sort_values("h")
    h = ac.h.values
    A_, M_ = ac.coef_pos.values, na.coeficiente.values

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.42, wspace=0.22)

    # ─────────────────────────── A. esquema
    a = fig.add_subplot(gs[0, :])
    a.set_xlim(-1.3, 7.4); a.set_ylim(-0.6, 6.4); a.axis("off")
    meses = ["t−1", "t", "t+1", "t+2", "t+3"]
    for j, m in enumerate(meses):
        a.text(j, 5.9, m, ha="center", fontsize=10, color=ESCURO, fontweight="bold")
    a.text(-1.25, 5.9, "mês:", ha="left", fontsize=9.5, color="0.4")

    def barra(y, ini, fim, cor, rot):
        a.add_patch(Rectangle((ini - 0.42, y - 0.24), (fim - ini) + 0.84, 0.48,
                              facecolor=cor, alpha=0.88, lw=0))
        a.text(4.9, y, rot, va="center", fontsize=9.2, color=ESCURO)

    a.text(-1.25, 4.75, "ACUMULADO", fontsize=10, color=TERRA, fontweight="bold")
    a.text(-1.25, 4.42, "modelo principal", fontsize=8.4, color="0.4", style="italic")
    for k, (yy, hh) in enumerate([(4.2, 0), (3.5, 1), (2.8, 3)]):
        barra(yy, 1, 1 + hh, TERRA, "h=%d: %s (t até t+%d)"
              % (hh, "1 mês" if hh == 0 else "%d meses" % (hh + 1), hh))

    a.text(-1.25, 1.9, "MENSAL", fontsize=10, color=OLIVA, fontweight="bold")
    a.text(-1.25, 1.57, "decomposição", fontsize=8.4, color="0.4", style="italic")
    for k, (yy, hh) in enumerate([(1.35, 0), (0.65, 1), (-0.05, 3)]):
        barra(yy, 1 + hh, 1 + hh, OLIVA, "h=%d: só o mês t+%d" % (hh, hh))

    for j in range(5):
        a.plot([j, j], [-0.5, 5.6], color="0.88", lw=0.8, zorder=0)
    a.set_title("A. Os dois lados esquerdos: o direito é idêntico nos dois, com 43 regressores "
                "datados em t ou antes", fontsize=10.6, loc="left", color=ESCURO, pad=8)

    # ─────────────────────────── B. as duas curvas
    b = fig.add_subplot(gs[1, 0])
    b.plot(h, A_, "o-", color=TERRA, lw=2.2, ms=5, label="acumulado (Tabela 2)")
    b.plot(h, M_, "s--", color=OLIVA, lw=1.9, ms=4.6, label="mensal (Tabela 3)")
    b.axhline(0, color="0.4", lw=0.9)
    b.annotate("0,356: março + abril", xy=(1, A_[1]), xytext=(3.2, A_[1] - 0.015),
               fontsize=8.6, color=TERRA,
               arrowprops=dict(arrowstyle="-", color="0.55", lw=0.9))
    b.annotate("0,159: só abril", xy=(1, M_[1]), xytext=(3.2, M_[1] + 0.042),
               fontsize=8.6, color=OLIVA,
               arrowprops=dict(arrowstyle="-", color="0.55", lw=0.9))
    b.set_ylim(min(M_.min(), 0) - 0.04, max(A_) * 1.13)
    b.set_xticks(range(0, 13, 2))
    b.set_xlabel("horizonte h, em meses", fontsize=9.5)
    b.set_ylabel("resposta, p.p.", fontsize=9.5)
    b.legend(fontsize=8.8, framealpha=0.92)
    b.tick_params(labelsize=8.5)
    b.set_title("B. Em h=0 coincidem; a partir daí não", fontsize=10.6, loc="left",
                color=ESCURO, pad=8)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)

    # ─────────────────────────── C. soma corrida
    c = fig.add_subplot(gs[1, 1])
    c.bar(h, M_, color=OLIVA, alpha=0.55, width=0.62, label="contribuição do mês t+h")
    c.plot(h, np.cumsum(M_), "^-", color=CINZA, lw=1.8, ms=5,
           label="soma corrida das mensais")
    c.plot(h, A_, "o-", color=TERRA, lw=2.2, ms=5, label="acumulado estimado")
    c.axhline(0, color="0.4", lw=0.9)
    c.set_xticks(range(0, 13, 2))
    c.set_xlabel("horizonte h, em meses", fontsize=9.5)
    c.set_ylabel("resposta, p.p.", fontsize=9.5)
    c.legend(fontsize=8.4, framealpha=0.92, loc="upper left")
    c.tick_params(labelsize=8.5)
    c.set_title("C. O acumulado é, em essência, a soma das mensais", fontsize=10.6,
                loc="left", color=ESCURO, pad=8)
    for s in ("top", "right"):
        c.spines[s].set_visible(False)

    fig.suptitle("O que a variável dependente mede: IPCA Geral, regime posterior a março de 2017",
                 fontsize=12.4, x=0.007, ha="left", y=0.985, color=ESCURO)
    fig.text(0.007, 0.951,
             "as duas especificações compartilham o mesmo lado direito; muda apenas até onde "
             "a inflação acumula do lado esquerdo",
             fontsize=9, color="0.35", ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    arq = FIGS / "acumulado_vs_mensal.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ", arq.name)
    print("   h=0  acumulado %.3f | mensal %.3f  (coincidem por construcao)" % (A_[0], M_[0]))
    print("   h=1  acumulado %.3f | mensal %.3f  (2 meses contra 1)" % (A_[1], M_[1]))
    print("   soma das mensais ate h=1: %.3f" % np.cumsum(M_)[1])


if __name__ == "__main__":
    main()
