# -*- coding: utf-8 -*-
"""
FIGURA: o topo do perfil de quebra e um PLATO, nao um pico.

Motivo: com as taxas em ln(1+r), o argmax do conjunto de cinco series passou de
marco para fevereiro de 2017. A figura mostra que isso e irrelevante: a diferenca
e de 0,013 em uma estatistica de 89, e janeiro, fevereiro e marco de 2017 sao
indistinguiveis nos dois conjuntos. E a ilustracao visual da frase do capitulo de
que a data e identificada com folga de anos, e nao de meses.

Painel de cima: perfil completo. Painel de baixo: ampliacao de 2016 a 2018.

Uso: python fig_plato_quebra.py
"""
import sys, warnings, math
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
MES = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun",
       7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}
CONJ = [("4 principais", "Quatro principais", "#0B5A56"),
        ("5 com refinaria", "Cinco com refinaria", "#9E3D25")]


def main():
    d = pd.read_excel(SAIDA / "onda3_3_data_comum.xlsx", sheet_name="perfil")
    d["data"] = pd.to_datetime(d["data"])
    fig, ax = plt.subplots(2, 2, figsize=(10.0, 6.2))

    for k, (chave, rot, cor) in enumerate(CONJ):
        p = d[d.conjunto == chave].sort_values("data")
        cr = float(p.critico_95.iloc[0])
        arg = p.loc[p.soma_F.idxmax()]

        a = ax[0, k]
        a.plot(p.data, p.soma_F, color=cor, lw=1.4)
        a.axhline(cr, color="0.4", ls=":", lw=1.1)
        a.axvspan(pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-01"),
                  color="0.88", alpha=0.7, lw=0)
        a.set_title(f"{rot}: perfil completo", fontsize=9.5)
        a.set_ylabel("soma dos F", fontsize=8.5)
        a.tick_params(labelsize=7.5)

        z = p[(p.data >= "2016-01-01") & (p.data <= "2018-12-01")]
        b = ax[1, k]
        b.plot(z.data, z.soma_F, color=cor, lw=1.8, marker="o", ms=3)
        b.axhline(cr, color="0.4", ls=":", lw=1.1)
        topo = p.nlargest(3, "soma_F").sort_values("data")
        b.plot(topo.data, topo.soma_F, "o", ms=9, mfc="none", mec=cor, mew=1.8)
        txt = "\n".join("%s/%d   %s" % (MES[r.data.month], r.data.year,
                                        ("%.3f" % r.soma_F).replace(".", ","))
                        for _, r in topo.iterrows())
        b.text(0.03, 0.05, "três maiores\n" + txt, transform=b.transAxes,
               fontsize=7.5, color=cor, va="bottom", ha="left", family="monospace",
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=cor, alpha=0.9, lw=0.8))
        marg = arg.soma_F - p.nlargest(2, "soma_F").soma_F.iloc[1]
        b.set_title("ampliação: %s/%d vence por %s" % (
            MES[arg.data.month], arg.data.year, ("%.3f" % marg).replace(".", ",")),
            fontsize=9.5)
        b.set_ylabel("soma dos F", fontsize=8.5)
        b.set_xlabel("data candidata da quebra", fontsize=8.5)
        b.tick_params(labelsize=7.5)
        b.set_ylim(z.soma_F.min() - 2, z.soma_F.max() + 6)

    # diferenca relativa maxima entre jan, fev e mar/2017, nos dois conjuntos
    # (antes fixa no texto como "indistinguiveis"; com a serie da ANP, 01/10/2026,
    # o conjunto de cinco passou a diferir na primeira casa decimal)
    tri = d[d.data.isin(pd.to_datetime(["2017-01-01", "2017-02-01", "2017-03-01"]))
            & d.conjunto.isin([c for c, _, _ in CONJ])]
    dif = max((g.soma_F.max() - g.soma_F.min()) / g.soma_F.max() for _, g in tri.groupby("conjunto"))
    teto = math.ceil(1000 * dif) / 10                     # em %, arredondado para cima
    fig.suptitle("O topo do perfil de quebra é um platô, não um pico\n"
                 "janeiro, fevereiro e março de 2017 diferem em no máximo %s%% nos dois conjuntos; "
                 "linha pontilhada = valor crítico de 5%%" % ("%.1f" % teto).replace(".", ","),
                 fontsize=10.5)
    fig.tight_layout()
    arq = FIGS / "plato_quebra.png"
    fig.savefig(arq, dpi=150); plt.close(fig)
    print(f"  {arq.name}")


if __name__ == "__main__":
    main()
