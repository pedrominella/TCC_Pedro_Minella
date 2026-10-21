# -*- coding: utf-8 -*-
"""
CONTRAFACTUAL POR ANO CIVIL, COM REINICIO EM JANEIRO (criado em 07/10/2026)

Mesma conta de fig_contrafactual.py (coeficientes NAO acumulados de onda2_6, regime
posterior = observado, regime anterior = contrafactual), mas feita separadamente para
2021 e para 2022, como o IPCA anual: a contribuicao acumulada volta a zero em janeiro.

Dentro de cada ano vale a mesma regra do TCC: so entram choques ocorridos no proprio
ano e so os efeitos que caem dentro dele (s - h >= janeiro e s <= dezembro). As bandas
de 90% saem da mesma funcao bandas() de fig_contrafactual.py, com a janela trocada.

Em 14/10/2026 entrou 2020 (pedido do autor), ano de queda do petroleo na pandemia, com a
mesma conta. A Figura 12 passa a ter tres paineis; o grafico do slide continua com 2021 e 2022.
Os coeficientes aplicados a 2020 foram estimados com os meses de mar-dez/2020 neutralizados
pelas indicadoras, ressalva registrada no texto do TCC.

Nao altera nenhum resultado do TCC. Saidas:
  resultados/contrafactual_anual.xlsx
  ../documentos_complementares/apresentacao_IC/graficos/slide_contrafactual_anual.png
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
import fig_contrafactual as FC

PASTA = Path(__file__).resolve().parent.parent
GRAF = PASTA.parent / "documentos_complementares" / "apresentacao_IC" / "graficos"
ANTES, DEPOIS, CINZA, FUNDO = "#5F6549", "#9E3D25", "#8C8C8C", "#f7f4ee"

b, dum = L.carregar("corrigida")
r = pd.read_excel(PASTA / "resultados" / "onda2_6_resposta_nao_acumulada.xlsx", sheet_name="por_regime")
r = r[r.variavel == "ipca_geral"]
coef = {a: {int(x.h): x.coeficiente for _, x in r[r.alvo == a].iterrows()} for a in ["choque_pre", "choque_pos"]}
ch = b.set_index("Data")["choque_petroleo"]
dl = b.set_index("Data")["dlog_ipca_geral"]

ANOS = (2020, 2021, 2022)
linhas, trajet = [], {}
for ano in ANOS:
    ini, fim = pd.Timestamp(f"{ano}-01-01"), pd.Timestamp(f"{ano}-12-01")
    meses = pd.date_range(ini, fim, freq="MS")
    contrib = {}
    for alvo in ["choque_pos", "choque_pre"]:
        s = pd.Series(0.0, index=meses)
        for t in meses:
            e = ch.get(t, np.nan)
            if pd.isna(e):
                continue
            for h, cf in coef[alvo].items():
                m = t + pd.DateOffset(months=h)
                if ini <= m <= fim:
                    s.loc[m] += cf * e
        contrib[alvo] = s.cumsum()
    ipca_log = float(dl.loc[meses].sum())
    ipca_pct = 100 * (np.exp(ipca_log / 100) - 1)
    FC.INI, FC.FIM = ini, fim
    bd = FC.bandas(b, ch, dum)
    trajet[ano] = (meses, contrib)
    # o ponto vem da trajetoria (coeficientes de onda2_6, como os 3,35 do TCC); a banda vem
    # de bandas(), reestimada na amostra comum aos 13 horizontes (como os 3,45 do TCC)
    pontos = {"choque_pos": contrib["choque_pos"].iloc[-1], "choque_pre": contrib["choque_pre"].iloc[-1]}
    pontos["diferenca"] = pontos["choque_pos"] - pontos["choque_pre"]
    for k, rot in [("choque_pos", "observado (regra de paridade)"),
                   ("choque_pre", "contrafactual (regra anterior)"),
                   ("diferenca", "diferenca atribuivel a regra")]:
        v = bd[k]
        linhas.append({"ano": ano, "grandeza": rot, "ponto_pp": pontos[k],
                       "ponto_amostra_comum": v["ponto"],
                       "ic5_hac": v["ic5_hac"], "ic95_hac": v["ic95_hac"], "p_hac": v["p_hac"],
                       "ipca_ano_pontos_log": ipca_log, "ipca_ano_pct": ipca_pct,
                       "pct_da_inflacao_do_ano": 100 * pontos[k] / ipca_log})

# conferencia: o mesmo laco, com a janela do bienio, tem de reproduzir os 2,92 e -0,43 do TCC
ini, fim = pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-01")
bien = {}
for alvo in ["choque_pos", "choque_pre"]:
    tot = 0.0
    for tt in pd.date_range(ini, fim, freq="MS"):
        for h, cf in coef[alvo].items():
            if tt + pd.DateOffset(months=h) <= fim:
                tot += cf * ch.get(tt, 0.0)
    bien[alvo] = tot
print("conferencia bienio: observado %.2f, contrafactual %.2f (TCC: 2,92 e -0,43)" % (bien["choque_pos"], bien["choque_pre"]))

t = pd.DataFrame(linhas)
with pd.ExcelWriter(PASTA / "resultados" / "contrafactual_anual.xlsx") as w:
    t.to_excel(w, sheet_name="por_ano", index=False)
    pd.concat([pd.DataFrame({"Data": m, "observado": c["choque_pos"].values, "contrafactual": c["choque_pre"].values})
               for m, c in trajet.values()]).to_excel(w, sheet_name="trajetoria", index=False)
pd.set_option("display.width", 220)
print(t.round(3).to_string(index=False))

# figura para o slide: um painel por ano, acumulado reiniciado em janeiro
fig, axs = plt.subplots(1, 2, figsize=(16.8, 6.0), dpi=200, sharey=True)
fig.patch.set_facecolor(FUNDO)
for ax, ano in zip(axs, (2021, 2022)):
    meses, c = trajet[ano]
    ax.set_facecolor(FUNDO)
    ax.plot(meses, c["choque_pos"], color=DEPOIS, lw=3, marker="o", ms=5, label="com a regra de paridade (observado)")
    ax.plot(meses, c["choque_pre"], color=ANTES, lw=3, marker="o", ms=5, label="com a regra anterior (contrafactual)")
    ax.fill_between(meses, c["choque_pre"], c["choque_pos"], color=DEPOIS, alpha=0.12, lw=0)
    ax.axhline(0, color="0.35", lw=0.9)
    d = t[(t.ano == ano) & (t.grandeza.str.startswith("diferenca"))].iloc[0]
    dif_txt = f"{d.ponto_pp:.2f}".replace(".", ",")
    ax.set_title(f"{ano}: diferença de {dif_txt} p.p. ({d.pct_da_inflacao_do_ano:.0f}% do IPCA do ano)",
                 fontsize=17, loc="left", color="#1f2328", fontweight="bold")
    ax.set_xticks(meses[::2])
    ax.set_xticklabels([m.strftime("%m/%y") for m in meses[::2]], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("contribuição acumulada no ano, p.p.", fontsize=14)
axs[0].legend(fontsize=13, loc="upper left", frameon=False)
fig.tight_layout()
saida = GRAF / "slide_contrafactual_anual.png"
fig.savefig(saida, facecolor=FUNDO)
plt.close(fig)
print("salvo", saida)

# figura do TCC (Figura 12), no estilo de fig_contrafactual.py, um painel por ano
def _n(x, casas=2):
    return (f"%.{casas}f" % x).replace(".", ",").replace("-", "−")


fig, axs = plt.subplots(1, len(ANOS), figsize=(12.4, 4.1), sharey=True)
for ax, ano, letra in zip(axs, ANOS, "ABC"):
    meses, c = trajet[ano]
    ax.plot(meses, c["choque_pos"], color=DEPOIS, lw=2.1, marker="o", ms=3.2, label="com a regra de paridade (observado)")
    ax.plot(meses, c["choque_pre"], color=ANTES, lw=2.1, marker="o", ms=3.2, label="com a regra anterior (contrafactual)")
    ax.fill_between(meses, c["choque_pre"], c["choque_pos"], color=DEPOIS, alpha=0.12, lw=0)
    ax.axhline(0, color="0.35", lw=0.9)
    d = t[(t.ano == ano) & (t.grandeza.str.startswith("diferenca"))].iloc[0]
    ax.set_title(f"{letra}. {ano}", fontsize=10.5, loc="left", pad=27)
    txt = (f"diferença de {_n(d.ponto_pp)} p.p. ao fim do ano, {_n(d.pct_da_inflacao_do_ano, 0)}% do IPCA\n"
           f"intervalo de 90%: {_n(d.ic5_hac)} a {_n(d.ic95_hac)} p.p.")
    ax.text(0, 1.015, txt, transform=ax.transAxes, fontsize=7.6, color=DEPOIS, va="bottom", linespacing=1.3)
    ax.set_xticks(meses[::2])
    ax.set_xticklabels([m.strftime("%m/%y") for m in meses[::2]], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("contribuição acumulada no ano, p.p.", fontsize=9)
axs[0].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
fig.tight_layout()
arq = PASTA / "figuras" / "contrafactual_anual.png"
fig.savefig(arq, dpi=150, bbox_inches="tight")
print("salvo", arq)
