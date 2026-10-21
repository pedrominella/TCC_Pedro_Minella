# -*- coding: utf-8 -*-
"""
EXERCICIO CONTRAFACTUAL: E SE A REGRA NAO TIVESSE MUDADO?

Toma os choques de petroleo efetivamente ocorridos em 2021 e 2022, periodo de
maior pressao inflacionaria da amostra, e acumula a contribuicao deles ao IPCA
sob duas parametrizacoes:

  observado      coeficientes do regime POSTERIOR, que e o que de fato vigorou
  contrafactual  coeficientes do regime ANTERIOR, isto e, o que teria ocorrido
                 se a regra de precos nao tivesse mudado

RESSALVA, e ela e obrigatoria. Isto e equilibrio parcial. A leitura correta e
"a CONTRIBUICAO dos choques de petroleo a inflacao acumulada teria sido X menor,
mantido todo o resto no caminho observado", e nao "a inflacao teria sido X menor".
Se o repasse fosse menor, a politica monetaria teria reagido de outro modo e as
expectativas seriam outras. Nao se trata de simulacao de equilibrio geral.

As respostas usadas sao as NAO acumuladas, de onda2_6, porque a contribuicao do
choque do mes t a inflacao do mes t+h e exatamente o coeficiente nao acumulado.

Uso: python fig_contrafactual.py
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
import onda2_bandas_placebo as O2          # ajustar(), replicas(), hac_conjunta_horizontes()
import onda2_6_resposta_nao_acumulada as O6  # y_mes(), pand_mes(), _base()

PASTA = Path(__file__).resolve().parent.parent
FIGS = PASTA / "figuras"
ANTES, DEPOIS, CINZA = "#5F6549", "#9E3D25", "#8C8C8C"
INI, FIM = pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-01")
B_BOOT, BLOCO, SEMENTE = 999, 12, 20260907


def bandas(b, ch, dum):
    """(F1.2) INCERTEZA DA CONTRIBUICAO ACUMULADA.

    A contribuicao do bienio e C_r = sum_h b_r(h) * a(h), com
        a(h) = sum_{t em [INI, FIM]} choque(t) * 1{t + h <= FIM},
    isto e, uma combinacao LINEAR dos coeficientes nao acumulados. Logo admite
    erro-padrao exato, desde que se use a covariancia CONJUNTA entre horizontes
    -- somar variancias isoladas ignoraria a correlacao entre b(0), b(1), ... e
    subestimaria a banda.

    Empilham-se os 13 horizontes dos DOIS alvos num vetor de 26 combinacoes e
    aplica-se hac_conjunta_horizontes, que so usa Xa e resid de cada ajuste e
    portanto aceita a mistura. Da mesma matriz sai a banda da diferenca, com
    peso -a para o regime anterior e +a para o posterior.

    O bootstrap wild em blocos entra como verificacao, com pesos COMUNS a todos
    os horizontes e alvos, de modo que a correlacao entre eles seja preservada.
    """
    alvos = ["choque_pre", "choque_pos"]
    b = b.copy()
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
    d, dinamica = O6._base("ipca_geral", b)
    montado = []
    for h in range(0, L.H_MAX + 1):
        d["__y"] = O6.y_mes(d, "ipca_geral", h)
        regs = alvos + ["regime_pos"] + dinamica + list(dum) + O6.pand_mes(d, h)
        am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
        montado.append((h, regs, am))
    # amostra comum: exigida pela covariancia conjunta entre horizontes
    comuns = set.intersection(*[set(am["Data"]) for _, _, am in montado])
    gs, ordem = [], []
    for a in alvos:
        for h, regs, am in montado:
            gs.append(O2.ajustar(am[am["Data"].isin(comuns)].reset_index(drop=True), regs, h, a))
            ordem.append((a, h))
    V = O2.hac_conjunta_horizontes(gs, L.H_MAX + 1)

    # pesos a(h): soma dos choques do bienio cujo efeito em t+h ainda cai no bienio
    meses = pd.date_range(INI, FIM, freq="MS")
    a_h = np.array([float(sum(ch.get(t, 0.0) for t in meses if t + pd.DateOffset(months=h) <= FIM))
                    for h in range(0, L.H_MAX + 1)])
    H = L.H_MAX + 1
    w = {"choque_pre": np.concatenate([a_h, np.zeros(H)]),
         "choque_pos": np.concatenate([np.zeros(H), a_h])}
    w["diferenca"] = w["choque_pos"] - w["choque_pre"]
    beta = np.array([g["valor"] for g in gs])

    # bootstrap com pesos comuns a todos os ajustes
    rng = np.random.default_rng(SEMENTE)
    datas_ref = pd.DatetimeIndex(sorted(set().union(*[set(g["datas"]) for g in gs])))
    nb = int(np.ceil(len(datas_ref) / BLOCO))
    etas = np.column_stack([np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:len(datas_ref)]
                            for _ in range(B_BOOT)])
    valb = np.row_stack([O2.replicas(g, etas, datas_ref)[0] for g in gs])   # 26 x B

    out = {}
    for k, wk in w.items():
        ponto = float(wk @ beta)
        ep = float(np.sqrt(max(wk @ V @ wk, 0.0)))
        bt = wk @ valb
        out[k] = {"ponto": ponto, "ep_hac": ep,
                  "ic5_hac": ponto - 1.645 * ep, "ic95_hac": ponto + 1.645 * ep,
                  "ic5_boot": float(np.nanpercentile(bt, 5)),
                  "ic95_boot": float(np.nanpercentile(bt, 95)),
                  "p_hac": float(2 * (1 - stats.norm.cdf(abs(ponto / ep)))) if ep > 0 else np.nan}
    out["_a_h"] = a_h
    out["_n_comum"] = len(comuns)
    return out


def main():
    b, dum = L.carregar("corrigida")
    b["pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    r = pd.read_excel(PASTA / "resultados" / "onda2_6_resposta_nao_acumulada.xlsx",
                      sheet_name="por_regime")
    r = r[r.variavel == "ipca_geral"]
    coef = {a: {int(x.h): x.coeficiente for _, x in r[r.alvo == a].iterrows()}
            for a in ["choque_pre", "choque_pos"]}

    meses = pd.date_range(INI, FIM, freq="MS")
    ch = b.set_index("Data")["choque_petroleo"]
    contrib = {}
    for alvo in ["choque_pos", "choque_pre"]:
        s = pd.Series(0.0, index=meses)
        for t in meses:
            e = ch.get(t, np.nan)
            if pd.isna(e):
                continue
            for h, cf in coef[alvo].items():
                alvo_mes = t + pd.DateOffset(months=h)
                if INI <= alvo_mes <= FIM:
                    s.loc[alvo_mes] += cf * e
        contrib[alvo] = s.cumsum()

    ipca = b.set_index("Data").loc[meses, "dlog_ipca_geral"].cumsum()

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.3),
                           gridspec_kw={"width_ratios": [1.55, 1]})
    a = ax[0]
    a.plot(meses, contrib["choque_pos"], color=DEPOIS, lw=2.1, marker="o", ms=3.2,
           label="com a regra de paridade (observado)")
    a.plot(meses, contrib["choque_pre"], color=ANTES, lw=2.1, marker="o", ms=3.2,
           label="com a regra anterior (contrafactual)")
    a.fill_between(meses, contrib["choque_pre"], contrib["choque_pos"],
                   color=DEPOIS, alpha=0.12, lw=0)
    a.axhline(0, color="0.35", lw=0.9)
    a.set_ylabel("contribuição acumulada, p.p.", fontsize=9)
    a.tick_params(labelsize=8)
    a.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    a.set_title("A. Contribuição dos choques de petróleo ao IPCA, 2021 e 2022",
                fontsize=10.5, loc="left", pad=16)
    a.text(0, 1.015, "a área sombreada é a diferença atribuível à mudança de regra",
           transform=a.transAxes, fontsize=8, color="0.35", va="bottom")
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)

    b_ = ax[1]
    fin_pos, fin_pre = contrib["choque_pos"].iloc[-1], contrib["choque_pre"].iloc[-1]
    dif, tot = fin_pos - fin_pre, ipca.iloc[-1]
    barras = [("IPCA acumulado\nno biênio", tot, CINZA),
              ("contribuição\nobservada", fin_pos, DEPOIS),
              ("contribuição\ncontrafactual", fin_pre, ANTES)]
    for i, (rot, v, cor) in enumerate(barras):
        b_.bar(i, v, width=0.6, color=cor)
        b_.text(i - 0.34, v + (0.35 if v >= 0 else -0.75), "%+.2f" % v, ha="right",
                fontsize=10, color="0.2")
    # (F1.2) hastes de 90% nas duas contribuicoes; o IPCA observado nao tem banda
    bd = bandas(b, ch, dum)
    for i, k in [(1, "choque_pos"), (2, "choque_pre")]:
        v = bd[k]
        b_.errorbar(i, v["ponto"], yerr=[[v["ponto"] - v["ic5_hac"]],
                                         [v["ic95_hac"] - v["ponto"]]],
                    fmt="none", ecolor="0.25", elinewidth=1.3, capsize=4, zorder=5)
    b_.axhline(0, color="0.35", lw=0.9)
    b_.set_xticks(range(3))
    b_.set_xticklabels([x[0] for x in barras], fontsize=8.5)
    b_.set_ylabel("pontos percentuais", fontsize=9)
    b_.tick_params(labelsize=8)
    b_.set_title("B. Magnitudes ao fim do período", fontsize=10.5, loc="left", pad=16)
    dv = bd["diferenca"]
    b_.text(0, 1.015, "diferença de %.2f p.p., ou %.0f%% da inflação do biênio; "
                      "intervalo de 90%%: %.2f a %.2f p.p."
            % (dif, 100 * dif / tot, dv["ic5_hac"], dv["ic95_hac"]),
            transform=b_.transAxes, fontsize=8, color=DEPOIS, va="bottom")
    for sp in ["top", "right"]:
        b_.spines[sp].set_visible(False)

    fig.tight_layout()
    arq = FIGS / "contrafactual.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ", arq.name)
    print("   contribuicao observada    %+.2f p.p." % fin_pos)
    print("   contribuicao contrafactual %+.2f p.p." % fin_pre)
    print("   diferenca                 %+.2f p.p.  (%.1f%% da inflacao do bienio, %.2f%%)"
          % (dif, 100 * dif / tot, tot))
    tab = pd.DataFrame({"Data": meses, "observado": contrib["choque_pos"].values,
                        "contrafactual": contrib["choque_pre"].values,
                        "ipca_acumulado": ipca.values})
    lin = []
    for k, rot in [("choque_pos", "contribuicao observada (regime posterior)"),
                   ("choque_pre", "contribuicao contrafactual (regime anterior)"),
                   ("diferenca", "diferenca atribuivel a regra de precos")]:
        v = bd[k]
        lin.append({"grandeza": rot, "ponto_pp": v["ponto"], "ep_hac": v["ep_hac"],
                    "ic5_hac": v["ic5_hac"], "ic95_hac": v["ic95_hac"],
                    "ic5_boot": v["ic5_boot"], "ic95_boot": v["ic95_boot"],
                    "p_hac": v["p_hac"],
                    "em_pct_da_inflacao": 100 * v["ponto"] / tot,
                    "ic5_em_pct": 100 * v["ic5_hac"] / tot,
                    "ic95_em_pct": 100 * v["ic95_hac"] / tot})
    inc = pd.DataFrame(lin)
    with pd.ExcelWriter(PASTA / "resultados" / "contrafactual_2021_2022.xlsx") as w:
        tab.to_excel(w, sheet_name="trajetoria", index=False)
        inc.to_excel(w, sheet_name="incerteza", index=False)
        pd.DataFrame({"h": range(L.H_MAX + 1), "peso_a_h": bd["_a_h"]}).to_excel(
            w, sheet_name="pesos", index=False)
    print("   planilha: resultados/contrafactual_2021_2022.xlsx")
    print("\n   === INCERTEZA (90%%, amostra comum de %d meses) ===" % bd["_n_comum"])
    for _, x in inc.iterrows():
        print("   %-46s %+6.2f  [%+6.2f, %+6.2f] HAC   [%+6.2f, %+6.2f] boot   p=%.3f"
              % (x.grandeza[:46], x.ponto_pp, x.ic5_hac, x.ic95_hac,
                 x.ic5_boot, x.ic95_boot, x.p_hac))


if __name__ == "__main__":
    main()
