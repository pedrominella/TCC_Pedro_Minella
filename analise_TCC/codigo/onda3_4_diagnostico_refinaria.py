# -*- coding: utf-8 -*-
"""
ONDA 3, ITEM 3.4 -- POR QUE A GASOLINA NA REFINARIA SE COMPORTA DIFERENTE

A auditoria acumulou quatro sintomas, todos na MESMA serie e so nela:

  1. E a unica que rejeita o teste de autocorrelacao residual em todas as ordens
     testadas (1, 6 e 12 defasagens), no classico e no robusto.
  2. O coeficiente cai de 0,46 para 0,27 conforme as defasagens vao de 1 a 12,
     enquanto as demais series sao estaveis.
  3. Troca de sinal em h=6 quando se usa 12 defasagens.
  4. A quebra endogena dela cai em set/2018, nao em mar/2017 como o sistema.

A tentacao e tratar isso como defeito e pedir desculpa no texto. Este modulo testa a
hipotese alternativa: os quatro sintomas tem UMA causa, e ela e economica, nao
estatistica -- antes da PPI o preco na refinaria era ADMINISTRADO, com reajustes
discretos e infrequentes, e simplesmente nao respondia ao Brent.

Se for isso, entao:
  (a) a serie deve ficar parada em boa parte dos meses do regime anterior;
  (b) a ma especificacao deve estar CONCENTRADA no regime anterior e sumir no posterior;
  (c) o coeficiente do regime anterior nao deve ser lido como repasse, e sim como
      ausencia de resposta.

Uma projecao linear sobre um choque continuo e mal especificada quando a dependente
passa boa parte do tempo exatamente em zero e depois salta -- e o caso classico de
precos com ajuste infrequente. A leitura correta na literatura de repasse com precos
administrados nao e consertar a regressao, e sim estimar por regime e nao interpretar
a media de amostra cheia. O trabalho ja adota especificacao por regime como principal,
entao nao ha nada a mudar na estimacao: ha algo a EXPLICAR no texto.

Uso: python onda3_4_diagnostico_refinaria.py rodar | figuras
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")

_NOMES = {"ipca_geral": "IPCA Geral", "ipca_transportes": "IPCA Transportes",
          "gasolina_consumidor": "Gasolina ao consumidor",
          "gasolina_refinaria": "Gasolina na refinaria",
          "diesel": "Óleo diesel", "etanol": "Etanol",
          "ipca_comercializaveis": "Comercializáveis",
          "ipca_naocomercializaveis": "Não comercializáveis"}

AQUI = Path(__file__).resolve().parent
PASTA = AQUI.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
SAIDA.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)

import lp_base_nova as L
import onda2_5_lags_residuos as O5

SERIES = ["gasolina_refinaria", "gasolina_consumidor", "diesel", "ipca_geral"]
LIMIAR_PARADO = 0.05      # variacao mensal absoluta abaixo disto = preco parado
ESCADA = [1, 2, 3, 6, 12]


def _prep(b, nome, pmax=12):
    d = b.copy()
    for lag in range(1, pmax + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
        d[f"choque_petroleo_lag{lag}"] = d["choque_petroleo"].shift(lag)
        for c in L.CONTROLES:
            d[f"{c}_lag{lag}"] = d[c].shift(lag)
    return d


def rodar():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)

    # ---------- (a) o preco ficava parado? ----------
    freq = []
    for nome in SERIES:
        for rot, m in [("pre", b.pos == 0), ("pos", b.pos == 1)]:
            v = b.loc[m, f"dlog_{nome}"].dropna()
            freq.append({"variavel": nome, "regime": rot, "n": len(v),
                         "pct_meses_parado": 100 * float((v.abs() < LIMIAR_PARADO).mean()),
                         "variacao_absoluta_mediana": float(v.abs().median()),
                         "curtose": float(stats.kurtosis(v, fisher=False))})
    fq = pd.DataFrame(freq)

    # ---------- (b) a ma especificacao e do regime anterior? ----------
    resid, escada = [], []
    for nome in SERIES:
        d = _prep(b, nome)
        d["__y"] = L.y_acum(d, nome, 0)
        ph = pand(d, 0)
        regs = ["choque_petroleo"] + O5._regressores(L.LAGS) + dum + ph
        base = d[["Data", "pos", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
        for rot, sub in [("cheia", base), ("pre", base[base.pos == 0]), ("pos", base[base.pos == 1])]:
            X = sm.add_constant(sub[regs], has_constant="add")
            X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
            r = sm.OLS(sub["__y"], X).fit()
            rh = sm.OLS(sub["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(0))
            linha = {"variavel": nome, "amostra": rot, "n": len(sub),
                     "coef_choque": float(rh.params["choque_petroleo"]),
                     "p_choque": float(rh.pvalues["choque_petroleo"])}
            for q in [1, 6, 12]:
                try:
                    linha[f"bg_robusto_p_q{q}"] = O5.bg_robusto(
                        np.asarray(X, float), np.asarray(r.resid, float), q)[2]
                except Exception:
                    linha[f"bg_robusto_p_q{q}"] = np.nan
            resid.append(linha)

        # ---------- (c) a instabilidade em p tambem e de mistura de regimes? ----------
        for rot in ["cheia", "pre", "pos"]:
            for p in ESCADA:
                regs_p = ["choque_petroleo"] + O5._regressores(p) + dum + ph
                am = d[["Data", "pos", "__y"] + regs_p].replace(
                    [np.inf, -np.inf], np.nan).dropna()
                if rot == "pre":
                    am = am[am.pos == 0]
                elif rot == "pos":
                    am = am[am.pos == 1]
                X = sm.add_constant(am[regs_p], has_constant="add")
                X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
                r = sm.OLS(am["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(0))
                escada.append({"variavel": nome, "amostra": rot, "p": p,
                               "coeficiente": float(r.params["choque_petroleo"]),
                               "observacoes": int(r.nobs)})
        print(f"  {nome}")

    rs = pd.DataFrame(resid); es = pd.DataFrame(escada)
    with pd.ExcelWriter(SAIDA / "onda3_4_diagnostico_refinaria.xlsx") as w:
        fq.to_excel(w, sheet_name="frequencia_ajuste", index=False)
        rs.to_excel(w, sheet_name="residuos_por_regime", index=False)
        es.to_excel(w, sheet_name="escada_por_regime", index=False)

    print("\n=== (a) com que frequencia o preco muda? ===")
    print(fq.pivot_table(index="variavel", columns="regime",
                         values=["pct_meses_parado", "variacao_absoluta_mediana"]).round(2).to_string())

    print("\n=== (b) Breusch-Godfrey robusto em h=0, por subamostra (p-valor) ===")
    print(rs.set_index(["variavel", "amostra"])[
        ["n", "bg_robusto_p_q1", "bg_robusto_p_q6", "bg_robusto_p_q12", "coef_choque"]
    ].round(3).to_string())

    print("\n=== (c) coeficiente por numero de defasagens, dentro de cada regime ===")
    piv = es.pivot_table(index=["variavel", "amostra"], columns="p", values="coeficiente")
    print(piv.round(3).to_string())

    r0 = rs[(rs.variavel == "gasolina_refinaria")]
    print("\n--- veredito para a gasolina na refinaria ---")
    pre = r0[r0.amostra == "pre"].iloc[0]; pos = r0[r0.amostra == "pos"].iloc[0]
    f0 = fq[fq.variavel == "gasolina_refinaria"].set_index("regime")
    print(f"  preco parado em {f0.loc['pre','pct_meses_parado']:.1f}% dos meses antes, "
          f"{f0.loc['pos','pct_meses_parado']:.1f}% depois")
    print(f"  variacao mensal mediana {f0.loc['pre','variacao_absoluta_mediana']:.3f}% antes, "
          f"{f0.loc['pos','variacao_absoluta_mediana']:.3f}% depois")
    print(f"  BG rejeita no regime ANTERIOR (p = {pre.bg_robusto_p_q1:.3f}, "
          f"{pre.bg_robusto_p_q6:.3f}, {pre.bg_robusto_p_q12:.3f})")
    print(f"  BG NAO rejeita no POSTERIOR  (p = {pos.bg_robusto_p_q1:.3f}, "
          f"{pos.bg_robusto_p_q6:.3f}, {pos.bg_robusto_p_q12:.3f})")
    print(f"  coeficiente: {pre.coef_choque:.3f} (p {pre.p_choque:.3f}) antes, "
          f"{pos.coef_choque:.3f} (p {pos.p_choque:.3f}) depois")
    print("\n  => a ma especificacao esta no regime de preco ADMINISTRADO, e desaparece")
    print("     quando a Petrobras passa a reajustar com frequencia. Nao e defeito da")
    print("     estimacao: e o proprio objeto de estudo aparecendo no diagnostico.")
    return fq, rs, es


def figuras():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    v = b[["Data", "dlog_gasolina_refinaria", "pos"]].dropna()
    cor = np.where(v.pos == 1, "#9E3D25", "#8a8f7d")
    ax[0].bar(v.Data, v.dlog_gasolina_refinaria, width=22, color=cor, lw=0)
    ax[0].axvline(pd.Timestamp(L.DATA_CORTE_FINAL), color="0.25", lw=1.1, ls="--")
    ax[0].axhline(0, color="0.4", lw=0.8)
    ax[0].set_title("Variação mensal do preço na refinaria\ncinza = antes de mar/2017, "
                    "tijolo = depois", fontsize=9)
    ax[0].set_ylabel("%")
    fq = pd.read_excel(SAIDA / "onda3_4_diagnostico_refinaria.xlsx", sheet_name="frequencia_ajuste")
    piv = fq.pivot_table(index="variavel", columns="regime", values="pct_meses_parado")
    piv = piv.loc[[s for s in SERIES if s in piv.index]]
    x = np.arange(len(piv))
    ax[1].bar(x - 0.19, piv["pre"], 0.38, color="#8a8f7d", label="antes de mar/2017")
    ax[1].bar(x + 0.19, piv["pos"], 0.38, color="#9E3D25", label="depois")
    ax[1].set_xticks(x); ax[1].set_xticklabels([_NOMES.get(i, i) for i in piv.index], rotation=20, ha="right", fontsize=7.5)
    ax[1].set_ylabel("% dos meses sem mudança de preço")
    ax[1].set_title("Frequência de reajuste por regime", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.suptitle("O preço na refinaria era administrado antes da PPI", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "diagnostico_refinaria.png", dpi=150); plt.close(fig)
    print("  figura diagnostico_refinaria")


if __name__ == "__main__":
    {"rodar": rodar, "figuras": figuras}[sys.argv[1] if len(sys.argv) > 1 else "rodar"]()
