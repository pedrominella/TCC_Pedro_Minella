# -*- coding: utf-8 -*-
"""
PRIMEIRA E SEGUNDA VOLTA, POR REGIME

Motivo: a decomposicao original (decompor(), em onda2_7_segunda_volta.py) roda em
AMOSTRA CHEIA, com um unico choque. O capitulo inteiro, porem, e por regime, e o
proprio trabalho argumenta que a media de amostra cheia mistura dois regimes e nao
e parametro estrutural. Este arquivo refaz a decomposicao separando os regimes.

Mantem tudo o mais igual a decompor():
  - desenho COMUM, com as defasagens do IPCA ex-combustiveis em todas as equacoes,
    para que a identidade total - primeira = segunda feche exatamente
  - os tres cenarios de peso
  - HAC com banda h+1 e a mesma rotina de ajuste (O2.ajustar)
  - conferencia contra a especificacao de referencia, que usa as defasagens do
    proprio IPCA Geral

  total(r)    = coeficiente de choque_r na equacao do IPCA Geral
  primeira(r) = soma_j w_j * coeficiente de choque_r na equacao do combustivel j
  segunda(r)  = coeficiente de choque_r na equacao do IPCA ex-combustiveis

Uso: python onda2_7b_voltas_por_regime.py
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
import onda2_bandas_placebo as O2      # ajustar(), como em onda2_7
from onda2_7_segunda_volta import PESOS, preparar, _dinamica

SAIDA = Path(__file__).resolve().parent.parent / "resultados"
ALVOS = [("choque_pre", "anterior"), ("choque_pos", "posterior")]


def main():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b, _ = preparar(b)
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]

    linhas, pior = [], 0.0
    for cen, w in PESOS.items():
        nome_ex = f"ipca_ex_comb_{cen}"
        d = b.copy()
        dinamica = _dinamica(d, nome_ex)          # desenho COMUM
        for h in range(0, L.H_MAX + 1):
            regs = ["choque_pre", "choque_pos", "regime_pos"] + dinamica + dum + pand(d, h)
            aj = {}
            for nome in ["ipca_geral", nome_ex] + list(w):
                d["__y"] = L.y_acum(d, nome, h)
                am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                aj[nome] = {a: O2.ajustar(am, regs, h, a) for a, _ in ALVOS}
            for a, rot in ALVOS:
                total = aj["ipca_geral"][a]["valor"]
                primeira = sum(w[f] * aj[f][a]["valor"] for f in w)
                g = aj[nome_ex][a]
                pior = max(pior, abs((total - primeira) - g["valor"]))
                linhas.append({
                    "cenario": cen, "regime": rot, "alvo": a, "h": h,
                    "efeito_total": total, "primeira_volta": primeira,
                    "segunda_volta": g["valor"], "ep_segunda_volta": g["ep"],
                    "p_segunda_volta": float(2 * (1 - stats.norm.cdf(abs(g["valor"] / g["ep"])))),
                    "ic5_segunda": g["valor"] - L.Z_90 * g["ep"],
                    "ic95_segunda": g["valor"] + L.Z_90 * g["ep"],
                    "parcela_primeira": primeira / total if abs(total) > 1e-8 else np.nan,
                    "observacoes": g["nobs"]})
        # conferencia: o total sob desenho comum reproduz a especificacao de referencia?
        d2 = b.copy(); din2 = _dinamica(d2, "ipca_geral")
        for h in range(0, L.H_MAX + 1):
            d2["__y"] = L.y_acum(d2, "ipca_geral", h)
            regs2 = ["choque_pre", "choque_pos", "regime_pos"] + din2 + dum + pand(d2, h)
            am2 = d2[["Data", "__y"] + regs2].replace([np.inf, -np.inf], np.nan).dropna()
            for a, rot in ALVOS:
                gn = O2.ajustar(am2, regs2, h, a)
                for r in linhas:
                    if r["cenario"] == cen and r["h"] == h and r["alvo"] == a:
                        r["total_especificacao_referencia"] = gn["valor"]
        print("  cenario %s" % cen)

    t = pd.DataFrame(linhas)
    t["dif_total"] = (t.efeito_total - t.total_especificacao_referencia).abs()
    with pd.ExcelWriter(SAIDA / "onda2_7b_voltas_por_regime.xlsx") as w_:
        t.to_excel(w_, sheet_name="decomposicao", index=False)

    print("\nidentidade total - primeira = segunda: desvio maximo %.2e  (%s)"
          % (pior, "OK" if pior < 1e-8 else "FALHOU"))
    print("desvio maximo contra a especificacao de referencia: %.4f" % t.dif_total.max())
    print()
    for rot in ["anterior", "posterior"]:
        z = t[(t.cenario == "central") & (t.regime == rot) & (t.h.isin([0, 1, 2, 3, 6]))]
        print("=== REGIME %s (cenario central)" % rot.upper())
        print(z[["h", "efeito_total", "primeira_volta", "segunda_volta",
                 "p_segunda_volta", "parcela_primeira"]].round(3).to_string(index=False))
        print()
    print("=== faixa entre cenarios, regime posterior, segunda volta")
    z = t[(t.regime == "posterior") & (t.h.isin([0, 1, 2, 3, 6]))]
    print(z.pivot_table(index="h", columns="cenario",
                        values=["segunda_volta", "p_segunda_volta"]).round(3).to_string())
    print("\ngravado:", SAIDA / "onda2_7b_voltas_por_regime.xlsx")


if __name__ == "__main__":
    main()
