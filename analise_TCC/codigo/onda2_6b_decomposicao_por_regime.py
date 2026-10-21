# -*- coding: utf-8 -*-
"""
DECOMPOSICAO MENSAL DA RESPOSTA ACUMULADA, POR REGIME  (Tabela 3 do capitulo 4)

Motivo: identidade() em onda2_6 faz a decomposicao apenas para o choque unico de
amostra cheia. A Tabela 3 e a Figura 6 herdavam isso, enquanto o capitulo e por
regime. Auditoria 360, camada 2, 06/09/2026.

Mantem a construcao de identidade() intacta: desenho X FIXO no horizonte alvo,
dummies de pandemia na versao de janela daquele horizonte, e todas as regressoes
na amostra comum, de modo que

    beta_acumulado(H)  =  soma_{j=0..H} beta_mensal(j)

feche exatamente. A unica mudanca e que o alvo passa a ser choque_pre e
choque_pos em lugar de choque_petroleo.

Uso: python onda2_6b_decomposicao_por_regime.py
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
import onda2_bandas_placebo as O2
from onda2_6_resposta_nao_acumulada import y_mes, _base

SAIDA = Path(__file__).resolve().parent.parent / "resultados"
ALVOS = [("choque_pre", "anterior"), ("choque_pos", "posterior")]
HORIZ = [3, 6, 12]


def main():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]

    linhas, pior = [], 0.0
    for nome in L.VARIAVEIS:
        d, dinamica = _base(nome, b)
        for H in HORIZ:
            d["__y"] = L.y_acum(d, nome, H)
            regs = ["choque_pre", "choque_pos", "regime_pos"] + dinamica + dum + pand(d, H)
            base = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
            datas = set(base["Data"])
            for j in range(0, H + 1):
                d[f"__m{j}"] = y_mes(d, nome, j)
                datas &= set(d[["Data", f"__m{j}"] + regs].replace(
                    [np.inf, -np.inf], np.nan).dropna()["Data"])
            am = base[base["Data"].isin(datas)]
            for a, rot in ALVOS:
                g = O2.ajustar(am, regs, H, a)
                acum, ep = g["valor"], g["ep"]
                mensais, soma = [], 0.0
                for j in range(0, H + 1):
                    aj = d[["Data", f"__m{j}"] + regs].replace(
                        [np.inf, -np.inf], np.nan).dropna()
                    aj = aj[aj["Data"].isin(datas)].rename(columns={f"__m{j}": "__y"})
                    gj = O2.ajustar(aj, regs, H, a)
                    mensais.append(gj); soma += gj["valor"]
                pior = max(pior, abs(soma - acum))
                r = {"variavel": nome, "regime": rot, "alvo": a, "H": H,
                     "acumulada": acum, "ep_acumulada": ep,
                     "p_acumulada": float(2 * (1 - stats.norm.cdf(abs(acum / ep)))),
                     "soma_mensais": soma, "desvio_identidade": abs(soma - acum),
                     "observacoes": len(am)}
                for j, gj in enumerate(mensais):
                    r[f"mes_{j}"] = gj["valor"]
                    r[f"p_mes_{j}"] = float(2 * (1 - stats.norm.cdf(
                        abs(gj["valor"] / gj["ep"]))))
                if H >= 1:
                    r["meses_0e1"] = mensais[0]["valor"] + mensais[1]["valor"]
                    r["parcela_0e1"] = r["meses_0e1"] / acum if abs(acum) > 1e-8 else np.nan
                linhas.append(r)
        print("  %s" % nome)

    t = pd.DataFrame(linhas)
    t.to_excel(SAIDA / "onda2_6b_decomposicao_por_regime.xlsx", index=False)
    print("\nidentidade acumulada = soma das mensais: desvio maximo %.2e  (%s)"
          % (pior, "OK" if pior < 1e-8 else "FALHOU"))
    print()

    def est(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    for rot in ["posterior", "anterior"]:
        z = t[(t.regime == rot) & (t.H == 3)]
        print("=== REGIME %s, decomposicao do acumulado de tres meses" % rot.upper())
        print("%-24s %10s %9s %9s %9s %9s %8s" %
              ("variavel", "acumulado", "mes 0", "mes 1", "mes 2", "mes 3", "0 e 1"))
        for _, r in z.iterrows():
            print("%-24s %7.3f%-3s %6.3f%-3s %6.3f%-3s %6.3f%-3s %6.3f%-3s %7.0f%%" % (
                r.variavel, r.acumulada, est(r.p_acumulada),
                r.mes_0, est(r.p_mes_0), r.mes_1, est(r.p_mes_1),
                r.mes_2, est(r.p_mes_2), r.mes_3, est(r.p_mes_3),
                100 * r.parcela_0e1))
        print()
    print("gravado:", SAIDA / "onda2_6b_decomposicao_por_regime.xlsx")


if __name__ == "__main__":
    main()
