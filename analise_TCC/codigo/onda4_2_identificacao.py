# -*- coding: utf-8 -*-
"""
ONDA 4.2 -- IDENTIFICACAO DO DESENHO LIGA-DESLIGA

O modelo de tres regimes (onda4_1) mostra que o repasse sobe em mar/2017 e cai
em jun/2023. Este script testa se esse padrao pode ser levado a serio como
efeito da POLITICA, e nao como acidente amostral. Quatro exercicios:

  (1) POTENCIA. O coeficiente do 3o regime vem de poucos meses. Se o choque
      quase nao variou depois de 2023, o coeficiente e mal identificado e o
      "desligou" pode ser so falta de variacao. Mede-se o desvio-padrao do
      choque DENTRO de cada regime e a variacao efetiva que identifica cada
      coeficiente.

  (2) VOLTA A LINHA DE BASE. A previsao forte da hipotese institucional nao e
      apenas "caiu": e que o repasse volte ao patamar anterior a politica. Testa
      b_pre = b_pos, por horizonte e conjuntamente em h=0..6. Nao rejeitar aqui
      e o resultado favoravel, entao o teste vem acompanhado da checagem de
      potencia do item (1): nao rejeitar com desenho fraco nao significa nada.

  (3) PLACEBO DE DATA. Se a quebra de 2023 fosse artefato, datas falsas dentro
      do periodo PPI produziriam rejeicoes parecidas. Varre-se cada mes de
      jan/2019 a dez/2022 como fim FALSO da PPI, mantendo mar/2017, e compara-se
      a estatistica dessas datas com a da data verdadeira.

  (4) EXOGENEIDADE DO CHOQUE AO REGIME. A media e a variancia do choque nao
      podem diferir sistematicamente entre regimes de um jeito que explique o
      resultado. Reporta-se media, desvio e teste de igualdade de variancia.

Uso: python onda4_2_identificacao.py
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
from onda2_bandas_placebo import (ajustar, replicas, hac_conjunta_horizontes,
                                  _amostra)
import onda4_1_tres_regimes as O41

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"
H_CONJUNTO = 6
B, BLOCO, SEMENTE = 999, 12, 20260831
PLACEBOS = pd.date_range("2019-01-01", "2022-12-01", freq="MS")


def prep(b, nome):
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    din = []
    for lag in range(1, L.LAGS + 1):
        din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
               [f"{c}_lag{lag}" for c in L.CONTROLES]
    return d, din


def potencia():
    """(1) e (4): quanta variacao de choque identifica cada regime."""
    b, _, _ = O41.montar(O41.CORTE_FIM)
    b["bloco"] = np.where(b.d_pos == 1, "pos-PPI",
                          np.where(b.d_ppi == 1, "PPI", "pre-PPI"))
    linhas = []
    for r, g in b.groupby("bloco"):
        c = g["choque_petroleo"].dropna()
        linhas.append({"regime": r, "meses": len(c), "media": c.mean(),
                       "desvio": c.std(), "min": c.min(), "max": c.max(),
                       "soma_quadrados": float((c ** 2).sum())})
    t = pd.DataFrame(linhas).set_index("regime").loc[["pre-PPI", "PPI", "pos-PPI"]]
    # Levene: a variancia do choque difere entre regimes?
    gr = [b.loc[b.bloco == r, "choque_petroleo"].dropna().values
          for r in ["pre-PPI", "PPI", "pos-PPI"]]
    lev = stats.levene(*gr, center="median")
    ano = stats.f_oneway(*gr)
    return t, float(lev.pvalue), float(ano.pvalue)


def volta_a_base():
    """(2) b_pre = b_pos, por horizonte e conjunto h=0..6."""
    rng = np.random.default_rng(SEMENTE)
    b, dum, pand = O41.montar(O41.CORTE_FIM)
    datas_ref = pd.DatetimeIndex(sorted(b["Data"].unique()))
    n = len(datas_ref); nb = int(np.ceil(n / BLOCO))
    etas = np.column_stack([np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:n]
                            for _ in range(B)])
    cab = O41.REGIMES + ["d_ppi", "d_pos"]
    linhas = []
    for nome in L.VARIAVEIS:
        d, din = prep(b, nome)
        mont = []
        for h in range(0, H_CONJUNTO + 1):
            d["__y"] = L.y_acum(d, nome, h)
            regs = cab + din + dum + pand(d, h)
            mont.append((h, regs, _amostra(d, regs)))
        com = set.intersection(*[set(am["Data"]) for _, _, am in mont])
        gs = [ajustar(am[am["Data"].isin(com)], regs, h, ("c_pre", "c_pos"))
              for h, regs, am in mont]
        V = hac_conjunta_horizontes(gs, H_CONJUNTO + 1)
        Vi = np.linalg.pinv(V)
        bh = np.array([g["valor"] for g in gs])
        W = float(bh @ Vi @ bh)
        Bm = np.column_stack([replicas(g, etas, datas_ref)[0] for g in gs])
        Wb = np.einsum("ij,jk,ik->i", Bm - bh, Vi, Bm - bh)
        linhas.append({"variavel": nome, "teste": "pre = pos (h=0..6)",
                       "estatistica": W, "p_bootstrap": float((Wb >= W).mean()),
                       "critico_95": float(np.quantile(Wb, 0.95)),
                       "dif_h0": bh[0], "ep_h0": gs[0]["ep"]})
    return pd.DataFrame(linhas)


def placebo_datas():
    """(3) datas falsas de fim da PPI, dentro do proprio periodo PPI."""
    b0, dum, pand = O41.montar(O41.CORTE_FIM)
    verdadeiro = {}
    linhas = []
    for nome in ["ipca_geral", "ipca_transportes", "gasolina_consumidor", "diesel"]:
        # estatistica na data verdadeira
        d, din = prep(b0, nome)
        cab = O41.REGIMES + ["d_ppi", "d_pos"]
        d["__y"] = L.y_acum(d, nome, 0)
        regs = cab + din + dum + pand(d, 0)
        g = ajustar(_amostra(d, regs), regs, 0, ("c_ppi", "c_pos"))
        verdadeiro[nome] = abs(g["valor"] / g["ep"])

        for data in PLACEBOS:
            bf, dumf, pandf = O41.montar(data.strftime("%Y-%m-%d"))
            df, dinf = prep(bf, nome)
            df["__y"] = L.y_acum(df, nome, 0)
            regsf = cab + dinf + dumf + pandf(df, 0)
            gf = ajustar(_amostra(df, regsf), regsf, 0, ("c_ppi", "c_pos"))
            linhas.append({"variavel": nome, "data_falsa": data,
                           "diferenca": gf["valor"], "erro_padrao": gf["ep"],
                           "abs_t": abs(gf["valor"] / gf["ep"]),
                           "n_apos": int(bf.d_pos.sum())})
        print(f"  {nome}: |t| verdadeiro = {verdadeiro[nome]:.2f}")
    t = pd.DataFrame(linhas)
    t["t_verdadeiro"] = t.variavel.map(verdadeiro)
    t["placebo_maior"] = t.abs_t >= t.t_verdadeiro
    return t


def main():
    print(__doc__)

    print("=== (1) e (4) potencia e exogeneidade do choque ===")
    tp, p_lev, p_ano = potencia()
    print(tp.round(3).to_string())
    print(f"\nLevene (igualdade de variancia do choque entre regimes): p = {p_lev:.3f}")
    print(f"ANOVA  (igualdade de media do choque entre regimes)     : p = {p_ano:.3f}")
    print("Nao rejeitar e o resultado desejado: o choque nao muda de natureza")
    print("entre regimes, logo a mudanca de coeficiente nao vem do regressor.")

    print("\n=== (2) o repasse volta ao patamar pre-politica? ===")
    tv = volta_a_base()
    print(tv.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print("p ALTO aqui e favoravel: nao se distingue o pos-PPI do pre-PPI.")

    print("\n=== (3) placebo: datas falsas de fim da PPI (jan/2019 a dez/2022) ===")
    tpl = placebo_datas()
    res = tpl.groupby("variavel").agg(
        placebos=("abs_t", "size"),
        maiores_que_o_verdadeiro=("placebo_maior", "sum"),
        t_max_placebo=("abs_t", "max"),
        t_verdadeiro=("t_verdadeiro", "first")).round(3)
    print(res.to_string())
    print("\np empirico do placebo (fracao de datas falsas com |t| >= verdadeiro):")
    for v, x in res.iterrows():
        print(f"  {v:22s} {x.maiores_que_o_verdadeiro / x.placebos:.3f}")

    with pd.ExcelWriter(SAIDA / "onda4_2_identificacao.xlsx") as w:
        tp.reset_index().to_excel(w, sheet_name="potencia", index=False)
        tv.to_excel(w, sheet_name="volta_a_base", index=False)
        tpl.to_excel(w, sheet_name="placebo_datas", index=False)
        res.reset_index().to_excel(w, sheet_name="placebo_resumo", index=False)
    print(f"\ngravado: {SAIDA / 'onda4_2_identificacao.xlsx'}")


if __name__ == "__main__":
    main()
