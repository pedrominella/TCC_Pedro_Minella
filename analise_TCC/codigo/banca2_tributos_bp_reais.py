# -*- coding: utf-8 -*-
"""(Banca 2) Tres exercicios pedidos pelo segundo parecer.

A. TRIBUTOS. O regime posterior contem intervencoes tributarias de primeira ordem
   sobre os precos que servem de dependente e de mediador, e o desenho nao as
   registra. Como todas rompem temporariamente o vinculo entre cotacao externa e
   preco final, o esperado e que ATENUEM beta_pos -- mas isso precisa ser
   mostrado, nao presumido, porque a hipotese de identificacao da secao 3.4 exige
   contaminacao comum aos dois regimes, e estas sao assimetricas por construcao.

B. BAI-PERRON SEQUENCIAL. A quebra estimada e unica, e o "regime anterior" funde
   duas fases institucionais distintas: alinhamento parcial ate cerca de 2010 e
   contencao deliberada de 2011 a 2014. Se o repasse em 2003-2010 nao era nulo, o
   zero estimado e media de um periodo com repasse e outro sem, e o contraste
   reportado SUBESTIMA a mudanca. Procura-se aqui uma segunda quebra dentro de
   cada segmento, pelo procedimento sequencial usual.

C. CHOQUE EM REAIS. Sob paridade de importacao o preco domestico acompanha Brent
   VEZES cambio. O choque do trabalho e a variacao do Brent em dolares, e o
   cambio entra apenas defasado e com coeficiente comum aos dois regimes.
   Reestima-se com o choque construido sobre o Brent em reais.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import lp_base_nova as L

SAIDA = L.SAIDA / "banca2"
SAIDA.mkdir(parents=True, exist_ok=True)
PRINCIPAIS = ["ipca_geral", "ipca_transportes", "gasolina_consumidor", "diesel"]

# Episodios tributarios de primeira ordem sobre combustiveis, em resolucao mensal.
TRIBUTOS = {
    "pis_cofins_2017": ["2017-07", "2017-08"],                  # Decreto 9.101, 20/07/2017
    "subvencao_diesel_2018": ["2018-06", "2018-07", "2018-08",  # pos-greve de mai/2018
                              "2018-09", "2018-10", "2018-11", "2018-12"],
    "desoneracao_2022": ["2022-03", "2022-06", "2022-07"],      # LC 192/2022 e LC 194/2022
    "reoneracao_2023": ["2023-03"],                             # retomada parcial
}


def _base(nome, b):
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    din = []
    for lag in range(1, L.LAGS + 1):
        din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
               [f"{c}_lag{lag}" for c in L.CONTROLES]
    return d, din


def _prep(choque="usd"):
    b, dum, pand = L.carregar_pandemia("corrigida")
    if choque == "brl":
        brl = b["Preco_Barril"] * b["Cambio"]
        ch = 100 * np.log(brl).diff()
        b["choque_petroleo"] = ch / ch.std()
        for lag in range(1, L.LAGS + 1):
            b[f"choque_petroleo_lag{lag}"] = b["choque_petroleo"].shift(lag)
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
    return b, dum, pand


def _ajusta(d, nome, h, cab, extra, dum, pand):
    d = d.copy()
    d["__y"] = L.y_acum(d, nome, h)
    _, din = _base(nome, d)
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    regs = list(cab) + ["regime_pos"] + din + dum + list(extra) + pand(d, h)
    am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(am[regs], has_constant="add")
    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
    return sm.OLS(am["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(h)), X


# ------------------------------------------------------------------ A
def tributos():
    b, dum, pand = _prep()
    mi = b["Data"].dt.strftime("%Y-%m")
    nomes = []
    for k, meses in TRIBUTOS.items():
        b[k] = mi.isin(meses).astype(float)
        nomes.append(k)
    print("  indicadoras tributarias:",
          {k: int(b[k].sum()) for k in nomes}, "meses")

    linhas = []
    for nome in PRINCIPAIS:
        for h in [0, 1, 3]:
            for modo, extra in [("sem tributos", []), ("com tributos", nomes)]:
                r, X = _ajusta(b, nome, h, ["choque_pre", "choque_pos"], extra, dum, pand)
                w = r.wald_test("choque_pre = choque_pos", scalar=True)
                linhas.append({"variavel": nome, "h": h, "modo": modo,
                               "coef_pre": r.params["choque_pre"], "ep_pre": r.bse["choque_pre"],
                               "coef_pos": r.params["choque_pos"], "ep_pos": r.bse["choque_pos"],
                               "p_pos": r.pvalues["choque_pos"],
                               "wald_p": float(w.pvalue), "obs": int(r.nobs)})
        print(f"  tributos {nome}")
    t = pd.DataFrame(linhas)
    t.to_excel(SAIDA / "tributos.xlsx", index=False)
    print("\n=== A. TRIBUTOS ===")
    print(f"  {'variavel':22s} {'h':>2s} {'pre s/':>7s} {'pre c/':>7s} "
          f"{'pos s/':>7s} {'pos c/':>7s} {'wald s/':>8s} {'wald c/':>8s}")
    for nome in PRINCIPAIS:
        for h in [0, 1, 3]:
            s = t[(t.variavel == nome) & (t.h == h) & (t.modo == "sem tributos")].iloc[0]
            c = t[(t.variavel == nome) & (t.h == h) & (t.modo == "com tributos")].iloc[0]
            print(f"  {nome:22s} {h:2d} {s.coef_pre:7.3f} {c.coef_pre:7.3f} "
                  f"{s.coef_pos:7.3f} {c.coef_pos:7.3f} {s.wald_p:8.3f} {c.wald_p:8.3f}")
    return t


# ------------------------------------------------------------------ B
def bai_perron():
    """Busca sequencial de segunda quebra dentro de cada segmento."""
    b, dum, pand = _prep()
    datas = pd.DatetimeIndex(b["Data"])
    tau = pd.Timestamp(L.DATA_CORTE_FINAL)
    MIN = 36                                   # segmento minimo, em meses

    resultados = []
    for seg, (ini, fim) in [("anterior", (datas.min(), tau)),
                            ("posterior", (tau, datas.max()))]:
        dentro = (datas >= ini) & (datas < fim) if seg == "anterior" else (datas >= ini)
        cand = datas[dentro]
        cand = cand[MIN:len(cand) - MIN]
        perfil = []
        for c in cand:
            soma = 0.0
            for nome in PRINCIPAIS:
                d = b.copy()
                d["s1"] = d["choque_petroleo"] * np.asarray(dentro & (datas < c), float)
                d["s2"] = d["choque_petroleo"] * np.asarray(dentro & (datas >= c), float)
                d["fora"] = d["choque_petroleo"] * np.asarray(~dentro, float)
                d["seg2"] = np.asarray(dentro & (datas >= c), float)
                try:
                    r, _ = _ajusta(d, nome, 0, ["s1", "s2", "fora"], ["seg2"], dum, pand)
                    soma += float(r.wald_test("s1 = s2", scalar=True).statistic)
                except Exception:
                    pass
            perfil.append({"segmento": seg, "data": c, "wald_soma": soma})
        p = pd.DataFrame(perfil)
        top = p.sort_values("wald_soma", ascending=False).head(3)
        resultados.append(p)
        print(f"\n  segmento {seg}: {len(p)} datas candidatas")
        for _, x in top.iterrows():
            print(f"    {x.data:%Y-%m}  soma de Wald = {x.wald_soma:.2f}")
    t = pd.concat(resultados)
    t.to_excel(SAIDA / "bai_perron.xlsx", index=False)
    return t


# ------------------------------------------------------------------ C
def reais():
    linhas = []
    for choque, rot in [("usd", "Brent em dólares"), ("brl", "Brent em reais")]:
        b, dum, pand = _prep(choque)
        for nome in PRINCIPAIS:
            for h in [0, 1, 3]:
                r, _ = _ajusta(b, nome, h, ["choque_pre", "choque_pos"], [], dum, pand)
                w = r.wald_test("choque_pre = choque_pos", scalar=True)
                linhas.append({"choque": rot, "variavel": nome, "h": h,
                               "coef_pre": r.params["choque_pre"],
                               "coef_pos": r.params["choque_pos"],
                               "ep_pos": r.bse["choque_pos"],
                               "p_pos": r.pvalues["choque_pos"],
                               "wald_p": float(w.pvalue)})
        print(f"  choque {rot}")
    t = pd.DataFrame(linhas)
    t.to_excel(SAIDA / "choque_reais.xlsx", index=False)
    print("\n=== C. CHOQUE EM REAIS ===")
    print(f"  {'variavel':22s} {'h':>2s} {'pre USD':>8s} {'pre BRL':>8s} "
          f"{'pos USD':>8s} {'pos BRL':>8s} {'wald USD':>9s} {'wald BRL':>9s}")
    for nome in PRINCIPAIS:
        for h in [0, 1, 3]:
            u = t[(t.variavel == nome) & (t.h == h) & (t.choque == "Brent em dólares")].iloc[0]
            r_ = t[(t.variavel == nome) & (t.h == h) & (t.choque == "Brent em reais")].iloc[0]
            print(f"  {nome:22s} {h:2d} {u.coef_pre:8.3f} {r_.coef_pre:8.3f} "
                  f"{u.coef_pos:8.3f} {r_.coef_pos:8.3f} {u.wald_p:9.3f} {r_.wald_p:9.3f}")
    # correlacao entre choque e cambio por regime
    b, _, _ = _prep()
    b["dcam"] = b["dlog_cambio"]
    for reg, m in [("anterior", b.regime_pos == 0), ("posterior", b.regime_pos == 1)]:
        x = b[m][["choque_petroleo", "dcam"]].dropna()
        print(f"  corr(choque Brent USD, variação cambial) no regime {reg}: "
              f"{x.corr().iloc[0, 1]:+.3f}")
    return t


if __name__ == "__main__":
    qual = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    if qual in ("tudo", "tributos"):
        tributos()
    if qual in ("tudo", "reais"):
        reais()
    if qual in ("tudo", "bp"):
        bai_perron()
