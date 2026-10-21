# -*- coding: utf-8 -*-
"""
INDICADORAS PARA A GREVE DOS CAMINHONEIROS E PARA O CHOQUE TRIBUTARIO DE 2022

Os dois maiores residuos do modelo principal em h = 0 sao jun/2018 (+1,04) e
jul/2022 (-0,77). Ambos correspondem a eventos datados e exogenos ao preco do
petroleo:

  greve  21 a 31 de maio de 2018. Paralisacao do transporte rodoviario de carga.
         O acordo que a encerrou impos reducao do preco do diesel, razao pela
         qual em jun/2018 o diesel CAI 5,83 enquanto a gasolina SOBE 4,88 e o
         choque do petroleo e negativo (-0,31). Efeito sobre precos em mai, jun
         e jul de 2018.

  icms   Lei Complementar 194, de 23 de junho de 2022, que fixou teto de ICMS
         sobre combustiveis ao classifica-los como bens essenciais, somada a
         desoneracao federal de PIS/Cofins. Em jul/2022 a gasolina ao consumidor
         cai 16,82 enquanto na refinaria cai 3,83: a diferenca de treze pontos e
         tributaria. Efeito em jul, ago e set de 2022.

ATENCAO. Escolher indicadora olhando o residuo e busca de especificacao. A
justificativa aqui e institucional e anterior ao dado: as duas datas sao
publicas e verificaveis, e o teste abaixo mostra que o efeito estimado tem o
sinal e a ordem de grandeza do evento. Nao se deve repetir o procedimento para
cada residuo grande.

As indicadoras entram na forma de janela 1{t <= s <= t+h}, coerente com a
dependente acumulada, igual as de pandemia.

Uso: python teste_greve_icms.py
"""
import sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import lp_base_nova as L

SERIES = ["ipca_geral", "ipca_transportes", "gasolina_consumidor",
          "diesel", "gasolina_refinaria", "etanol"]

EVENTOS = {
    "greve_jun":  ["2018-06"],
    "greve":      ["2018-05", "2018-06", "2018-07"],
    "icms_jul":   ["2022-07"],
    "icms":       ["2022-07", "2022-08", "2022-09"],
}
VARIANTES = {
    "base":            [],
    "+greve(jun)":     ["greve_jun"],
    "+greve(mai-jul)": ["greve"],
    "+icms(jul)":      ["icms_jul"],
    "+icms(jul-set)":  ["icms"],
    "+greve+icms":     ["greve", "icms"],
}


def idx(datas):
    return set(pd.Timestamp(x).year * 12 + pd.Timestamp(x).month for x in datas)


def prepara():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["mi"] = b["Data"].dt.year * 12 + b["Data"].dt.month
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
    return b, dum, pand


def extras(d, h, quais):
    """Indicadoras de janela dos eventos pedidos. Uma por mes do evento."""
    nomes = []
    mi = d["mi"].values
    for ev in quais:
        for s in idx(EVENTOS[ev]):
            c = f"{ev}_{s}"
            d[c] = np.array([1.0 if (m <= s <= m + h) else 0.0 for m in mi])
            nomes.append(c)
    return nomes


def roda(b, dum, pand, nome, h, quais, corte=None):
    d = b.copy()
    if corte is not None:
        d["regime_pos"] = L.regime(d.Data, corte)
        d["choque_pre"] = d["choque_petroleo"] * (1 - d["regime_pos"])
        d["choque_pos"] = d["choque_petroleo"] * d["regime_pos"]
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    regs = ["choque_pre", "choque_pos", "regime_pos"]
    for lag in range(1, L.LAGS + 1):
        regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                [f"{c}_lag{lag}" for c in L.CONTROLES]
    regs += dum + L.controles_pandemia(d, h, "janela") + extras(d, h, quais)
    d["__y"] = L.y_acum(d, nome, h)
    am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
    r = sm.OLS(am["__y"], sm.add_constant(am[regs], has_constant="add")).fit(
        cov_type="HAC", cov_kwds=L.hac_kw(h))
    w = r.wald_test("choque_pre = choque_pos", scalar=True)
    return dict(coef=r.params["choque_pos"], p=r.pvalues["choque_pos"],
                wald=float(w.pvalue), K=int(r.df_model + 1), r=r,
                res=r.resid.values, datas=am.Data.values, wstat=float(w.statistic))


def main():
    b, dum, pand = prepara()

    # ---------------- o que as indicadoras estimam
    print("EFEITO ESTIMADO DAS PROPRIAS INDICADORAS, h = 0")
    print("  (coeficiente = deslocamento do mes, em pontos percentuais)")
    for nome in SERIES:
        o = roda(b, dum, pand, nome, 0, ["greve", "icms"])
        pj = [(k, o["r"].params[k], o["r"].pvalues[k])
              for k in o["r"].params.index if k.startswith(("greve_", "icms_"))]
        pj.sort(key=lambda x: x[0])
        print("  %-20s" % nome + "  ".join("%s %+6.2f%s" % (
            k.split("_")[0][:5] + k.split("_")[1][-2:], v,
            "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else "")
            for k, v, p in pj))

    # ---------------- efeito sobre o objeto do trabalho
    for h in [0, 1, 3]:
        print("\n" + "=" * 100)
        print("HORIZONTE h = %d" % h)
        print("%-20s" % "variavel" + "".join("%17s" % v for v in VARIANTES))
        for nome in SERIES:
            cel = []
            for v, quais in VARIANTES.items():
                o = roda(b, dum, pand, nome, h, quais)
                cel.append("%6.3f/%5.3f" % (o["coef"], o["wald"]))
            print("%-20s" % nome + "".join("%17s" % c for c in cel))
        ks = [roda(b, dum, pand, "ipca_geral", h, q)["K"] for q in VARIANTES.values()]
        print("%-20s" % "parametros K" + "".join("%17d" % k for k in ks))
    print("\n  cada celula: coeficiente do regime posterior / p-valor do teste de Wald")

    # ---------------- residuos que sobram
    print("\n" + "=" * 100)
    print("MAIORES RESIDUOS EM h = 0 (ipca_geral), ANTES E DEPOIS DAS INDICADORAS")
    for rot, quais in [("base", []), ("+greve+icms", ["greve", "icms"])]:
        o = roda(b, dum, pand, "ipca_geral", 0, quais)
        s = pd.Series(o["res"], index=pd.to_datetime(o["datas"]))
        top = s.abs().nlargest(5)
        print("  %-14s desvio %.3f | " % (rot, s.std(ddof=1))
              + ", ".join("%s %+.2f" % (d.strftime("%m/%y"), s[d]) for d in top.index))

    # ---------------- data de quebra
    print("\n" + "=" * 100)
    print("DATA DE QUEBRA: sup-Wald conjunto das quatro principais, h = 0")
    datas = pd.date_range("2008-01-01", "2021-12-01", freq="MS")
    for rot, quais in [("base", []), ("+greve+icms", ["greve", "icms"])]:
        perfil = {}
        for dt in datas:
            perfil[dt] = sum(roda(b, dum, pand, n, 0, quais, corte=dt)["wstat"]
                             for n in SERIES[:4])
        s = pd.Series(perfil).sort_values(ascending=False)
        print("  %-14s pico %s (%.1f) | mar/2017 = %.1f | 5 maiores: %s"
              % (rot, s.index[0].strftime("%m/%Y"), s.iloc[0],
                 perfil[pd.Timestamp("2017-03-01")],
                 ", ".join("%s %.0f" % (d.strftime("%m/%y"), v) for d, v in s.head(5).items())))


if __name__ == "__main__":
    main()
