# -*- coding: utf-8 -*-
"""
O OXFORD STRINGENCY INDEX NO LUGAR DAS DEZ INDICADORAS DE PANDEMIA

Motivo: em h = 0 a indicadora de janela 1{t <= s <= t+h} colapsa em uma
indicadora por mes. Sao dez indicadoras para dez observacoes, que ficam
perfeitamente ajustadas e nao informam o coeficiente de impacto. No horizonte de
impacto a especificacao equivale a DESCARTAR os dez meses.

O indice de rigor do OxCGRT substitui esse bloco por um regressor continuo, que
mede a severidade das medidas de contencao em vez de marcar o mes. As dez
observacoes deixam de ser apagadas: passam a ser deslocadas na proporcao do
rigor efetivamente vigente.

Variantes estimadas, todas sobre a especificacao por regime de lp_base_nova:

  janela      dez indicadoras de janela            (o que esta no TCC hoje)
  ox_covid    Oxford em nivel, zerado fora de mar-dez/2020
  ox_janela   Oxford somado na janela [t, t+h], zerado fora de mar-dez/2020
  ox_cheio    Oxford em nivel em toda a sua extensao (jan/2020 a dez/2022)
  contagem    numero de meses de pandemia dentro da janela

ox_janela e a construcao coerente com a projecao local: a dependente acumula
os meses t..t+h, entao o controle acumula o rigor dos mesmos meses.

Uso: python teste_oxford_covid.py
"""
import sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import lp_base_nova as L

RAIZ = Path(__file__).resolve().parent.parent.parent
ARQ = RAIZ / "base_de_dados" / "Stringency_index.csv"
COVID = pd.date_range("2020-03-01", "2020-12-01", freq="MS")
CI = set(s.year * 12 + s.month for s in COVID)
SERIES = ["ipca_geral", "ipca_transportes", "gasolina_consumidor",
          "diesel", "gasolina_refinaria", "etanol"]
MODOS = ["janela", "ox_covid", "ox_janela", "ox_cheio", "contagem"]


def oxford():
    d = pd.read_csv(ARQ, sep=";", low_memory=False)
    d = d[(d.CountryCode == "BRA") & (d.Jurisdiction == "NAT_TOTAL")].copy()
    d["Data"] = pd.to_datetime(d["Date"].astype(str), format="%Y%m%d", errors="coerce")
    d["si"] = pd.to_numeric(d["StringencyIndex_Average"], errors="coerce")
    d = d.dropna(subset=["Data", "si"])
    m = (d.assign(mes=d.Data.values.astype("datetime64[M]"))
           .groupby("mes", as_index=False).si.mean().rename(columns={"mes": "Data"}))
    return m


def prepara():
    b, dum, pand = L.carregar_pandemia("corrigida")
    m = oxford()
    b = b.merge(m, on="Data", how="left")
    b["si"] = b["si"].fillna(0.0) / 100.0            # escala 0 a 1
    mi = b["Data"].dt.year * 12 + b["Data"].dt.month
    b["si_covid"] = np.where(mi.isin(CI), b["si"], 0.0)
    b["mi"] = mi
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
    return b, dum, pand, m


def monta(b, dum, pand, d, h, modo):
    """Devolve a lista de regressores do controle de pandemia no modo pedido."""
    mi = d["mi"].values
    if modo == "janela":
        return L.controles_pandemia(d, h, "janela")
    if modo == "ox_covid":
        return ["si_covid"]
    if modo == "ox_cheio":
        return ["si"]
    if modo == "ox_janela":
        base = dict(zip(b["mi"].values, b["si_covid"].values))
        d["ox_jan"] = [sum(base.get(m + k, 0.0) for k in range(h + 1)) for m in mi]
        return ["ox_jan"]
    if modo == "contagem":
        d["pand_n"] = [sum(1 for s in CI if m <= s <= m + h) for m in mi]
        return ["pand_n"]
    raise ValueError(modo)


def roda(b, dum, pand, nome, h, modo, corte=None):
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
    regs += dum + monta(b, dum, pand, d, h, modo)
    d["__y"] = L.y_acum(d, nome, h)
    am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
    r = sm.OLS(am["__y"], sm.add_constant(am[regs], has_constant="add")).fit(
        cov_type="HAC", cov_kwds=L.hac_kw(h))
    w = r.wald_test("choque_pre = choque_pos", scalar=True)
    return dict(coef=r.params["choque_pos"], p=r.pvalues["choque_pos"],
                wald=float(w.pvalue), n=int(r.nobs), K=int(r.df_model + 1),
                res=r.resid.values, datas=am.Data.values, r2=r.rsquared)


def main():
    b, dum, pand, m = prepara()
    print("OXFORD STRINGENCY, BRASIL, MENSAL")
    print("  %d meses, de %s a %s | faixa %.1f a %.1f"
          % (len(m), m.Data.min().strftime("%Y-%m"), m.Data.max().strftime("%Y-%m"),
             m.si.min(), m.si.max()))
    dentro = m[m.Data.isin(COVID)]
    print("  nos dez meses de mar a dez/2020: faixa %.1f a %.1f, media %.1f"
          % (dentro.si.min(), dentro.si.max(), dentro.si.mean()))
    print("  valores mensais:", ", ".join("%s %.0f" % (r.Data.strftime("%m/%y"), r.si)
                                          for _, r in dentro.iterrows()))
    fora = m[(~m.Data.isin(COVID)) & (m.si > 0)]
    print("  fora da janela o indice ainda cobre %d meses (%s a %s), media %.1f"
          % (len(fora), fora.Data.min().strftime("%Y-%m"),
             fora.Data.max().strftime("%Y-%m"), fora.si.mean()))

    # ---------------- residuo em h=0: quantos sao exatamente zero
    print("\nRESIDUO DE IMPACTO NOS DEZ MESES DE PANDEMIA (ipca_geral, h = 0)")
    for modo in MODOS:
        o = roda(b, dum, pand, "ipca_geral", 0, modo)
        dt = pd.to_datetime(o["datas"])
        u = pd.Series(o["res"], index=dt).reindex(COVID)
        zer = int((u.abs() < 1e-10).sum())
        print("  %-10s  %2d de 10 resíduos exatamente zero | desvio nesses meses %.3f"
              % (modo, zer, u.std(ddof=1)))

    # ---------------- coeficientes
    for h in [0, 1, 3]:
        print("\n" + "=" * 92)
        print("HORIZONTE h = %d      b_pos (p) | Wald p" % h)
        print("%-20s" % "" + "".join("%22s" % s for s in MODOS[:3]))
        for nome in SERIES:
            cel = []
            for modo in MODOS[:3]:
                o = roda(b, dum, pand, nome, h, modo)
                cel.append("%7.3f (%.3f) %5.3f" % (o["coef"], o["p"], o["wald"]))
            print("%-20s" % nome + "".join("%22s" % c for c in cel))
        o1 = roda(b, dum, pand, "ipca_geral", h, "janela")
        o2 = roda(b, dum, pand, "ipca_geral", h, "ox_janela")
        print("  parametros: janela K=%d | ox_janela K=%d" % (o1["K"], o2["K"]))

    # ---------------- extensao completa do indice, para mostrar o vazamento
    print("\n" + "=" * 92)
    print("EFEITO DE DEIXAR O INDICE CORRER ATE DEZ/2022 (h = 0)")
    print("%-20s %12s %12s %12s" % ("", "ox_covid", "ox_cheio", "diferença"))
    for nome in SERIES:
        a = roda(b, dum, pand, nome, 0, "ox_covid")
        c = roda(b, dum, pand, nome, 0, "ox_cheio")
        print("%-20s %12.3f %12.3f %12.3f"
              % (nome, a["coef"], c["coef"], c["coef"] - a["coef"]))

    # ---------------- data de quebra
    print("\n" + "=" * 92)
    print("DATA DE QUEBRA: sup-Wald conjunto das quatro principais, h = 0")
    datas = pd.date_range("2008-01-01", "2021-12-01", freq="MS")
    for modo in ["janela", "ox_covid", "ox_janela"]:
        perfil = {}
        for dt in datas:
            tot = 0.0
            for nome in SERIES[:4]:
                d = b.copy()
                d["regime_pos"] = L.regime(d.Data, dt)
                d["choque_pre"] = d["choque_petroleo"] * (1 - d["regime_pos"])
                d["choque_pos"] = d["choque_petroleo"] * d["regime_pos"]
                for lag in range(1, L.LAGS + 1):
                    d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
                regs = ["choque_pre", "choque_pos", "regime_pos"]
                for lag in range(1, L.LAGS + 1):
                    regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                            [f"{c}_lag{lag}" for c in L.CONTROLES]
                regs += dum + monta(b, dum, pand, d, 0, modo)
                d["__y"] = L.y_acum(d, nome, 0)
                am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                r = sm.OLS(am["__y"], sm.add_constant(am[regs], has_constant="add")).fit(
                    cov_type="HAC", cov_kwds=L.hac_kw(0))
                tot += float(r.wald_test("choque_pre = choque_pos", scalar=True).statistic)
            perfil[dt] = tot
        s = pd.Series(perfil).sort_values(ascending=False)
        print("  %-10s pico %s (%.1f) | mar/2017 = %.1f | 5 maiores: %s"
              % (modo, s.index[0].strftime("%m/%Y"), s.iloc[0],
                 perfil[pd.Timestamp("2017-03-01")],
                 ", ".join("%s %.0f" % (d.strftime("%m/%y"), v) for d, v in s.head(5).items())))


if __name__ == "__main__":
    main()
