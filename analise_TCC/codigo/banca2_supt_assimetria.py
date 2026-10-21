# -*- coding: utf-8 -*-
"""(Banca 2) Dois calculos pedidos pelo segundo parecer.

A. CRITERIO UNIFORME NA TABELA 2. A secao 3.5 introduz a banda sup-t de Montiel
   Olea e Plagborg-Moller (2019) e a Tabela 1 reporta o placar sob os dois
   criterios; a Tabela 2, que e onde a afirmacao central 18/18 vive, so traz
   asteriscos pontuais. Calcula-se aqui o valor critico sup-t para cada serie e
   cada regime, sobre a covariancia HAC conjunta entre os horizontes 0..12
   estimada na amostra comum, e recontam-se as celulas.

B. PANDEMIA x ASSIMETRIA. Em h = 0 as dez indicadoras de janela equivalem, por
   Salkever (1976), a excluir mar-dez/2020. Os DOIS maiores choques negativos da
   amostra (-5,434 em mar/2020 e -5,396 em abr/2020) estao nessa janela e no
   regime posterior, e a secao 4.7.9 mostra que sob a paridade a resposta a
   quedas e muito menor que a altas. Retirar as duas maiores quedas do regime em
   que a resposta a quedas e menor tem efeito de primeira ordem sobre um
   coeficiente estimado sob simetria. Estima-se aqui o coeficiente do regime
   posterior em quatro combinacoes: com e sem os dez meses, sob simetria e com o
   termo de assimetria ativo.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import lp_base_nova as L
import onda2_bandas_placebo as O2

SAIDA = L.SAIDA / "banca2"
SAIDA.mkdir(parents=True, exist_ok=True)


def _base(nome, b):
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    din = []
    for lag in range(1, L.LAGS + 1):
        din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
               [f"{c}_lag{lag}" for c in L.CONTROLES]
    return d, din


# ------------------------------------------------------------------ A
def uniforme():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]

    linhas = []
    for nome in L.VARIAVEIS:
        d, din = _base(nome, b)
        montado = []
        for h in range(0, L.H_MAX + 1):
            d["__y"] = L.y_acum(d, nome, h)
            regs = ["choque_pre", "choque_pos", "regime_pos"] + din + dum + pand(d, h)
            am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
            montado.append((h, regs, am))
        comuns = set.intersection(*[set(am["Data"]) for _, _, am in montado])
        for alvo in ["choque_pre", "choque_pos"]:
            gs = [O2.ajustar(am[am["Data"].isin(comuns)].reset_index(drop=True), regs, h, alvo)
                  for h, regs, am in montado]
            c_unif = O2.sup_t_plugin(O2.hac_conjunta_horizontes(gs, L.H_MAX + 1))
            for (h, regs, am), g in zip(montado, gs):
                # ponto e ep na amostra propria do horizonte (como na Tabela 2)
                gh = O2.ajustar(am.reset_index(drop=True), regs, h, alvo)
                t = abs(gh["valor"] / gh["ep"]) if gh["ep"] > 0 else np.nan
                linhas.append({
                    "variavel": nome, "regime": "anterior" if alvo == "choque_pre" else "posterior",
                    "h": h, "coef": gh["valor"], "ep": gh["ep"], "t": t,
                    "critico_pontual_10": 1.645, "critico_uniforme_10": c_unif,
                    "signif_pontual_10": bool(t > 1.645),
                    "signif_uniforme_10": bool(t > c_unif),
                })
        print(f"  uniforme {nome}")
    t = pd.DataFrame(linhas)
    t.to_excel(SAIDA / "supt_tabela2.xlsx", index=False)

    print("\n=== A. CRITERIO UNIFORME NA TABELA 2 ===")
    hs = [0, 1, 3, 6, 12]
    v = t[t.h.isin(hs)]
    for reg in ["anterior", "posterior"]:
        x = v[v.regime == reg]
        print(f"  regime {reg:9s}: {int(x.signif_pontual_10.sum()):2d}/30 pontual"
              f"  |  {int(x.signif_uniforme_10.sum()):2d}/30 uniforme")
    curto = v[v.h.isin([0, 1, 3])]
    for reg in ["anterior", "posterior"]:
        x = curto[curto.regime == reg]
        print(f"  h=0,1,3  {reg:9s}: {int(x.signif_pontual_10.sum()):2d}/18 pontual"
              f"  |  {int(x.signif_uniforme_10.sum()):2d}/18 uniforme")
    print("\n  valores criticos sup-t por serie (regime posterior):")
    for nome in L.VARIAVEIS:
        c = t[(t.variavel == nome) & (t.regime == "posterior")].critico_uniforme_10.iloc[0]
        print(f"    {nome:24s} {c:.3f}")
    return t


# ------------------------------------------------------------------ B
def pandemia_assimetria():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    ch = b["choque_petroleo"]
    b["choque_pre"] = ch * (1 - b["regime_pos"])
    b["choque_pos"] = ch * b["regime_pos"]
    alta = (ch > 0).astype(float)
    b["pos_alta"] = b["choque_pos"] * alta
    b["pos_baixa"] = b["choque_pos"] * (1 - alta)
    b["pre_alta"] = b["choque_pre"] * alta
    b["pre_baixa"] = b["choque_pre"] * (1 - alta)
    jan = set(pd.to_datetime(L.PANDEMIA))

    linhas = []
    for nome in ["ipca_geral", "diesel", "gasolina_consumidor", "gasolina_refinaria"]:
        d, din = _base(nome, b)
        for h in [0, 1]:
            for pand_modo in ["indicadoras", "sem 2020"]:
                for spec in ["simetrica", "assimetrica"]:
                    d["__y"] = L.y_acum(d, nome, h)
                    if spec == "simetrica":
                        cab = ["choque_pre", "choque_pos"]
                    else:
                        cab = ["pre_alta", "pre_baixa", "pos_alta", "pos_baixa"]
                    regs = cab + ["regime_pos"] + din + dum
                    if pand_modo == "indicadoras":
                        regs = regs + pand(d, h)
                    am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                    if pand_modo == "sem 2020":
                        am = am[~am["Data"].isin(jan)]
                    X = sm.add_constant(am[regs], has_constant="add")
                    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
                    r = sm.OLS(am["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
                    reg = {"variavel": nome, "h": h, "pandemia": pand_modo, "spec": spec,
                           "obs": int(r.nobs)}
                    for c in cab:
                        if c in r.params.index:
                            reg[c] = r.params[c]
                            reg["p_" + c] = r.pvalues[c]
                    if spec == "assimetrica" and "pos_alta" in r.params.index:
                        w = r.wald_test("pos_alta = pos_baixa", scalar=True)
                        reg["p_simetria_pos"] = float(w.pvalue)
                    linhas.append(reg)
        print(f"  pandemia x assimetria {nome}")
    t = pd.DataFrame(linhas)
    t.to_excel(SAIDA / "pandemia_assimetria.xlsx", index=False)

    print("\n=== B. PANDEMIA x ASSIMETRIA ===")
    for nome in ["ipca_geral", "diesel"]:
        print(f"\n  --- {nome} ---")
        x = t[t.variavel == nome]
        for h in [0, 1]:
            print(f"   h = {h}")
            for pm in ["indicadoras", "sem 2020"]:
                s = x[(x.h == h) & (x.pandemia == pm) & (x.spec == "simetrica")]
                a = x[(x.h == h) & (x.pandemia == pm) & (x.spec == "assimetrica")]
                if len(s) and len(a):
                    print(f"     {pm:12s} simetrico pos = {s.choque_pos.iloc[0]:+.3f}"
                          f"   |  alta = {a.pos_alta.iloc[0]:+.3f}"
                          f"  baixa = {a.pos_baixa.iloc[0]:+.3f}"
                          f"  p(sim) = {a.p_simetria_pos.iloc[0]:.3f}   N={int(s.obs.iloc[0])}")
    return t


if __name__ == "__main__":
    qual = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    if qual in ("tudo", "uniforme"):
        uniforme()
    if qual in ("tudo", "assimetria"):
        pandemia_assimetria()
