# -*- coding: utf-8 -*-
"""
COMPARAÇÃO DOS RESULTADOS CONFORME O TRATAMENTO DE 2020 (criado em 12/10/2026)

Quatro tratamentos, todos sobre a especificação por regime do TCC (IPCA Geral):

  indicadoras   dez indicadoras de pandemia, mar-dez/2020 (a adotada no TCC)
  oxford        índice de rigor do OxCGRT no lugar das indicadoras (seção 4.7.2)
  sem_controle  2020 na amostra, sem controle algum de pandemia
  sem_2020      excluídas as observações cuja janela, do mês do choque ao mês da
                resposta, toca qualquer mês de 2020; sem controle (as indicadoras
                ficariam todas nulas, então com ou sem elas o resultado é o mesmo)

Para cada um: coeficiente acumulado no impacto e em três meses, antes e depois de
mar/2017, com o p do teste de Wald, e o contrafactual por ano civil (2020, 2021 e 2022),
com os coeficientes não acumulados de cada tratamento, como em contrafactual_anual.py.

Conferências: o tratamento adotado reproduz a Tabela 2 (0,005 e 0,176; Wald 0,001), o
Oxford reproduz 0,088 (Wald 0,016), os coeficientes não acumulados reproduzem
onda2_6 e o contrafactual reproduz 1,72 e 1,15 (seção 4.7.5).

Não altera nenhum resultado do TCC. Saída: resultados/comparacao_2020_pandemia.xlsx
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lp_base_nova as L
import teste_oxford_covid as OX
import onda2_6_resposta_nao_acumulada as N

SAIDA = AQUI.parent / "resultados"
NOME = "ipca_geral"
MODOS = ["indicadoras", "oxford", "sem_controle", "sem_2020"]
ROTULO = {"indicadoras": "Indicadoras de pandemia (TCC)",
          "oxford": "Índice de rigor de Oxford",
          "sem_controle": "Sem controle de pandemia",
          "sem_2020": "Sem o ano de 2020 na amostra"}
M2020 = (2020 * 12 + 1, 2020 * 12 + 12)

b, dum, pand, _ = OX.prepara()
d0, DIN = N._base(NOME, b)


def toca_2020(d, h):
    """Verdadeiro para as observações cuja janela [t, t+h] contém algum mês de 2020."""
    return (d["mi"] <= M2020[1]) & (d["mi"] + h >= M2020[0])


def estima(y, d, h, modo, ctrl_ind):
    """MQO com HAC, na especificação por regime; ctrl_ind são as indicadoras do horizonte."""
    d = d.copy()
    d["__y"] = y
    if modo == "indicadoras":
        ctrl = ctrl_ind
    elif modo == "oxford":
        ctrl = ["__ox"]
    else:
        ctrl = []
    regs = ["choque_pre", "choque_pos", "regime_pos"] + DIN + dum + ctrl
    am = d[["Data", "mi", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
    if modo == "sem_2020":
        am = am[~toca_2020(am, h)]
    X = sm.add_constant(am[regs], has_constant="add")
    r = sm.OLS(am["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
    w = r.wald_test("choque_pre = choque_pos", scalar=True)
    return {"pre": r.params["choque_pre"], "pos": r.params["choque_pos"],
            "p_pos": r.pvalues["choque_pos"], "wald": float(w.pvalue), "n": int(r.nobs)}


def acumulado(h, modo):
    d = d0.copy()
    ctrl_ind = L.controles_pandemia(d, h, "janela") if modo == "indicadoras" else []
    # Oxford como na seção 4.7.2 e em banca2_pandemia_assimetria.py: rigor do mês do choque
    d["__ox"] = d["si_covid"]
    return estima(L.y_acum(d, NOME, h), d, h, modo, ctrl_ind)


def nao_acumulado(h, modo):
    d = d0.copy()
    ctrl_ind = N.pand_mes(d, h) if modo == "indicadoras" else []
    # a dependente é a inflação do mês t+h, então o controle é o rigor do mês t+h
    d["__ox"] = d["si_covid"].shift(-h)
    return estima(N.y_mes(d, NOME, h), d, h, modo, ctrl_ind)


ch = b.set_index("Data")["choque_petroleo"]
dl = b.set_index("Data")["dlog_ipca_geral"]


def contrafactual(coef, ano):
    ini, fim = pd.Timestamp(f"{ano}-01-01"), pd.Timestamp(f"{ano}-12-01")
    meses = pd.date_range(ini, fim, freq="MS")
    tot = {}
    for alvo in ("pos", "pre"):
        s = 0.0
        for t in meses:
            e = ch.get(t, np.nan)
            if pd.isna(e):
                continue
            for h, cf in coef[alvo].items():
                if ini <= t + pd.DateOffset(months=h) <= fim:
                    s += cf * e
        tot[alvo] = s
    ipca_log = float(dl.loc[meses].sum())
    return tot["pos"], tot["pre"], ipca_log


linhas, contra = [], []
coef_na = {}
for modo in MODOS:
    a0, a1, a3 = acumulado(0, modo), acumulado(1, modo), acumulado(3, modo)
    linhas.append({"tratamento": ROTULO[modo],
                   "impacto_antes": a0["pre"], "impacto_depois": a0["pos"], "p_depois_impacto": a0["p_pos"],
                   "wald_impacto": a0["wald"], "um_mes_depois": a1["pos"], "wald_um_mes": a1["wald"],
                   "tres_meses_antes": a3["pre"], "tres_meses_depois": a3["pos"], "wald_tres_meses": a3["wald"],
                   "obs_impacto": a0["n"]})
    na = {h: nao_acumulado(h, modo) for h in range(L.H_MAX + 1)}
    coef_na[modo] = {"pos": {h: na[h]["pos"] for h in na}, "pre": {h: na[h]["pre"] for h in na}}
    for ano in (2020, 2021, 2022):
        pos, pre, ipl = contrafactual(coef_na[modo], ano)
        contra.append({"tratamento": ROTULO[modo], "ano": ano, "com_paridade": pos, "regra_anterior": pre,
                       "diferenca": pos - pre, "ipca_ano_pontos_log": ipl,
                       "pct_da_inflacao_do_ano": 100 * (pos - pre) / ipl})
    print("  ", ROTULO[modo])

T = pd.DataFrame(linhas)
C = pd.DataFrame(contra)

# ------------------------------------------------------------------ conferências
t = T.set_index("tratamento")
adot, ox = t.loc[ROTULO["indicadoras"]], t.loc[ROTULO["oxford"]]
assert round(adot.impacto_antes, 3) == 0.005 and round(adot.impacto_depois, 3) == 0.176
assert round(adot.wald_impacto, 3) == 0.001 and round(adot.tres_meses_depois, 3) == 0.314
assert round(ox.impacto_depois, 3) == 0.088 and round(ox.wald_impacto, 3) == 0.016
r6 = pd.read_excel(SAIDA / "onda2_6_resposta_nao_acumulada.xlsx", sheet_name="por_regime")
r6 = r6[r6.variavel == NOME]
for _, x in r6.iterrows():
    alvo = "pos" if x.alvo == "choque_pos" else "pre"
    assert abs(coef_na["indicadoras"][alvo][int(x.h)] - x.coeficiente) < 1e-10
c = C.set_index(["tratamento", "ano"])
assert round(c.loc[(ROTULO["indicadoras"], 2021), "diferenca"], 2) == 1.72
assert round(c.loc[(ROTULO["indicadoras"], 2022), "diferenca"], 2) == 1.15

with pd.ExcelWriter(SAIDA / "comparacao_2020_pandemia.xlsx") as w:
    T.to_excel(w, sheet_name="coeficientes", index=False)
    C.to_excel(w, sheet_name="contrafactual", index=False)
print(T.round(3).to_string(index=False))
print(C.round(2).to_string(index=False))
print("salvo", SAIDA / "comparacao_2020_pandemia.xlsx")
