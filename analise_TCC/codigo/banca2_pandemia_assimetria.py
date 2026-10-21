# -*- coding: utf-8 -*-
"""(Banca 2, achado B) A faixa 0,088-0,176 e efeito de assimetria?

HIPOTESE A TESTAR. Em h = 0 as dez indicadoras de janela saturam e equivalem, por
Salkever (1976), a excluir mar-dez/2020. Os DOIS maiores choques negativos da
amostra -- -5,434 em mar/2020 e -5,396 em abr/2020 -- estao nessa janela e no
regime posterior. A secao 4.7.9 mostra que, sob a paridade, a resposta a quedas e
muito menor que a altas nos combustiveis. Se isso valesse tambem para o indice
cheio, retirar as duas maiores quedas do regime posterior elevaria um coeficiente
estimado sob restricao de simetria, e a faixa 0,088-0,176 seria consequencia de um
mecanismo economico, e nao de uma escolha tecnica.

DESENHO. O contraste relevante NAO e "indicadoras contra excluir": em h = 0 os
dois excluem os mesmos dez meses e o coeficiente e identico por construcao. O
contraste e contra as especificacoes que DEVOLVEM aqueles meses a estimacao:

  janela      dez indicadoras de janela (adotada)      -> 10 meses fora, em h = 0
  rigor       indice de rigor do OxCGRT, continuo      -> 10 meses DENTRO
  sem_ctrl    nenhum controle de pandemia              -> 10 meses DENTRO

Cada uma e estimada sob simetria e com o choque partido em alta e queda. Se a
assimetria explicar a faixa, o coeficiente de ALTA deve ser estavel entre os tres
modos, e a variacao deve concentrar-se no de QUEDA e no simetrico.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import lp_base_nova as L
import teste_oxford_covid as OX

SAIDA = L.SAIDA / "banca2"
SAIDA.mkdir(parents=True, exist_ok=True)
SERIES = ["ipca_geral", "ipca_transportes", "diesel", "gasolina_consumidor"]


def rodar():
    b, dum, pand, _ = OX.prepara()
    ch = b["choque_petroleo"]
    alta = (ch > 0).astype(float)
    for pre, mask in [("pre", 1 - b["regime_pos"]), ("pos", b["regime_pos"])]:
        b[f"{pre}_alta"] = ch * mask * alta
        b[f"{pre}_baixa"] = ch * mask * (1 - alta)

    linhas = []
    for nome in SERIES:
        d = b.copy()
        for lag in range(1, L.LAGS + 1):
            d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
        din = []
        for lag in range(1, L.LAGS + 1):
            din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                   [f"{c}_lag{lag}" for c in L.CONTROLES]

        for h in [0, 1]:
            d["__y"] = L.y_acum(d, nome, h)
            for modo in ["janela", "rigor", "sem_ctrl"]:
                if modo == "janela":
                    ctrl = pand(d, h)
                elif modo == "rigor":
                    ctrl = ["si_covid"]
                else:
                    ctrl = []
                for spec, cab in [("simetrica", ["choque_pre", "choque_pos"]),
                                  ("assimetrica", ["pre_alta", "pre_baixa",
                                                   "pos_alta", "pos_baixa"])]:
                    regs = cab + ["regime_pos"] + din + dum + list(ctrl)
                    am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                    X = sm.add_constant(am[regs], has_constant="add")
                    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
                    r = sm.OLS(am["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
                    reg = {"variavel": nome, "h": h, "modo": modo, "spec": spec,
                           "obs": int(r.nobs), "n_par": int(X.shape[1])}
                    for c in cab:
                        if c in r.params.index:
                            reg[c] = float(r.params[c])
                            reg["p_" + c] = float(r.pvalues[c])
                    if spec == "assimetrica" and {"pos_alta", "pos_baixa"} <= set(r.params.index):
                        reg["p_simetria_pos"] = float(
                            r.wald_test("pos_alta = pos_baixa", scalar=True).pvalue)
                    linhas.append(reg)
        print(f"  {nome}")

    t = pd.DataFrame(linhas)
    t.to_excel(SAIDA / "pandemia_assimetria.xlsx", index=False)

    print("\n" + "=" * 74)
    print("CONTRASTE: os dez meses de 2020 FORA (janela) contra DENTRO (rigor, sem_ctrl)")
    print("=" * 74)
    for nome in SERIES:
        print(f"\n--- {nome} ---")
        for h in [0, 1]:
            print(f"  h = {h}")
            print(f"    {'modo':10s} {'simétrico':>10s} | {'alta':>8s} {'baixa':>8s} "
                  f"{'p(sim)':>8s} {'N':>5s}")
            for modo in ["janela", "rigor", "sem_ctrl"]:
                s = t[(t.variavel == nome) & (t.h == h) & (t.modo == modo) &
                      (t.spec == "simetrica")]
                a = t[(t.variavel == nome) & (t.h == h) & (t.modo == modo) &
                      (t.spec == "assimetrica")]
                if len(s) and len(a):
                    print(f"    {modo:10s} {s.choque_pos.iloc[0]:10.3f} | "
                          f"{a.pos_alta.iloc[0]:8.3f} {a.pos_baixa.iloc[0]:8.3f} "
                          f"{a.p_simetria_pos.iloc[0]:8.3f} {int(s.obs.iloc[0]):5d}")
    return t


if __name__ == "__main__":
    rodar()
