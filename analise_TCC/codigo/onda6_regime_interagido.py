# -*- coding: utf-8 -*-
"""
ESPECIFICACAO TOTALMENTE INTERAGIDA COM O REGIME (01/10/2026)

Motivo. A especificacao principal (onda3_3, estimar) deixa variar entre regimes apenas o
coeficiente do choque e o intercepto; defasagens proprias, defasagens do choque, controles,
sazonalidade e indicadoras de pandemia sao comuns aos dois periodos. Para series cujo processo
de formacao de preco muda com o regime (a refinaria passa de preco administrado a preco
indexado), impor a mesma dinamica nos dois periodos e uma restricao forte. Aqui todos os
regressores sao interagidos com o regime:

  y(t,h) = a + d*D(t) + b_pre*choque(t)*(1-D) + b_pos*choque(t)*D
           + sum_k [ c_k,pre * X_k(t)*(1-D) + c_k,pos * X_k(t)*D ] + u(t,h)

o que equivale a estimar cada regime separadamente, mas numa unica regressao, de modo que o
teste b_pre = b_pos usa a covariancia HAC conjunta. Colunas identicamente nulas (indicadoras de
pandemia no regime anterior) sao descartadas.

Tambem se testa a restricao da especificacao principal: H0: c_k,pre = c_k,pos para todos os k
(Wald HAC). Rejeitar indica que a dinamica comum nao e suportada pelos dados.

Checagem: a versao restrita aqui montada reproduz os coeficientes de
resultados/onda3_3_regimes_data_escolhida.xlsx (corte escolhido).

Uso: python onda6_regime_interagido.py
"""
import warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
warnings.filterwarnings("ignore")
import lp_base_nova as L

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"
CORTE = pd.Timestamp(L.DATA_CORTE_FINAL)


def sem_colunas_vazias(X):
    return X.loc[:, (X.std() > 0) | (X.columns == "const")]


def estimar():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["regime_pos"] = L.regime(b.Data, CORTE)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
    linhas = []
    for nome in L.VARIAVEIS:
        d = b.copy()
        for lag in range(1, L.LAGS + 1):
            d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
        comuns = []
        for lag in range(1, L.LAGS + 1):
            comuns += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in L.CONTROLES]
        for h in range(0, L.H_MAX + 1):
            d["__y"] = L.y_acum(d, nome, h)
            outros = comuns + dum + pand(d, h)
            am = d[["__y", "regime_pos", "choque_pre", "choque_pos"] + outros].replace(
                [np.inf, -np.inf], np.nan).dropna()
            D = am["regime_pos"].values
            # ---- restrita (especificacao principal)
            Xr = sem_colunas_vazias(sm.add_constant(am[["choque_pre", "choque_pos", "regime_pos"] + outros], has_constant="add"))
            rr = sm.OLS(am["__y"], Xr).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
            # ---- totalmente interagida
            Xi = pd.DataFrame({"const": 1.0, "regime_pos": D, "choque_pre": am["choque_pre"], "choque_pos": am["choque_pos"]},
                              index=am.index)
            for v in outros:
                Xi[f"{v}__pre"] = am[v].values * (1 - D)
                Xi[f"{v}__pos"] = am[v].values * D
            Xi = sem_colunas_vazias(Xi)
            ri = sm.OLS(am["__y"], Xi).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
            w = ri.wald_test("choque_pre = choque_pos", scalar=True)
            pares = [v for v in outros if f"{v}__pre" in Xi.columns and f"{v}__pos" in Xi.columns]
            R = np.zeros((len(pares), Xi.shape[1]))
            for k, v in enumerate(pares):
                R[k, Xi.columns.get_loc(f"{v}__pre")] = 1.0
                R[k, Xi.columns.get_loc(f"{v}__pos")] = -1.0
            wr = ri.wald_test(R, scalar=True)
            V = ri.cov_params()
            ep_dif = float(np.sqrt(max(V.loc["choque_pre", "choque_pre"] + V.loc["choque_pos", "choque_pos"]
                                       - 2 * V.loc["choque_pre", "choque_pos"], 0.0)))
            n_pos = int(D.sum())
            linhas.append({"variavel": nome, "h": h,
                           "restrita_coef_pre": rr.params["choque_pre"], "restrita_p_pre": rr.pvalues["choque_pre"],
                           "restrita_coef_pos": rr.params["choque_pos"], "restrita_p_pos": rr.pvalues["choque_pos"],
                           "coef_pre": ri.params["choque_pre"], "ep_pre": ri.bse["choque_pre"], "p_pre": ri.pvalues["choque_pre"],
                           "coef_pos": ri.params["choque_pos"], "ep_pos": ri.bse["choque_pos"], "p_pos": ri.pvalues["choque_pos"],
                           "diferenca": ri.params["choque_pos"] - ri.params["choque_pre"], "ep_diferenca": ep_dif,
                           "wald_p_igualdade": float(w.pvalue),
                           "restricoes_dinamica_comum": len(pares), "wald_p_dinamica_comum": float(wr.pvalue),
                           "obs": int(ri.nobs), "obs_pos": n_pos, "obs_pre": int(ri.nobs) - n_pos,
                           "parametros_restrita": Xr.shape[1], "parametros_interagida": Xi.shape[1],
                           "gl_pos": n_pos - int(sum(c.endswith("__pos") or c in ("choque_pos", "regime_pos") for c in Xi.columns))})
        print(f"  {nome}")
    t = pd.DataFrame(linhas)
    # checagem contra a especificacao principal
    ref = pd.read_excel(SAIDA / "onda3_3_regimes_data_escolhida.xlsx")
    ref = ref[ref.corte == "escolhida"].set_index(["variavel", "h"])
    chk = t.set_index(["variavel", "h"])
    dif = max((chk.restrita_coef_pre - ref.coef_pre).abs().max(), (chk.restrita_coef_pos - ref.coef_pos).abs().max())
    print(f"  checagem: versao restrita x onda3_3 (dif max nos coeficientes) = {dif:.2e}")
    if dif > 1e-8:
        raise ValueError("a versao restrita nao reproduz a especificacao principal")
    t.to_excel(SAIDA / "onda6_regime_interagido.xlsx", index=False)
    return t


def resumo(t):
    H3, H5 = [0, 1, 3], [0, 1, 3, 6, 12]
    x3, x5 = t[t.h.isin(H3)], t[t.h.isin(H5)]
    print("\n  CONTAGENS (horizontes 0, 1 e 3; 6 series = 18 coeficientes)")
    for rot, p in [("restrita", "restrita_p"), ("interagida", "p")]:
        print(f"   {rot:10s}: anterior signif. 10% {int((x3[p + '_pre'] < .10).sum())}/18 | "
              f"posterior signif. 10% {int((x3[p + '_pos'] < .10).sum())}/18, a 1% {int((x3[p + '_pos'] < .01).sum())}")
    print(f"   interagida: Wald pre = pos a 5% em {int((x3.wald_p_igualdade < .05).sum())}/18 | "
          f"5 horizontes, posterior signif. {int((x5.p_pos < .10).sum())}/30")
    print(f"   restricao de dinamica comum rejeitada a 5% em {int((t.wald_p_dinamica_comum < .05).sum())} de {len(t)} equacoes")
    print("\n  POR SERIE, h = 0 / 1 / 3")
    for nome, g in x3.groupby("variavel", sort=False):
        g = g.set_index("h")
        print(f"   {nome:20s} " + " | ".join(
            f"h{h}: ant {g.at[h,'coef_pre']:+.3f} (p {g.at[h,'p_pre']:.3f}) pos {g.at[h,'coef_pos']:+.3f} (p {g.at[h,'p_pos']:.3f}) W {g.at[h,'wald_p_igualdade']:.3f} DC {g.at[h,'wald_p_dinamica_comum']:.3f}"
            for h in H3))


if __name__ == "__main__":
    resumo(estimar())
