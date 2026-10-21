# -*- coding: utf-8 -*-
"""
TRES ESTIMADORES ALTERNATIVOS, MESMA ESPECIFICACAO DO MODELO PRINCIPAL

Todos usam exatamente: LAGS=3, corte mar/2017, controles so defasados, 11 sazonais
mensais, 10 indicadoras de pandemia de janela, amostra 2003-01 a 2025-09.

  A) QUANTILICA  repasse nos quantis 10, 25, 50, 75 e 90 da distribuicao condicional.
                 Responde: o repasse e o mesmo em meses de inflacao baixa e alta?
                 Inferencia por bootstrap de blocos moveis de 12 meses, porque o
                 erro-padrao padrao da QuantReg supoe iid e aqui nao vale.

  B) ROBUSTA     LAD (mediana) e M-estimador de Huber, rodados COM e SEM as
                 indicadoras de pandemia. A pergunta: o estimador robusto dispensa
                 as dummies? Se o coeficiente sem dummias bater com o MQO com
                 dummias, a escolha de quais meses tratar deixa de importar.

  C) VAR CHOLESKY  VAR por regime com as mesmas variaveis e as mesmas 3 defasagens,
                 identificacao recursiva com o petroleo em primeiro. Resposta
                 acumulada a um choque de um desvio-padrao, comparada ao LP.

Uso:  python onda5_alternativos.py quantilica | robusta | var | tudo
"""
import sys, warnings, time
import numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L

SAIDA = Path(__file__).resolve().parent.parent / "resultados" / "modelo_11_alternativos"
SAIDA.mkdir(parents=True, exist_ok=True)

QUANTIS = [0.10, 0.25, 0.50, 0.75, 0.90]
HS = L.HORIZONTES                      # [0, 1, 2, 3, 6, 12]
BLOCO, BREP = 12, 299
FOCO = ["ipca_geral", "ipca_transportes"]
H_FOCO = [0, 1, 3]
rng = np.random.default_rng(20260905)


# ══════════════════════════════════════════════ montagem comum
def monta(b, dum, nome, h):
    """Replica exatamente a matriz de regressores do modelo principal."""
    d = b.copy()
    pand = L.controles_pandemia(d, h, modo="janela")
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    regs = ["choque_pre", "choque_pos", "regime_pos"]
    for lag in range(1, L.LAGS + 1):
        regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"]
        regs += [f"{c}_lag{lag}" for c in L.CONTROLES]
    regs += list(dum) + pand
    d["__y"] = L.y_acum(d, nome, h)
    am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(am[regs].astype(float), has_constant="add")
    return am["__y"].astype(float).values, X.values, list(X.columns)


def base():
    b, dum = L.carregar("corrigida")
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
    return b, dum


def blocos_moveis(n, tam, gerador):
    """Indices de um bootstrap de blocos moveis de comprimento `tam`."""
    k = int(np.ceil(n / tam))
    inicios = gerador.integers(0, n - tam + 1, size=k)
    idx = np.concatenate([np.arange(i, i + tam) for i in inicios])
    return idx[:n]


# ══════════════════════════════════════════════ A) QUANTILICA
def quantilica():
    b, dum = base()
    t0 = time.time()
    pontos, infer = [], []
    for nome in L.VARIAVEIS:
        for h in HS:
            y, X, cols = monta(b, dum, nome, h)
            i_pre, i_pos = cols.index("choque_pre"), cols.index("choque_pos")
            for q in QUANTIS:
                try:
                    r = QuantReg(y, X).fit(q=q, max_iter=4000)
                except Exception:
                    continue
                pontos.append({"variavel": nome, "h": h, "quantil": q,
                               "coef_pre": r.params[i_pre], "coef_pos": r.params[i_pos],
                               "diferenca": r.params[i_pos] - r.params[i_pre],
                               "n": len(y)})
                if nome in FOCO and h in H_FOCO:
                    bs = np.empty((BREP, 3)); bs[:] = np.nan
                    for rr in range(BREP):
                        ii = blocos_moveis(len(y), BLOCO, rng)
                        try:
                            rb = QuantReg(y[ii], X[ii]).fit(q=q, max_iter=1500)
                            bs[rr] = [rb.params[i_pre], rb.params[i_pos],
                                      rb.params[i_pos] - rb.params[i_pre]]
                        except Exception:
                            pass
                    bs = bs[~np.isnan(bs).any(axis=1)]
                    ep = bs.std(axis=0, ddof=1)
                    dif = r.params[i_pos] - r.params[i_pre]
                    # p bilateral do tipo percentil, centrado na estimativa
                    p_dif = float(2 * min(np.mean(bs[:, 2] - dif <= -dif),
                                          np.mean(bs[:, 2] - dif >= -dif)))
                    infer.append({"variavel": nome, "h": h, "quantil": q,
                                  "coef_pre": r.params[i_pre], "ep_pre": ep[0],
                                  "coef_pos": r.params[i_pos], "ep_pos": ep[1],
                                  "diferenca": dif, "ep_dif": ep[2],
                                  "p_diferenca": min(p_dif, 1.0), "replicas": len(bs)})
        print("   quantilica: %-22s  %.0fs" % (nome, time.time() - t0))
    P, I = pd.DataFrame(pontos), pd.DataFrame(infer)
    with pd.ExcelWriter(SAIDA / "quantilica.xlsx") as w:
        P.to_excel(w, "pontos", index=False)
        I.to_excel(w, "inferencia_bootstrap", index=False)
    return P, I


# ══════════════════════════════════════════════ B) ROBUSTA / LAD
def robusta():
    b, dum = base()
    lin = []
    for nome in L.VARIAVEIS:
        for h in HS:
            y, X, cols = monta(b, dum, nome, h)
            i_pre, i_pos = cols.index("choque_pre"), cols.index("choque_pos")
            npand = [i for i, c in enumerate(cols) if c.startswith("pand_")]
            manter = [i for i in range(len(cols)) if i not in npand]
            Xs = X[:, manter]
            j_pre = manter.index(i_pre); j_pos = manter.index(i_pos)
            casos = [("MQO", "com dummies", sm.OLS(y, X), i_pre, i_pos),
                     ("MQO", "sem dummies", sm.OLS(y, Xs), j_pre, j_pos),
                     ("LAD", "com dummies", QuantReg(y, X), i_pre, i_pos),
                     ("LAD", "sem dummies", QuantReg(y, Xs), j_pre, j_pos),
                     ("Huber", "com dummies", RLM(y, X, M=HuberT()), i_pre, i_pos),
                     ("Huber", "sem dummies", RLM(y, Xs, M=HuberT()), j_pre, j_pos)]
            for est, dumm, mod, a, c in casos:
                try:
                    r = mod.fit(q=0.5, max_iter=4000) if est == "LAD" else (
                        mod.fit(cov_type="HAC", cov_kwds=L.hac_kw(h)) if est == "MQO"
                        else mod.fit())
                except Exception:
                    continue
                lin.append({"estimador": est, "pandemia": dumm, "variavel": nome, "h": h,
                            "coef_pre": r.params[a], "coef_pos": r.params[c],
                            "diferenca": r.params[c] - r.params[a],
                            "p_pos": (r.pvalues[c] if hasattr(r, "pvalues") else np.nan),
                            "n": len(y)})
        print("   robusta: %s" % nome)
    R = pd.DataFrame(lin)
    R.to_excel(SAIDA / "robusta_lad.xlsx", index=False)
    return R


# ══════════════════════════════════════════════ C) VAR CHOLESKY
def var_cholesky():
    b, _ = base()
    lin, irfs = [], []
    for nome in L.VARIAVEIS:
        for rg, rot in [(0, "pre"), (1, "pos")]:
            d = b[b.regime_pos == rg].copy()
            cols = ["choque_petroleo", "dlog_cambio", "dlog_atividade",
                    f"dlog_{nome}", "selic", "expectativa"]
            m = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(m) < 40:
                continue
            try:
                res = VAR(m.values).fit(L.LAGS, trend="c")
                ir = res.irf(L.H_MAX)
                # ordem recursiva com o petroleo em primeiro: choque na eq. 0
                cum = ir.orth_cum_effects[:, cols.index(f"dlog_{nome}"), 0]
                pt = ir.orth_irfs[:, cols.index(f"dlog_{nome}"), 0]
            except Exception as e:
                print("      falhou %s %s: %s" % (nome, rot, e)); continue
            for h in range(L.H_MAX + 1):
                irfs.append({"variavel": nome, "regime": rot, "h": h,
                             "acumulada": cum[h], "pontual": pt[h], "n": len(m)})
            for h in HS:
                lin.append({"variavel": nome, "regime": rot, "h": h,
                            "var_acumulada": cum[h], "n_obs": len(m)})
        print("   VAR: %s" % nome)
    V = pd.DataFrame(lin); I = pd.DataFrame(irfs)
    W = V.pivot_table(index=["variavel", "h"], columns="regime",
                      values="var_acumulada").reset_index()
    W["diferenca"] = W["pos"] - W["pre"]
    with pd.ExcelWriter(SAIDA / "var_cholesky.xlsx") as w:
        I.to_excel(w, "irf_completa", index=False)
        W.to_excel(w, "comparacao", index=False)
    return W, I


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    if cmd in ("quantilica", "tudo"):
        print("== A) QUANTILICA"); quantilica()
    if cmd in ("robusta", "tudo"):
        print("== B) ROBUSTA / LAD"); robusta()
    if cmd in ("var", "tudo"):
        print("== C) VAR CHOLESKY"); var_cholesky()
    print("saida em", SAIDA)
