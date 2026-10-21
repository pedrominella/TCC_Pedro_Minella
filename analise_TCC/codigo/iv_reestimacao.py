# -*- coding: utf-8 -*-
"""
REESTIMAÇÃO DOS INSTRUMENTOS, COM A ESPECIFICAÇÃO DO MODELO PRINCIPAL (14/10/2026)

Pedido do autor: rodar de novo os modelos com variável instrumental e ver o teste F, escolhendo a
melhor defasagem, com bootstrap e erro-padrão robusto de Newey-West.

Instrumentos e variável instrumentada
  surpresa_kanzig   surpresa de oferta de Känzig (2021), externa      -> choque do petróleo
  news_kanzig       choque estrutural de Känzig (referência)          -> choque do petróleo
  margem_gasolina   margem de refino da gasolina no Golfo, ortogonal  -> gasolina ao consumidor
  margem_diesel     margem de refino do diesel no Golfo, ortogonal    -> diesel
  golfo_cheio       variação da gasolina no Golfo, em dólar           -> gasolina ao consumidor
  golfo_com_brent   idem, com o choque do Brent como controle         -> gasolina ao consumidor

Primeiro estágio (h = 0, e h = 0 a 6 como faixa)
  x(t,h) = a + g·z(t) + Σ(i=1..p)[defasagens de x, do choque, de z e dos controles X] + sazonais
           + indicadoras de pandemia da janela de h + u
  com x(t,h) = choque do petróleo (instrumentos de Känzig) ou a variação acumulada do combustível de
  t-1 a t+h (margens e Golfo). Por regime: g_pré·z·1{t<τ} + g_pós·z·1{t≥τ} + δ·1{t≥τ}, com a dinâmica
  comum, como no modelo principal (τ = março de 2017).

Escolhas
  defasagens   p de 1 a 12 por AIC e BIC na amostra comum (12 primeiras observações descartadas);
               usa-se p_AIC + 1, o aumento de defasagem do modelo principal, e reporta-se também p = 3
  erro-padrão  Newey-West, núcleo de Bartlett, banda automática floor(4(T/100)^(2/9)), nunca menor
               que h+1, com correção T/(T-K); com um instrumento, o F robusto é o F efetivo de
               Montiel Olea e Pflueger
  bootstrap    selvagem em blocos de 12 meses, pesos de Rademacher, 999 réplicas: p-valor sob a nula
               (resíduos restritos) e intervalo percentil de 90% do F (resíduos irrestritos)
  forma        regressão do IPCA Geral acumulado sobre o instrumento, mesmos controles: é o teste de
  reduzida     Anderson-Rubin de efeito nulo, robusto a instrumento fraco

Conferência: o mesmo código, com a escolha antiga (p = 3, banda 1, subamostras separadas pelo anúncio
de outubro de 2016), reproduz os F da Tabela G.1 (5,45; 1,48; 0,25; 0,91).

Saída: resultados/iv_reestimacao.xlsx
Uso: python iv_reestimacao.py
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lp_base_nova as L

RAIZ = AQUI.parents[1]
DADOS = RAIZ / "base_de_dados"
SAIDA = AQUI.parent / "resultados" / "iv_reestimacao.xlsx"
B, BLOCO, PMAX, SEMENTE = 999, 12, 12, 20261014
CORTE = L.DATA_CORTE_FINAL
GAL = 42.0
EIA = {"gasolina": DADOS / "EER_EPMRU_PF4_RGC_DPGm.xls", "diesel": DADOS / "EER_EPD2DXL0_PF4_RGC_DPGm.xls"}


# ------------------------------------------------------------------ dados
def serie_eia(arq):
    d = pd.read_excel(arq, sheet_name="Data 1", skiprows=2)
    d.columns = ["Data", "usd_gal"]
    d = d.dropna()
    d["Data"] = pd.to_datetime(d["Data"]).dt.to_period("M").dt.to_timestamp()
    d["usd_bbl"] = pd.to_numeric(d["usd_gal"], errors="coerce") * GAL
    return d[["Data", "usd_bbl"]]


def com_golfo(b):
    dlb = 100.0 * np.log(pd.to_numeric(b["Preco_Barril"])).diff()
    for nome, arq in EIA.items():
        b = b.merge(serie_eia(arq).rename(columns={"usd_bbl": f"bbl_{nome}"}), on="Data", how="left")
        dlp = 100.0 * np.log(b[f"bbl_{nome}"]).diff()
        m = pd.concat([dlp, dlb], axis=1).dropna()
        r = sm.OLS(m.iloc[:, 0], sm.add_constant(m.iloc[:, 1])).fit()
        b[f"margem_{nome}"] = np.nan
        b.loc[m.index, f"margem_{nome}"] = r.resid.values
        b[f"margem_{nome}"] = (b[f"margem_{nome}"] - b[f"margem_{nome}"].mean()) / b[f"margem_{nome}"].std()
        if nome == "gasolina":
            b["golfo"] = (dlp - dlp.mean()) / dlp.std()
    return b


def bases():
    bk, dum, pand = L.carregar_pandemia("corrigida", L.DATA_FIM_IV, com_kanzig=True)
    bm, _, _ = L.carregar_pandemia("corrigida")
    bm = com_golfo(bm)
    for b in (bk, bm):
        b["regime_pos"] = L.regime(b.Data, CORTE)
        # choque do Brent com coeficiente próprio em cada regime, como no modelo principal
        b["choque_pre_c"] = b["choque_petroleo"] * (1 - b["regime_pos"])
        b["choque_pos_c"] = b["choque_petroleo"] * b["regime_pos"]
    return bk, bm, dum, pand


CASOS = {
    # rótulo: (base, instrumento, variável instrumentada, defasagens próprias, controles contemporâneos)
    "surpresa_kanzig": ("k", "instrumento_surpresa", "choque", "dlog_ipca_geral", []),
    "news_kanzig":     ("k", "instrumento_kanzig", "choque", "dlog_ipca_geral", []),
    "margem_gasolina": ("m", "margem_gasolina", "gasolina_consumidor", "dlog_gasolina_consumidor", ["choque_petroleo"]),
    "margem_diesel":   ("m", "margem_diesel", "diesel", "dlog_diesel", ["choque_petroleo"]),
    "golfo_cheio":     ("m", "golfo", "gasolina_consumidor", "dlog_gasolina_consumidor", []),
    "golfo_com_brent": ("m", "golfo", "gasolina_consumidor", "dlog_gasolina_consumidor", ["choque_petroleo"]),
    # variantes com o choque contemporâneo por regime: a margem só é ortogonal ao Brent na amostra
    # inteira (correlação de -0,18 depois e +0,32 antes de mar/2017, no diesel), e o efeito do
    # petróleo muda entre regimes, que é o resultado do trabalho
    "margem_gasolina_choque_regime": ("m", "margem_gasolina", "gasolina_consumidor", "dlog_gasolina_consumidor",
                                      ["choque_pre_c", "choque_pos_c"]),
    "margem_diesel_choque_regime": ("m", "margem_diesel", "diesel", "dlog_diesel", ["choque_pre_c", "choque_pos_c"]),
    "golfo_choque_regime": ("m", "golfo", "gasolina_consumidor", "dlog_gasolina_consumidor",
                            ["choque_pre_c", "choque_pos_c"]),
}


# ------------------------------------------------------------------ álgebra
def hac_var(c, u, banda, T, K):
    v = c * u
    s = float(v @ v)
    for l in range(1, banda + 1):
        s += 2.0 * (1.0 - l / (banda + 1)) * float(v[l:] @ v[:-l])
    return s * T / (T - K)


def banda_nw(T, h):
    return max(int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0))), h + 1)


def pesos_blocos(T, rng):
    nb = int(np.ceil(T / BLOCO))
    return np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:T]


def montar(b, dum, pand, caso, p, h, regime, ini=PMAX, lags_z=True):
    _, z, x, propria, contemp = CASOS[caso]
    d = b.copy()
    d["__x"] = d["choque_petroleo"] if x == "choque" else L.y_acum(d, x, h)
    d["__y"] = L.y_acum(d, "ipca_geral", h)
    lagvars = [propria, "choque_petroleo"] + ([z] if lags_z else []) + list(L.CONTROLES)
    cols = []
    for v in dict.fromkeys(lagvars):
        for l in range(1, p + 1):
            d[f"{v}__L{l}"] = d[v].shift(l)
            cols.append(f"{v}__L{l}")
    exog = cols + contemp + dum + pand(d, h)
    if regime:
        d["z_pre"] = d[z] * (1 - d["regime_pos"]); d["z_pos"] = d[z] * d["regime_pos"]
        inst = ["z_pre", "z_pos"]; exog = exog + ["regime_pos"]
    else:
        d["z_all"] = d[z]; inst = ["z_all"]
    am = d.iloc[ini:][["Data", "__x", "__y"] + inst + exog].replace([np.inf, -np.inf], np.nan).dropna()
    return am, inst, exog


def ajuste(am, inst, exog, dep, alvo):
    X = np.column_stack([np.ones(len(am)), am[inst + exog].to_numpy(float)])
    y = am[dep].to_numpy(float)
    P = np.linalg.pinv(X)
    j = 1 + inst.index(alvo)
    return X, y, P, j


def f_robusto(X, y, P, j, banda):
    beta = P @ y
    u = y - X @ beta
    T, K = X.shape
    var = hac_var(P[j], u, banda, T, K)
    return beta[j] ** 2 / var, beta[j], np.sqrt(var)


def teste(am, inst, exog, dep, alvo, h, rng, banda=None):
    """F robusto, p-valor por bootstrap sob a nula e intervalo percentil do F."""
    X, y, P, j = ajuste(am, inst, exog, dep, alvo)
    T, K = X.shape
    bw = banda if banda is not None else banda_nw(T, h)
    F, coef, ep = f_robusto(X, y, P, j, bw)
    Xr = np.delete(X, j, axis=1); Pr = np.linalg.pinv(Xr)
    ajr = Xr @ (Pr @ y); ur = y - ajr
    aju = X @ (P @ y); uu = y - aju
    Fnull, Fboot = np.empty(B), np.empty(B)
    for k in range(B):
        w = pesos_blocos(T, rng)
        Fnull[k] = f_robusto(X, ajr + ur * w, P, j, bw)[0]
        Fboot[k] = f_robusto(X, aju + uu * w, P, j, bw)[0]
    return dict(F=F, coef=coef, ep=ep, p_assint=float(1 - stats.chi2.cdf(F, 1)),
                p_boot=float((1 + (Fnull >= F).sum()) / (B + 1)),
                F_ic5=float(np.percentile(Fboot, 5)), F_ic95=float(np.percentile(Fboot, 95)),
                n=T, K=K, banda=bw)


# ------------------------------------------------------------------ rotina
def selecao(b, dum, pand, caso):
    linhas = []
    comum = montar(b, dum, pand, caso, PMAX, 0, regime=False)[0].index      # mesma amostra para todo p
    for p in range(1, PMAX + 1):
        am, inst, exog = montar(b, dum, pand, caso, p, 0, regime=False)
        am = am.loc[comum]
        r = sm.OLS(am["__x"], sm.add_constant(am[inst + exog], has_constant="add")).fit()
        linhas.append({"caso": caso, "p": p, "aic": r.aic, "bic": r.bic, "n": int(r.nobs)})
    t = pd.DataFrame(linhas)
    return t, int(t.loc[t.aic.idxmin(), "p"]), int(t.loc[t.bic.idxmin(), "p"])


def conferencia_g1(bm, dum, pand):
    """Reproduz a Tabela G.1: p = 3, banda 1, subamostras pelo anúncio de outubro de 2016."""
    out = {}
    b = bm.copy(); b["pos16"] = L.regime(b.Data, "2016-10-01")
    for caso in ("margem_gasolina", "margem_diesel"):
        am, inst, exog = montar(b, dum, pand, caso, 3, 0, regime=False, ini=0, lags_z=False)
        am = am.join(b["pos16"])
        for rot, sub in (("pos", am[am.pos16 == 1]), ("pre", am[am.pos16 == 0])):
            X, y, P, j = ajuste(sub, inst, exog, "__x", "z_all")
            out[(caso, rot)] = round(f_robusto(X, y, P, j, 1)[0], 2)
    return out


def main():
    bk, bm, dum, pand = bases()
    g1 = conferencia_g1(bm, dum, pand)
    print("conferência com a Tabela G.1:", g1)
    assert g1 == {("margem_gasolina", "pos"): 5.45, ("margem_gasolina", "pre"): 1.48,
                  ("margem_diesel", "pos"): 0.25, ("margem_diesel", "pre"): 0.91}, g1

    rng = np.random.default_rng(SEMENTE)
    sel_t, fs, fr, faixa = [], [], [], []
    for caso, (qual, *_r) in CASOS.items():
        b = bk if qual == "k" else bm
        t, p_aic, p_bic = selecao(b, dum, pand, caso)
        p_sel = min(p_aic + 1, PMAX)
        sel_t.append(t.assign(p_aic=p_aic, p_bic=p_bic, p_usado=p_sel))
        for p in sorted({p_sel, 3}):
            for regime in (False, True):
                am, inst, exog = montar(b, dum, pand, caso, p, 0, regime)
                for alvo in inst:
                    amostra = {"z_all": "completa", "z_pre": "antes de mar/2017", "z_pos": "depois de mar/2017"}[alvo]
                    if regime:
                        n_reg = int(am[alvo].ne(0).sum())
                    else:
                        n_reg = len(am)
                    o = teste(am, inst, exog, "__x", alvo, 0, rng)
                    fs.append({"caso": caso, "p": p, "p_escolhido": p == p_sel, "amostra": amostra,
                               "meses_com_instrumento": n_reg, **o})
                    r = teste(am, inst, exog, "__y", alvo, 0, rng)
                    fr.append({"caso": caso, "p": p, "p_escolhido": p == p_sel, "amostra": amostra,
                               "coef_forma_reduzida": r["coef"], "ep": r["ep"], "p_assint": r["p_assint"],
                               "p_boot": r["p_boot"], "n": r["n"]})
        # faixa do F nos horizontes 0 a 6, com p escolhido, sem bootstrap
        for regime in (False, True):
            for h in range(0, 7):
                am, inst, exog = montar(b, dum, pand, caso, p_sel, h, regime)
                for alvo in inst:
                    X, y, P, j = ajuste(am, inst, exog, "__x", alvo)
                    F = f_robusto(X, y, P, j, banda_nw(len(am), h))[0]
                    faixa.append({"caso": caso, "alvo": alvo, "h": h, "F": F})
        print(f"  {caso:16s} p_AIC={p_aic} p_BIC={p_bic} p usado={p_sel}")

    fs, fr = pd.DataFrame(fs), pd.DataFrame(fr)
    fh = pd.DataFrame(faixa)
    fx = fh.groupby(["caso", "alvo"]).F.agg(["min", "max"]).reset_index()
    with pd.ExcelWriter(SAIDA) as w:
        fs.to_excel(w, sheet_name="primeiro_estagio", index=False)
        fr.to_excel(w, sheet_name="forma_reduzida_ipca", index=False)
        fx.to_excel(w, sheet_name="faixa_F_h0_a_6", index=False)
        fh.pivot_table(index=["caso", "alvo"], columns="h", values="F").reset_index().to_excel(
            w, sheet_name="F_por_horizonte", index=False)
        pd.concat(sel_t).to_excel(w, sheet_name="selecao_defasagens", index=False)
        pd.DataFrame([{"caso": k[0], "amostra": k[1], "F": v} for k, v in g1.items()]).to_excel(
            w, sheet_name="conferencia_G1", index=False)
    pd.set_option("display.width", 220)
    print(fs[fs.p_escolhido][["caso", "p", "amostra", "F", "p_boot", "F_ic5", "F_ic95", "coef", "n", "banda"]]
          .round(3).to_string(index=False))
    print(fx.round(2).to_string(index=False))
    print("salvo", SAIDA)


if __name__ == "__main__":
    main()
