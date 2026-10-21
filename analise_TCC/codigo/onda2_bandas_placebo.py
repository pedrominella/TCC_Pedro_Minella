# -*- coding: utf-8 -*-
"""
ONDA 2, ITENS 2.1 E 2.2 -- INFERENCIA UNIFORME E PLACEBO NAS LOCAL PROJECTIONS

Base: analise_TCC (meta Selic oficial SGS 432, resposta acumulada em log-nivel,
indicadoras de pandemia por janela, quebra em mar/2017).

2.1 -- BANDAS UNIFORMES (sup-t) E TESTES CONJUNTOS
    A banda sup-t (Montiel Olea e Plagborg-Moller, 2019) cobre a funcao de resposta
    INTEIRA com 90% de probabilidade: banda = beta(h) +- c * ep(h), com c tal que
    P(max_h |t(h)| <= c) = 0,90. Sempre c >= 1,645.

2.2 -- PLACEBO EM HORIZONTES NEGATIVOS
    O choque em t nao deveria prever variacoes ANTERIORES a t.

CORRECOES DE 27/08 (ver LEIA-ME, secao "Auditoria")
---------------------------------------------------
(A1, A2) UMA SO ESTIMATIVA DE VARIANCIA EM TODO O ARQUIVO.
     A versao anterior tinha DUAS, lado a lado e discordantes entre 8% e 35%: as
     bandas usavam o erro-padrao HAC analitico, e os testes de Wald conjuntos usavam
     `np.cov` das replicas de um wild block bootstrap. Pior, a banda uniforme era
     calculada como |beta* - beta|/ep com o `ep` HAC FIXO, o que nao e
     estudentizacao -- o `c` resultante misturava a multiplicidade entre horizontes
     com o descasamento entre as duas variancias, descasamento que ia de 1,27 em h=0
     a 0,94 em h=12. A banda saia estreita demais em horizonte curto e larga demais
     em horizonte longo, exatamente na direcao que produzia a leitura "so cai a
     cauda longa".

     Agora TUDO sai da mesma covariancia HAC:
       - banda pontual  : beta(h) +- 1,645 * ep_HAC(h)
       - banda uniforme : beta(h) +- c * ep_HAC(h), com c pelo sup-t PLUG-IN, isto
                          e, o quantil de max_h |Z_h| com Z ~ N(0, R) e R a matriz
                          de correlacao implicada pela covariancia HAC conjunta
                          entre horizontes. E o mesmo erro-padrao, outro valor
                          critico -- nao um segundo tipo de erro-padrao.
       - Wald conjunto  : forma quadratica na mesma covariancia HAC conjunta.

     O wild block bootstrap sobreviveu em UM lugar so: o valor critico do Wald
     conjunto, onde a qui-quadrado com sete restricoes e HAC sobrerejeita e nao ha
     alternativa analitica. Os dois p-valores sao reportados lado a lado.

     Por que o plug-in e nao o bootstrap para a banda: com blocos de 12 meses o
     bootstrap perde a dependencia MA(h) quando h se aproxima do tamanho do bloco
     (medido: razao dp(bootstrap)/ep_HAC caindo de ~1,2 em h=0 para ~0,8 em h=12),
     e o valor critico herdaria esse vies.

(A3) PLACEBO SEM CONTROLE FUTURO. Antes so as defasagens de y eram deslocadas: com
     dependente em t-k, os controles e as defasagens do choque continuavam em
     t-1..t-3, POSTERIORES a dependente. Condicionar em variavel futura e "bad
     control". Agora todo o conjunto de condicionamento vai para t-k-1..t-k-3.

(A4) Regime em mar/2017 sem `.shift(1)` (C1 do modulo) e dummies de pandemia por
     janela do horizonte (C3), com HAC corrigido por T/(T-K) (C4).

Uso: python onda2_bandas_placebo.py rodar | figuras
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
PASTA = AQUI.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
SAIDA.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)

import lp_base_nova as L        # reaproveita carregar(), VARIAVEIS, constantes

B = 999
BLOCO = 12
SEMENTE = 20260826
NIVEL = 0.90
H_CONJUNTO = 6
PLACEBOS = [1, 2, 3]
PRINCIPAIS = ["gasolina_consumidor", "diesel", "ipca_transportes", "ipca_geral"]


# ------------------------------------------------------------------
# pecas de algebra
# ------------------------------------------------------------------
def ajustar(am, regs, h, alvo):
    """OLS + tudo que o bootstrap precisa.

    alvo = "coluna"            -> a combinacao de interesse e beta[coluna]
    alvo = ("c1", "c2")        -> a combinacao e beta[c2] - beta[c1]
    """
    X = sm.add_constant(am[regs], has_constant="add")
    cols = list(X.columns)
    Xv = np.asarray(X, float); yv = np.asarray(am["__y"], float)
    A = np.linalg.pinv(Xv.T @ Xv)
    P = np.linalg.pinv(Xv)
    beta = P @ yv
    resid = yv - Xv @ beta
    if isinstance(alvo, tuple):
        a = A[:, cols.index(alvo[1])] - A[:, cols.index(alvo[0])]
        valor = float(beta[cols.index(alvo[1])] - beta[cols.index(alvo[0])])
    else:
        a = A[:, cols.index(alvo)]
        valor = float(beta[cols.index(alvo)])
    banda = max(1, h + 1)
    lev = np.clip(np.einsum("ij,ji->i", Xv, P), 0.0, 0.999)      # alavancagem h_tt
    return {"datas": pd.DatetimeIndex(am["Data"]), "X": Xv, "P": P, "a": a,
            "Xa": Xv @ a, "beta": beta, "resid": resid, "ajuste": Xv @ beta,
            "resid_boot": resid / np.sqrt(1.0 - lev),
            "valor": valor, "ep": float(np.sqrt(L.hac_var_combo(Xv, resid, a, banda))),
            "banda": banda, "nobs": len(am), "K": Xv.shape[1]}


def replicas(g, etas, datas_ref):
    """(A1) valor* e erro-padrao* de cada replica, vetorizado sobre o bootstrap.

    Os residuos entram corrigidos por alavancagem, u/sqrt(1-h_tt) (Davidson e
    Flachaire): E[u_t^2] = sigma^2 (1 - h_tt) e, com K/T ~ 15%, usar o residuo cru
    encolhe a variancia do bootstrap em cerca de 15% de forma sistematica.
    """
    pos = datas_ref.get_indexer(g["datas"])
    E = etas[pos, :]                                       # T x B
    Y = g["ajuste"][:, None] + g["resid_boot"][:, None] * E
    val = g["a"] @ (g["X"].T @ Y)                          # a'X'y* = combinacao
    U = Y - g["X"] @ (g["P"] @ Y)
    V = g["Xa"][:, None] * U
    s = (V * V).sum(axis=0)
    for l in range(1, g["banda"] + 1):
        s += 2.0 * (1.0 - l / (g["banda"] + 1)) * (V[l:] * V[:-l]).sum(axis=0)
    s *= g["nobs"] / (g["nobs"] - g["K"])
    return val, np.sqrt(np.maximum(s, 1e-300))


def sup_t_plugin(V, nivel=NIVEL, nsim=200000, semente=SEMENTE):
    """Valor critico sup-t plug-in (Montiel Olea e Plagborg-Moller, 2019).

    c tal que P(max_h |Z_h| <= c) = nivel, com Z ~ N(0, R) e R a matriz de
    CORRELACAO implicada pela covariancia HAC conjunta entre horizontes. Usa
    exatamente a mesma estimativa de variancia das bandas pontuais -- e por isso
    que este, e nao o bootstrap de blocos, e o criterio principal: o bootstrap com
    blocos de 12 meses subestima a dependencia quando h se aproxima do tamanho do
    bloco (medido: razao dp(bootstrap)/ep_HAC caindo de 1,2 em h=0 para 0,8 em
    h=12), e o valor critico herda esse vies.
    """
    dp = np.sqrt(np.diag(V))
    R = V / np.outer(dp, dp)
    R = (R + R.T) / 2
    w, Q = np.linalg.eigh(R)
    Ar = Q @ np.diag(np.sqrt(np.maximum(w, 0)))
    Z = np.random.default_rng(semente).standard_normal((nsim, R.shape[0])) @ Ar.T
    return float(np.quantile(np.abs(Z).max(axis=1), nivel))


def hac_conjunta_horizontes(gs, banda):
    """(A2) covariancia HAC entre as combinacoes de varios horizontes.

    Cov(h, g) = sum_t v_h(t) v_g(t) + sum_l w_l sum_t [v_h(t)v_g(t-l) + v_h(t-l)v_g(t)]
    com v_h(t) = (X_h a_h)_t * u_h(t). Exige amostra comum entre os horizontes.
    """
    Vm = np.column_stack([g["Xa"] * g["resid"] for g in gs])   # T x H
    S = Vm.T @ Vm
    for l in range(1, banda + 1):
        w = 1.0 - l / (banda + 1)
        G = Vm[l:].T @ Vm[:-l]
        S += w * (G + G.T)
    T, K = gs[0]["nobs"], gs[0]["K"]
    return S * (T / (T - K))


def _amostra(d, regs):
    return d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()


# ------------------------------------------------------------------
def rodar():
    rng = np.random.default_rng(SEMENTE)
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)          # (A4)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
    datas_ref = pd.DatetimeIndex(sorted(b["Data"].unique()))
    n = len(datas_ref); nb = int(np.ceil(n / BLOCO))
    etas = np.column_stack([np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:n]
                            for _ in range(B)])                     # n x B

    kmax = L.LAGS + max(PLACEBOS)
    linhas, conjuntos = [], []
    for nome in L.VARIAVEIS:
        d = b.copy()
        for lag in range(1, kmax + 1):
            d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
            d[f"choque_petroleo_lag{lag}"] = d["choque_petroleo"].shift(lag)
            for c in L.CONTROLES:
                d[f"{c}_lag{lag}"] = d[c].shift(lag)

        def defasagens(desloc):
            """defasagens de y, do choque e dos controles a partir de t-desloc-1."""
            r = []
            for lag in range(desloc + 1, desloc + L.LAGS + 1):
                r += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                     [f"{c}_lag{lag}" for c in L.CONTROLES]
            return r

        # ---------- respostas h = 0..12 ----------
        guard, montado = {}, []
        for h in range(0, L.H_MAX + 1):
            d["__y"] = L.y_acum(d, nome, h)
            regs = ["choque_petroleo"] + defasagens(0) + dum + pand(d, h)
            am = _amostra(d, regs)
            montado.append((h, regs, am))
            guard[h] = ajustar(am, regs, h, "choque_petroleo")
            linhas.append({"variavel": nome, "h": h, "tipo": "resposta",
                           "coeficiente": guard[h]["valor"], "erro_padrao": guard[h]["ep"],
                           "observacoes": guard[h]["nobs"]})

        # ---------- (A2) sup-t PLUG-IN: criterio principal ----------
        # Matriz de correlacao entre os beta(h) pela covariancia HAC conjunta, na
        # amostra comum aos 13 horizontes. Os erros-padrao continuam sendo os de
        # cada horizonte na sua amostra maxima, como e praxe.
        comuns = set.intersection(*[set(am["Data"]) for _, _, am in montado])
        gs_c = [ajustar(am[am["Data"].isin(comuns)], regs, h, "choque_petroleo")
                for h, regs, am in montado]
        V_h = hac_conjunta_horizontes(gs_c, L.H_MAX + 1)
        c_unif = sup_t_plugin(V_h)

        # ---------- placebos h = -1, -2, -3 ----------
        for k in PLACEBOS:
            d["__y"] = d[f"dlog_{nome}"].shift(k)
            mi = d["Data"].dt.year * 12 + d["Data"].dt.month
            pk = []
            for s in L.PANDEMIA:                       # pandemia do mes da dependente
                c = f"pand_{s:%Y%m}"
                d[c] = (mi - k == s.year * 12 + s.month).astype(float)
                pk.append(c)
            regs = ["choque_petroleo"] + defasagens(k) + dum + pk    # (A3)
            guard[-k] = ajustar(_amostra(d, regs), regs, 0, "choque_petroleo")
            linhas.append({"variavel": nome, "h": -k, "tipo": "placebo",
                           "coeficiente": guard[-k]["valor"],
                           "erro_padrao": guard[-k]["ep"],
                           "observacoes": guard[-k]["nobs"]})

        # ---------- bandas: pontual e uniforme, da MESMA variancia HAC ----------
        for reg in linhas:
            if reg["variavel"] != nome:
                continue
            cf, ep = reg["coeficiente"], reg["erro_padrao"]
            t_obs = abs(cf / ep)
            reg["pontual_inferior"] = cf - L.Z_90 * ep
            reg["pontual_superior"] = cf + L.Z_90 * ep
            reg["p_valor_pontual"] = float(2 * (1 - stats.norm.cdf(t_obs)))
            reg["signif_pontual_10"] = bool(t_obs > L.Z_90)
            if reg["tipo"] == "resposta":
                reg["critico_uniforme"] = c_unif
                reg["uniforme_inferior"] = cf - c_unif * ep
                reg["uniforme_superior"] = cf + c_unif * ep
                reg["signif_uniforme_10"] = bool(t_obs > c_unif)

        # ---------- (A2) testes conjuntos, covariancia HAC analitica ----------
        for rotulo, alvo, cols in [
                ("resposta nula h=0..6", "choque_petroleo", ["choque_petroleo"]),
                ("igualdade de regimes h=0..6", ("choque_pre", "choque_pos"),
                 ["choque_pre", "choque_pos", "regime_pos"])]:
            # 1) monta cada horizonte, 2) intersecta as datas, 3) reestima na comum
            montado = []
            for h in range(0, H_CONJUNTO + 1):
                d["__y"] = L.y_acum(d, nome, h)
                regs = cols + defasagens(0) + dum + pand(d, h)
                montado.append((h, regs, _amostra(d, regs)))
            comuns = set.intersection(*[set(am["Data"]) for _, _, am in montado])
            gs = [ajustar(am[am["Data"].isin(comuns)], regs, h, alvo)
                  for h, regs, am in montado]
            V = hac_conjunta_horizontes(gs, H_CONJUNTO + 1)
            Vi = np.linalg.pinv(V)
            bhat = np.array([g["valor"] for g in gs])
            W = float(bhat @ Vi @ bhat)
            Bm = np.column_stack([replicas(g, etas, datas_ref)[0] for g in gs])
            Wb = np.einsum("ij,jk,ik->i", Bm - bhat, Vi, Bm - bhat)
            conjuntos.append({"variavel": nome, "teste": rotulo, "estatistica": W,
                              "p_qui2": float(stats.chi2.sf(W, H_CONJUNTO + 1)),
                              "p_bootstrap": float((Wb >= W).mean()),
                              "critico_95_qui2": float(stats.chi2.ppf(0.95, H_CONJUNTO + 1)),
                              "critico_95_bootstrap": float(np.quantile(Wb, 0.95)),
                              "observacoes": gs[0]["nobs"]})
        print(f"  {nome}: c_uniforme = {c_unif:.3f}")

    t = pd.DataFrame(linhas); tc = pd.DataFrame(conjuntos)
    t["principal"] = t["variavel"].isin(PRINCIPAIS)
    with pd.ExcelWriter(SAIDA / "onda2_bandas_placebo_final.xlsx") as w:
        t.to_excel(w, sheet_name="bandas", index=False)
        tc.to_excel(w, sheet_name="testes_conjuntos", index=False)

    print("\n=== 2.1 bandas: pontual x uniforme (horizontes do artigo) ===")
    v = t[(t.tipo == "resposta") & (t.h.isin(L.HORIZONTES))]
    print(v[["variavel", "h", "coeficiente", "erro_padrao", "critico_uniforme",
             "signif_pontual_10", "signif_uniforme_10"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))
    r = t[t.tipo == "resposta"]
    print("\ntotal das 78 respostas -- significativas a 10%:")
    print(f"  pontual  (HAC, c = 1,645)      : {int(r.signif_pontual_10.sum())}/{len(r)}")
    print(f"  uniforme (HAC, sup-t plug-in)  : {int(r.signif_uniforme_10.sum())}/{len(r)}")
    print("  as duas saem da MESMA variancia HAC; muda so o valor critico.")

    print("\n=== 2.2 placebo em horizontes negativos ===")
    p = t[t.tipo == "placebo"]
    print(p[["variavel", "h", "coeficiente", "erro_padrao", "p_valor_pontual"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\nplacebos significativos a 10%: {int((p.p_valor_pontual < 0.10).sum())} de {len(p)}")

    print("\n=== 2.1 testes conjuntos (covariancia HAC, critico por bootstrap) ===")
    print(tc.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    return t, tc


def figuras():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.read_excel(SAIDA / "onda2_bandas_placebo_final.xlsx", sheet_name="bandas")
    for nome in PRINCIPAIS:
        s = t[t.variavel == nome].sort_values("h")
        r = s[s.tipo == "resposta"]; p = s[s.tipo == "placebo"]
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        ax.fill_between(r.h, r.uniforme_inferior, r.uniforme_superior,
                        color="#9E3D25", alpha=0.14, label="banda uniforme sup-t 90%")
        ax.fill_between(r.h, r.pontual_inferior, r.pontual_superior,
                        color="#0B5A56", alpha=0.20, label="banda pontual 90%")
        ax.plot(r.h, r.coeficiente, color="#0B5A56", lw=2, label="resposta acumulada")
        ax.errorbar(p.h, p.coeficiente, yerr=1.645 * p.erro_padrao, fmt="o",
                    color="#5F6549", ms=4, capsize=3, lw=1, label="placebo (pre-choque)")
        ax.axhline(0, color="0.4", lw=0.8); ax.axvline(-0.5, color="0.7", lw=0.8, ls=":")
        ax.set_xlabel("meses"); ax.set_ylabel("%")
        ax.set_title(f"{nome}: resposta ao choque do petroleo\nbandas pontual e uniforme, com placebo", fontsize=9)
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(FIGS / f"lp_bandas_placebo_final_{nome}.png", dpi=150); plt.close(fig)
        print(f"  figura {nome}")


if __name__ == "__main__":
    {"rodar": rodar, "figuras": figuras}[sys.argv[1] if len(sys.argv) > 1 else "rodar"]()
