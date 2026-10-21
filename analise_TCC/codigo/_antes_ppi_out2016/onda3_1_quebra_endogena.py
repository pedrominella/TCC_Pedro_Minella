# -*- coding: utf-8 -*-
"""
QUEBRA ENDOGENA -- em que mes os dados dizem que o repasse mudou?

Hoje o corte em set/2016 e IMPOSTO pela narrativa institucional (adocao da PPI). Isso e a
critica mais previsivel ao resultado central: a data foi escolhida pelo autor. Aqui a data e
ESTIMADA.

Teste: sup-Wald de quebra em data desconhecida (Andrews, 1993). Para cada data candidata tau
no intervalo aparado de 15% em cada ponta, estima-se

    y(t,h) = a + b1*choque(t)*1{t<tau} + b2*choque(t)*1{t>=tau} + d*1{t>=tau} + controles + u

e calcula-se a estatistica F da hipotese b1 = b2. A estatistica do teste e o maximo de F(tau)
sobre todas as datas candidatas, e a data estimada e o argmax. A quebra e testada SOMENTE no
coeficiente do choque, mais um deslocamento de intercepto: a pergunta e se o REPASSE mudou,
nao se tudo mudou.

Valor critico: por WILD BLOCK BOOTSTRAP sob H0 de ausencia de quebra (blocos de 12 meses,
pesos de Rademacher, residuos corrigidos por alavancagem). A distribuicao assintotica do
sup-F depende da aparagem e nao e qui-quadrado, entao o bootstrap e o caminho limpo.

Sobre a estatistica da busca: F(tau) e a versao homocedastica, por velocidade -- sao ate 188
datas candidatas vezes 299 replicas vezes 18 casos. Isso NAO invalida o teste, porque o valor
critico vem do bootstrap, que reproduz a heterocedasticidade e a dependencia dos dados: o que
se perde e potencia, nao tamanho. A inferencia FINAL na data escolhida usa HAC.

CORRECOES DE 27/08 (ver LEIA-ME, secao "Auditoria")
---------------------------------------------------
(D1) A BUSCA SEQUENCIAL DE SEGUNDA QUEBRA, que o cabecalho anterior anunciava
     ("ao estilo Bai-Perron") mas que NAO EXISTIA no codigo, esta implementada.
     Fixada a primeira quebra, procura-se uma segunda em cada segmento, com valor
     critico por bootstrap sob H0 de UMA quebra so.
(D2) A convencao de regime aqui (1{t >= tau}) era incompativel com a da estimacao,
     que aplicava `.shift(1)` e portanto comecava o regime um mes depois. Resolvido
     no modulo (`L.regime`, correcao C1): agora as duas pontas usam 1{t >= tau}.
(D3) HAC com correcao de amostra finita (C4) e residuo do bootstrap corrigido por
     alavancagem.

Uso: python onda3_1_quebra_endogena.py buscar | figuras
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
PASTA = AQUI.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
import lp_base_nova as L

B = 299
BLOCO = 12
SEMENTE = 20260827
APARA = 0.15
HORIZONTES_BUSCA = [0, 3, 6]
DATA_PPI = pd.Timestamp("2016-09-01")


def montar(nome, h, b, dum):
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    regs = []
    for lag in range(1, L.LAGS + 1):
        regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in L.CONTROLES]
    regs += dum
    d["__y"] = L.y_acum(d, nome, h)
    am = d[["Data", "__y", "choque_petroleo"] + regs].replace(
        [np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    W = sm.add_constant(am[regs], has_constant="add")
    return am, np.asarray(W, float)


def _ssr(y, X, P):
    r = y - X @ (P @ y)
    return float(r @ r)


def _ssr_multi(Y, X, P):
    """SSR de varias replicas de uma vez: Y e T x B."""
    R = Y - X @ (P @ Y)
    return (R * R).sum(axis=0)


def supF_multi(Y, X0, P0, pack, n):
    """sup-F de cada coluna de Y, vetorizado sobre as replicas do bootstrap."""
    s0 = _ssr_multi(Y, X0, P0)
    melhor = np.full(Y.shape[1], -np.inf)
    for Xt, Pt in pack:
        s1 = _ssr_multi(Y, Xt, Pt)
        melhor = np.maximum(melhor, ((s0 - s1) / 2) / (s1 / (n - Xt.shape[1])))
    return melhor


def _resid_boot(y, X, P):
    """(D3) residuo do ajuste sob H0, corrigido por alavancagem."""
    ajus = X @ (P @ y)
    lev = np.clip(np.einsum("ij,ji->i", X, P), 0.0, 0.999)
    return ajus, (y - ajus) / np.sqrt(1.0 - lev)


def perfil(am, W, candidatas):
    """Pre-calcula X(tau) e P(tau) para cada data candidata."""
    y = np.asarray(am["__y"], float)
    z = np.asarray(am["choque_petroleo"], float)
    datas = pd.DatetimeIndex(am["Data"])
    X0 = np.column_stack([z, W])                       # sem quebra
    P0 = np.linalg.pinv(X0)
    pack = {}
    for tau in candidatas:
        pos = (datas >= tau).astype(float)
        Xt = np.column_stack([z * (1 - pos), z * pos, pos, W])
        pack[tau] = (Xt, np.linalg.pinv(Xt))
    return y, X0, P0, pack


def supF(y, X0, P0, pack, n, k0=None):
    ssr0 = _ssr(y, X0, P0)
    melhor, fmax, perfilF = None, -np.inf, {}
    for tau, (Xt, Pt) in pack.items():
        ssr1 = _ssr(y, Xt, Pt)
        F = ((ssr0 - ssr1) / 2) / (ssr1 / (n - Xt.shape[1]))
        perfilF[tau] = F
        if F > fmax:
            fmax, melhor = F, tau
    return fmax, melhor, perfilF


def _pesos(rng, n):
    nb = int(np.ceil(n / BLOCO))
    return np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:n]


def segunda_quebra(am, W, tau1, rng):
    """(D1) busca sequencial de uma SEGUNDA quebra, ao estilo Bai-Perron.

    Fixada tau1, o modelo de referencia ja tem uma quebra. Em cada segmento
    procura-se tau2 que maximize o F de uma quebra ADICIONAL no coeficiente do
    choque (mais deslocamento de intercepto). Valor critico por bootstrap sob H0
    de UMA quebra: o DGP das replicas e o ajuste do modelo de uma quebra so.
    """
    y = np.asarray(am["__y"], float)
    z = np.asarray(am["choque_petroleo"], float)
    datas = pd.DatetimeIndex(am["Data"])
    n = len(y)
    pos1 = (datas >= tau1).astype(float)
    X1 = np.column_stack([z * (1 - pos1), z * pos1, pos1, W])
    P1 = np.linalg.pinv(X1)

    out = []
    for rot, dentro in [(f"antes de {tau1:%Y-%m}", datas < tau1),
                        (f"depois de {tau1:%Y-%m}", datas >= tau1)]:
        idx = np.flatnonzero(dentro)
        if len(idx) < 40:
            continue
        corte = max(int(APARA * len(idx)), 1)
        cand2 = list(datas[idx[corte:len(idx) - corte]])
        if len(cand2) < 5:
            continue
        pack2, taus2 = [], []
        for tau2 in cand2:
            # quebra ADICIONAL dentro do segmento, aninhada no modelo de 1 quebra
            pos2 = ((datas >= tau2) & dentro).astype(float)
            X2 = np.column_stack([z * (1 - pos1), z * pos1, pos1, z * pos2, pos2, W])
            pack2.append((X2, np.linalg.pinv(X2)))
            taus2.append(tau2)

        s1_obs = _ssr(y, X1, P1)
        Fobs = []
        for X2, P2 in pack2:
            s2 = _ssr(y, X2, P2)
            Fobs.append(((s1_obs - s2) / 2) / (s2 / (n - X2.shape[1])))
        j = int(np.argmax(Fobs)); f2, tau2 = float(Fobs[j]), taus2[j]

        ajus1, res1 = _resid_boot(y, X1, P1)
        Y = ajus1[:, None] + res1[:, None] * np.column_stack([_pesos(rng, n) for _ in range(B)])
        s1b = _ssr_multi(Y, X1, P1)
        fb = np.full(B, -np.inf)
        for X2, P2 in pack2:
            s2b = _ssr_multi(Y, X2, P2)
            fb = np.maximum(fb, ((s1b - s2b) / 2) / (s2b / (n - X2.shape[1])))
        out.append({"segmento": rot, "data_2a_quebra": tau2, "supF_2a": f2,
                    "p_bootstrap_2a": float((fb >= f2).mean()),
                    "critico_95_2a": float(np.quantile(fb, 0.95))})
    return out


def buscar():
    rng = np.random.default_rng(SEMENTE)
    b, dum = L.carregar("corrigida")
    linhas, perfis, segundas = [], [], []
    for nome in L.VARIAVEIS:
        for h in HORIZONTES_BUSCA:
            am, W = montar(nome, h, b, dum)
            n = len(am)
            datas = pd.DatetimeIndex(am["Data"])
            lo, hi = int(APARA * n), int((1 - APARA) * n)
            candidatas = list(datas[lo:hi])
            y, X0, P0, pack = perfil(am, W, candidatas)
            fmax, tau_hat, perfilF = supF(y, X0, P0, pack, n)

            # bootstrap sob H0 de ausencia de quebra, vetorizado sobre as replicas
            ajust0, resid0 = _resid_boot(y, X0, P0)
            Yb = ajust0[:, None] + resid0[:, None] * np.column_stack(
                [_pesos(rng, n) for _ in range(B)])
            fmaxb = supF_multi(Yb, X0, P0, list(pack.values()), n)
            p_boot = float((fmaxb >= fmax).mean())

            # inferencia HAC na data estimada e na data imposta
            def wald_hac(tau):
                pos = (datas >= tau).astype(float)          # (D2) mesma convencao de L.regime
                Xd = pd.DataFrame(np.column_stack(
                    [np.asarray(am["choque_petroleo"]) * (1 - pos),
                     np.asarray(am["choque_petroleo"]) * pos, pos, W]),
                    columns=["choque_pre", "choque_pos", "regime"] + [f"w{i}" for i in range(W.shape[1])])
                r = sm.OLS(y, Xd).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
                w = r.wald_test("choque_pre = choque_pos", scalar=True)
                return (float(r.params["choque_pre"]), float(r.params["choque_pos"]),
                        float(w.pvalue))

            pre_e, pos_e, p_e = wald_hac(tau_hat)
            pre_p, pos_p, p_p = wald_hac(DATA_PPI)

            linhas.append({
                "variavel": nome, "h": h, "n": n,
                "data_estimada": tau_hat, "supF": fmax, "p_bootstrap": p_boot,
                "critico_95_bootstrap": float(np.quantile(fmaxb, 0.95)),
                "meses_ate_a_PPI": int(round((tau_hat - DATA_PPI).days / 30.44)),
                "coef_pre_estimada": pre_e, "coef_pos_estimada": pos_e, "wald_p_estimada": p_e,
                "coef_pre_set2016": pre_p, "coef_pos_set2016": pos_p, "wald_p_set2016": p_p,
                "F_em_set2016": perfilF.get(DATA_PPI, np.nan),
            })
            for tau, F in perfilF.items():
                perfis.append({"variavel": nome, "h": h, "data": tau, "F": F})

            # (D1) segunda quebra, so onde a primeira rejeita
            if p_boot < 0.05:
                for s in segunda_quebra(am, W, tau_hat, rng):
                    segundas.append({"variavel": nome, "h": h, "data_1a_quebra": tau_hat, **s})
            print(f"  {nome:20s} h={h:2d}  quebra em {tau_hat:%Y-%m}  supF={fmax:6.1f}  p={p_boot:.3f}")

    t = pd.DataFrame(linhas); pf = pd.DataFrame(perfis); sg = pd.DataFrame(segundas)
    with pd.ExcelWriter(SAIDA / "onda3_1_quebra_endogena.xlsx") as w:
        t.to_excel(w, sheet_name="resumo", index=False)
        pf.to_excel(w, sheet_name="perfil_F", index=False)
        if len(sg):
            sg.to_excel(w, sheet_name="segunda_quebra", index=False)

    print("\n=== data estimada da quebra, por variavel e horizonte ===")
    print(t[["variavel", "h", "data_estimada", "supF", "p_bootstrap", "critico_95_bootstrap",
             "meses_ate_a_PPI"]].to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    print("\n=== repasse antes e depois: data estimada x set/2016 imposta ===")
    print(t[["variavel", "h", "data_estimada", "coef_pre_estimada", "coef_pos_estimada",
             "wald_p_estimada", "coef_pre_set2016", "coef_pos_set2016", "wald_p_set2016"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))
    if len(sg):
        print("\n=== (D1) segunda quebra, busca sequencial ===")
        print(sg.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
        print(f"\nsegundas quebras significativas a 5%: "
              f"{int((sg.p_bootstrap_2a < 0.05).sum())} de {len(sg)}")
    return t, pf, sg


def figuras():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pf = pd.read_excel(SAIDA / "onda3_1_quebra_endogena.xlsx", sheet_name="perfil_F")
    t = pd.read_excel(SAIDA / "onda3_1_quebra_endogena.xlsx", sheet_name="resumo")
    for nome in ["gasolina_refinaria", "gasolina_consumidor", "diesel", "ipca_geral"]:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        cores = {0: "#0B5A56", 3: "#9E3D25", 6: "#5F6549"}
        for h in HORIZONTES_BUSCA:
            s = pf[(pf.variavel == nome) & (pf.h == h)].sort_values("data")
            ax.plot(pd.to_datetime(s.data), s.F, color=cores[h], lw=1.6, label=f"h = {h}")
            r = t[(t.variavel == nome) & (t.h == h)].iloc[0]
            ax.axhline(r.critico_95_bootstrap, color=cores[h], lw=0.7, ls=":")
            ax.plot(pd.to_datetime(r.data_estimada), r.supF, "o", color=cores[h], ms=6)
        ax.axvline(DATA_PPI, color="0.25", lw=1.2, ls="--")
        ax.annotate("PPI, set/2016", xy=(DATA_PPI, ax.get_ylim()[1] * 0.94),
                    fontsize=7.5, rotation=90, ha="right", va="top", color="0.25")
        ax.set_ylabel("estatistica F da igualdade de repasse")
        ax.set_title(f"{nome}: onde os dados colocam a quebra\n"
                     "pontilhado = valor critico de 5% por bootstrap; bola = data estimada", fontsize=9)
        ax.legend(fontsize=7.5); fig.tight_layout()
        fig.savefig(FIGS / f"quebra_endogena_{nome}.png", dpi=150); plt.close(fig)
        print(f"  figura {nome}")


if __name__ == "__main__":
    {"buscar": buscar, "figuras": figuras}[sys.argv[1] if len(sys.argv) > 1 else "buscar"]()
