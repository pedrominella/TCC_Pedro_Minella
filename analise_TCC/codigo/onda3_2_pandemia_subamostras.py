# -*- coding: utf-8 -*-
"""
DOIS TESTES QUE FALTAVAM

(A) A quebra de 2020-2021 e regra de preco ou e a pandemia?
    As quebras de gasolina ao consumidor, diesel e IPCA Transportes caem todas entre
    jul/2020 e fev/2021 -- exatamente a recuperacao do Brent de US$20 para US$60. O teste
    de quebra pode estar achando o ciclo, nao a politica. Refaz-se a busca em quatro
    versoes:
        base                 como estava
        dummies de pandemia  indicadoras de mar/2020 a dez/2020 nos controles
        excluindo 2020       as observacoes de mar/2020 a dez/2020 saem da estimacao
        choque winsorizado   choque aparado em 1% e 99%
    Se a data sobreviver as tres correcoes, e regra de preco. Se sumir, era o ciclo.

(B) Interacao ou subamostras separadas?
    A especificacao por interacao impoe que TODA a dinamica restante -- defasagens da
    propria variavel, controles, sazonalidade -- seja identica nos dois regimes, e deixa
    apenas o coeficiente do choque mudar. A pergunta e se essa restricao distorce.

CORRECOES DE 27/08 (ver LEIA-ME, secao "Auditoria")
---------------------------------------------------
(E1) O TESTE DE (B) ESTAVA ERRADO E A CONCLUSAO SE INVERTE. A versao anterior comparava
     o coeficiente da interacao com o da subamostra usando

         t = (b_sub - b_int) / sqrt(ep_int^2 + ep_sub^2)

     ou seja, tratando os dois estimadores como INDEPENDENTES. Nao sao: a subamostra e
     um subconjunto dos mesmos dados, e a correlacao entre os dois e alta e positiva.
     Somar as variancias infla o denominador e o "0 de 78 casos" era manufaturado.
     O teste correto da restricao e direto: no modelo TOTALMENTE INTERAGIDO, as
     interacoes regime x (defasagens de y, defasagens do choque, controles) sao
     conjuntamente nulas? Esse Wald e o que se reporta agora. A comparacao de
     magnitudes entre interacao e subamostra permanece, mas como estatistica
     DESCRITIVA, sem p-valor -- que e tudo o que ela pode ser.
(E2) O teste rodava no corte antigo (set/2016) e SEM as dummies de pandemia, ou seja,
     nunca havia sido refeito na especificacao de referencia. Agora roda em mar/2017
     com as dummies de janela, e o corte antigo entra como comparacao.
(E3) Regime sem `.shift(1)` (C1) e bootstrap com residuo corrigido por alavancagem.

Nota sobre o tamanho do Wald de (E1): com 18 restricoes, HAC e T ~ 265, o Wald
assintotico sobrerejeita. Por isso o valor critico tambem sai por bootstrap, e os dois
p-valores sao reportados lado a lado.

Uso: python onda3_2_pandemia_subamostras.py pandemia | subamostras
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"
import lp_base_nova as L
from onda3_1_quebra_endogena import (perfil, supF, supF_multi, _resid_boot, _pesos,
                                     APARA, BLOCO, DATA_PPI)

B = 299
SEMENTE = 20260828


def sem_colunas_vazias(X):
    """Descarta colunas sem variacao MAS preserva a constante.

    `X.loc[:, X.std() > 0]` parece inocente e nao e: a coluna de uns criada por
    `add_constant` tem desvio-padrao zero e era eliminada junto, deixando a regressao
    SEM INTERCEPTO. Com as indicadoras mensais em drop_first, tirar o intercepto muda
    todos os coeficientes -- media de 0,02 a 0,12 nos coeficientes de regime.
    O filtro existe para descartar dummies de pandemia vazias numa subamostra; e so
    isso que ele deve fazer.
    """
    manter = (X.std() > 0) | (X.columns == "const")
    return X.loc[:, manter]


def base_variante(variante):
    b, dum = L.carregar("corrigida")
    extras = []
    if variante == "dummies de pandemia":
        extras = L.controles_pandemia(b, 0)          # busca roda em h=0
    if variante == "choque winsorizado":
        x = b["choque_petroleo"]
        b["choque_petroleo"] = x.clip(x.quantile(0.01), x.quantile(0.99))
    return b, dum + extras


def montar(nome, h, b, regs_extra):
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    regs = []
    for lag in range(1, L.LAGS + 1):
        regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in L.CONTROLES]
    regs += regs_extra
    d["__y"] = L.y_acum(d, nome, h)
    am = d[["Data", "__y", "choque_petroleo"] + regs].replace(
        [np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return am, regs


def pandemia():
    rng = np.random.default_rng(SEMENTE)
    alvos = [("gasolina_consumidor", 0), ("diesel", 0), ("ipca_transportes", 0),
             ("ipca_geral", 0), ("gasolina_refinaria", 0)]
    variantes = ["base", "dummies de pandemia", "excluindo 2020", "choque winsorizado"]
    linhas = []
    for variante in variantes:
        b, regs_extra = base_variante(variante)
        for nome, h in alvos:
            am, regs = montar(nome, h, b, regs_extra)
            if variante == "excluindo 2020":
                am = am[~am.Data.isin(L.PANDEMIA)].reset_index(drop=True)
            W = sm.add_constant(am[regs], has_constant="add")
            W = np.asarray(sem_colunas_vazias(W), float)
            n = len(am)
            datas = pd.DatetimeIndex(am["Data"])
            cand = list(datas[int(APARA * n):int((1 - APARA) * n)])
            y, X0, P0, pack = perfil(am, W, cand)
            fmax, tau, _ = supF(y, X0, P0, pack, n)
            ajus0, res0 = _resid_boot(y, X0, P0)                       # (E3)
            Y = ajus0[:, None] + res0[:, None] * np.column_stack([_pesos(rng, n) for _ in range(B)])
            fb = supF_multi(Y, X0, P0, list(pack.values()), n)
            linhas.append({"variavel": nome, "h": h, "variante": variante, "n": n,
                           "data_estimada": tau, "supF": fmax,
                           "p_bootstrap": float((fb >= fmax).mean()),
                           "critico_95": float(np.quantile(fb, 0.95))})
            print(f"  {variante:22s} {nome:20s} {tau:%Y-%m}  supF={fmax:6.1f}  "
                  f"p={linhas[-1]['p_bootstrap']:.3f}")
    t = pd.DataFrame(linhas)
    t["data_estimada"] = pd.to_datetime(t.data_estimada).dt.strftime("%Y-%m")
    t["rejeita_5"] = t.p_bootstrap < 0.05
    t.to_excel(SAIDA / "onda3_2_pandemia.xlsx", index=False)
    print("\n=== data estimada por variante ===")
    print(t.pivot(index="variavel", columns="variante", values="data_estimada").to_string())
    print("\n=== p-valor por variante ===")
    print(t.pivot(index="variavel", columns="variante", values="p_bootstrap").to_string(
        float_format=lambda x: f"{x:6.3f}"))
    return t


def subamostras():
    rng = np.random.default_rng(SEMENTE)
    b, dum, pand = L.carregar_pandemia("corrigida")
    linhas, restricao = [], []
    for corte_rot, corte in [("mar/2017", L.DATA_CORTE_FINAL), ("out/2016", L.DATA_CORTE)]:
        b["regime_pos"] = L.regime(b.Data, corte)                       # (E3)
        b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
        b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
        for nome in L.VARIAVEIS:
            d = b.copy()
            for lag in range(1, L.LAGS + 1):
                d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
            dinamica = []
            for lag in range(1, L.LAGS + 1):
                dinamica += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                            [f"{c}_lag{lag}" for c in L.CONTROLES]
            for h in range(0, L.H_MAX + 1):
                d["__y"] = L.y_acum(d, nome, h)
                comuns = dinamica + dum + pand(d, h)                    # (E2)

                # --- interacao (especificacao principal) ---
                regs = ["choque_pre", "choque_pos", "regime_pos"] + comuns
                am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                r = sm.OLS(am["__y"], sm.add_constant(am[regs], has_constant="add")).fit(
                    cov_type="HAC", cov_kwds=L.hac_kw(h))
                reg = {"corte": corte_rot, "variavel": nome, "h": h,
                       "int_pre": r.params["choque_pre"], "int_pre_ep": r.bse["choque_pre"],
                       "int_pos": r.params["choque_pos"], "int_pos_ep": r.bse["choque_pos"],
                       "int_n": int(r.nobs)}

                # --- (E1) teste DIRETO da restricao, no modelo totalmente interagido ---
                a2 = am.copy()
                inter = []
                for c in dinamica:
                    a2[f"x_{c}"] = a2[c] * a2["regime_pos"]
                    inter.append(f"x_{c}")
                X = sm.add_constant(a2[regs + inter], has_constant="add")
                X = sem_colunas_vazias(X)
                inter = [c for c in inter if c in X.columns]
                rf = sm.OLS(a2["__y"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
                W = float(rf.wald_test(" = 0, ".join(inter) + " = 0", scalar=True).statistic)
                reg["wald_restricao"] = W
                reg["gl_restricao"] = len(inter)
                reg["p_restricao_qui2"] = float(stats.chi2.sf(W, len(inter)))
                reg["coef_choque_pos_irrestrito"] = float(rf.params["choque_pos"])
                reg["coef_choque_pre_irrestrito"] = float(rf.params["choque_pre"])

                # --- subamostras, como descricao de magnitude ---
                regs2 = ["choque_petroleo"] + comuns
                for rot, dentro in [("sub_pre", d.Data < pd.Timestamp(corte)),
                                    ("sub_pos", d.Data >= pd.Timestamp(corte))]:
                    s = d[dentro]
                    b2 = s[["Data", "__y"] + regs2].replace([np.inf, -np.inf], np.nan).dropna()
                    X2 = sm.add_constant(b2[regs2], has_constant="add")
                    X2 = sem_colunas_vazias(X2)
                    if len(b2) <= X2.shape[1] + 5:
                        reg[rot] = reg[f"{rot}_ep"] = np.nan
                        reg[f"{rot}_n"] = len(b2)
                        continue
                    r2 = sm.OLS(b2["__y"], X2).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
                    reg[rot] = r2.params["choque_petroleo"]
                    reg[f"{rot}_ep"] = r2.bse["choque_petroleo"]
                    reg[f"{rot}_n"] = int(r2.nobs)
                linhas.append(reg)
            print(f"  {corte_rot}  {nome}")
    t = pd.DataFrame(linhas)
    for lado in ["pre", "pos"]:
        t[f"dif_{lado}"] = t[f"sub_{lado}"] - t[f"int_{lado}"]
    t.to_excel(SAIDA / "onda3_2_subamostras.xlsx", index=False)

    print("\n=== (E1) teste DIRETO da restricao de dinamica comum ===")
    print("H0: no modelo totalmente interagido, as interacoes regime x (defasagens de y,")
    print("    defasagens do choque, controles) sao conjuntamente nulas.")
    for corte_rot in t.corte.unique():
        s = t[t.corte == corte_rot]
        print(f"\n  corte {corte_rot}: rejeicoes a 10% = "
              f"{int((s.p_restricao_qui2 < 0.10).sum())} de {len(s)}"
              f" | a 5% = {int((s.p_restricao_qui2 < 0.05).sum())} de {len(s)}")
        v = s[s.h.isin([0, 3, 6])]
        print(v[["variavel", "h", "gl_restricao", "wald_restricao",
                 "p_restricao_qui2"]].to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    print("\n=== magnitude: interacao x subamostra (DESCRITIVO, sem p-valor) ===")
    for corte_rot in t.corte.unique():
        s = t[t.corte == corte_rot].dropna(subset=["sub_pre", "sub_pos"])
        for lado in ["pre", "pos"]:
            c = s[[f"int_{lado}", f"sub_{lado}"]].corr().iloc[0, 1]
            print(f"  {corte_rot} {lado}: correlacao {c:.3f} | "
                  f"mediana |dif| {s[f'dif_{lado}'].abs().median():.3f} | "
                  f"mediana |dif| relativa "
                  f"{(s[f'dif_{lado}'].abs() / s[f'int_{lado}'].abs().clip(lower=0.05)).median():.2f}")
    print("\nLeitura: a restricao E rejeitada, mas o coeficiente de interesse quase nao se")
    print("move. A redacao correta e 'a restricao e rejeitada e as subamostras entram como")
    print("robustez', nao 'a restricao nao e binding'.")
    return t


if __name__ == "__main__":
    {"pandemia": pandemia, "subamostras": subamostras}[
        sys.argv[1] if len(sys.argv) > 1 else "pandemia"]()
