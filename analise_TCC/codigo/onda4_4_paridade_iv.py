# -*- coding: utf-8 -*-
"""
ONDA 4.4 -- A PARIDADE COMO INSTRUMENTO DO PRECO DOMESTICO DE COMBUSTIVEL

O que a decomposicao de canal (onda2_4) e a segunda volta (onda2_7) NAO fazem e
identificar causalmente o efeito do preco do combustivel sobre os demais precos.
La o mediador entra por OLS, e a leitura defensavel e contabil, nao causal.

Aqui o mediador e instrumentado. A logica vem da propria politica: sob paridade
de importacao, o preco domestico segue MECANICAMENTE a cotacao internacional do
produto refinado. A cotacao do Golfo dos EUA e preco mundial, o Brasil e tomador,
e ela nao tem por que afetar precos brasileiros nao-combustiveis a nao ser
ATRAVES do combustivel brasileiro.

    2o estagio : y_acum(t,h) = a + c * med_acum(t,h) + controles + u
    1o estagio : med_acum(t,h) = p + f * golfo_acum(t,h) + controles + v

    y     : IPCA nao-comercializaveis e IPCA ex-combustiveis, que EXCLUEM
            combustivel por construcao, entao c e efeito de segunda ordem puro
    med   : gasolina ao consumidor, acumulada de t-1 a t+h
    golfo : gasolina convencional no Golfo dos EUA, em DOLAR, mesma acumulacao

DUAS VERSOES DO INSTRUMENTO
  (a) sem Brent no modelo: instrumento e a variacao cheia da cotacao do Golfo.
      Primeiro estagio forte, exclusao mais exigente.
  (b) com Brent no modelo: a variacao do Golfo comum ao petroleo bruto e
      absorvida, e o que identifica e o CRACK SPREAD, a margem de refino.
      Primeiro estagio mais fraco, exclusao bem mais defensavel: margem de refino
      americana nao tem canal proprio para o preco de servicos no Brasil.

O INSTRUMENTO EM DOLAR, E NAO EM REAL
  A paridade completa multiplica a cotacao pelo cambio. Usar a paridade em real
  daria primeiro estagio mais forte e importaria toda a endogeneidade do cambio
  com a inflacao brasileira. Mantem-se o instrumento em DOLAR e o cambio como
  controle defasado, como no modelo principal. A versao em real entra so como
  robustez, com a ressalva registrada.

O PRIMEIRO ESTAGIO E, ELE PROPRIO, UM TESTE DO MECANISMO
  Se a PPI de fato amarrou o preco domestico a paridade, o F do primeiro estagio
  tem de ser alto no periodo PPI e baixo antes. Isso e reportado por regime.

Uso: python onda4_4_paridade_iv.py
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
import onda4_1_tres_regimes as O41
from onda2_7_segunda_volta import preparar

RAIZ = Path(__file__).resolve().parent.parent.parent
PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
GOLFO = RAIZ / "base_de_dados" / "gasolina_golfo.csv"
DESFECHOS = ["ipca_naocomercializaveis", "ipca_ex_comb_central"]
MEDIADOR = "gasolina_consumidor"


def carregar_golfo():
    g = pd.read_csv(GOLFO, skiprows=4)
    g.columns = [c.strip() for c in g.columns]
    g["Data"] = pd.to_datetime(g["mes_texto"].str.strip(), format="%b %Y")
    g["ln_golfo"] = 100 * np.log(pd.to_numeric(g["preco_dolar_galao"],
                                               errors="coerce"))
    return g[["Data", "ln_golfo"]].dropna().sort_values("Data")


def base():
    b, dum, pand = O41.montar(O41.CORTE_FIM)
    b = preparar(b)[0] if isinstance(preparar(b), tuple) else b
    g = carregar_golfo()
    b = b.merge(g, on="Data", how="left")
    faltam = int(b["ln_golfo"].isna().sum())
    assert faltam == 0, f"{faltam} meses sem cotacao do Golfo"
    b["dlog_golfo"] = b["ln_golfo"].diff()
    b["ln_paridade_brl"] = b["ln_golfo"] + 100 * np.log(b["Cambio"])
    for lag in range(1, L.LAGS + 1):
        b[f"dlog_golfo_lag{lag}"] = b["dlog_golfo"].shift(lag)
    return b, dum, pand


def acum(b, nome, h):
    return b[f"ln_{nome}"].shift(-h) - b[f"ln_{nome}"].shift(1)


def hac_iv(y, X, Z, banda):
    """2SLS com covariancia HAC de Newey-West e correcao T/(T-K)."""
    Xv, Zv, yv = np.asarray(X, float), np.asarray(Z, float), np.asarray(y, float)
    Pz = Zv @ np.linalg.pinv(Zv.T @ Zv) @ Zv.T
    Xh = Pz @ Xv
    A = np.linalg.pinv(Xh.T @ Xv)
    beta = A @ (Xh.T @ yv)
    u = yv - Xv @ beta
    T, K = len(yv), Xv.shape[1]
    S = (Xh * u[:, None]).T @ (Xh * u[:, None])
    for l in range(1, banda + 1):
        w = 1.0 - l / (banda + 1)
        G = (Xh[l:] * u[l:, None]).T @ (Xh[:-l] * u[:-l, None])
        S += w * (G + G.T)
    S *= T / (T - K)
    V = A @ S @ A.T
    return beta, np.sqrt(np.maximum(np.diag(V), 0.0)), T, K


def primeiro_estagio_por_regime(b, dum, pand, h, com_brent):
    """F robusto do instrumento, dentro de cada regime."""
    d = b.copy()
    d["__m"] = acum(d, MEDIADOR, h)
    din = []
    for lag in range(1, L.LAGS + 1):
        din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}", f"dlog_golfo_lag{lag}"] + \
               [f"{c}_lag{lag}" for c in L.CONTROLES]
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{MEDIADOR}"].shift(lag)
    exog = (["choque_petroleo"] if com_brent else []) + din + dum + pand(d, h)
    d["bloco"] = np.where(d.d_pos == 1, "pos-PPI",
                          np.where(d.d_ppi == 1, "PPI", "pre-PPI"))
    lin = []
    for r in ["pre-PPI", "PPI", "pos-PPI"]:
        s = d[d.bloco == r]
        cols = ["__m", "golfo_acum"] + exog
        am = s[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(am) < len(exog) + 6:
            lin.append({"regime": r, "n": len(am), "coef": np.nan, "F": np.nan})
            continue
        X = sm.add_constant(am[["golfo_acum"] + exog], has_constant="add")
        m = sm.OLS(am["__m"], X).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
        t = m.params["golfo_acum"] / m.bse["golfo_acum"]
        lin.append({"regime": r, "n": int(m.nobs),
                    "coef": float(m.params["golfo_acum"]), "F": float(t ** 2)})
    return pd.DataFrame(lin)


def rodar():
    b, dum, pand = base()
    linhas, primeiros = [], []
    for h in range(0, 7):
        b["golfo_acum"] = acum(b, "golfo", h)
        for com_brent in [False, True]:
            fe = primeiro_estagio_por_regime(b, dum, pand, h, com_brent)
            fe["h"] = h; fe["com_brent"] = com_brent
            primeiros.append(fe)
            for desf in DESFECHOS:
                d = b.copy()
                d["__y"] = acum(d, desf, h)
                d["__m"] = acum(d, MEDIADOR, h)
                for lag in range(1, L.LAGS + 1):
                    d[f"y_lag{lag}"] = d[f"dlog_{desf}"].shift(lag)
                din = []
                for lag in range(1, L.LAGS + 1):
                    din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}",
                            f"dlog_golfo_lag{lag}"] + \
                           [f"{c}_lag{lag}" for c in L.CONTROLES]
                exog = (["choque_petroleo"] if com_brent else []) + \
                       din + dum + pand(d, h)
                cols = ["__y", "__m", "golfo_acum"] + exog
                am = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
                Xo = sm.add_constant(am[["__m"] + exog], has_constant="add")
                Zi = sm.add_constant(am[["golfo_acum"] + exog], has_constant="add")
                bt, ep, T, K = hac_iv(am["__y"], Xo, Zi, max(1, h + 1))
                i = list(Xo.columns).index("__m")
                # OLS para comparacao
                mo = sm.OLS(am["__y"], Xo).fit(cov_type="HAC", cov_kwds=L.hac_kw(h))
                linhas.append({
                    "desfecho": desf, "h": h, "com_brent": com_brent,
                    "c_2sls": bt[i], "ep_2sls": ep[i],
                    "t_2sls": bt[i] / ep[i],
                    "c_ols": float(mo.params["__m"]), "ep_ols": float(mo.bse["__m"]),
                    "observacoes": T})
    return pd.DataFrame(linhas), pd.concat(primeiros, ignore_index=True)


def main():
    print(__doc__)
    t, fe = rodar()
    with pd.ExcelWriter(SAIDA / "onda4_4_paridade_iv.xlsx") as w:
        t.to_excel(w, sheet_name="segundo_estagio", index=False)
        fe.to_excel(w, sheet_name="primeiro_estagio", index=False)

    print("=== PRIMEIRO ESTAGIO: o preco domestico segue a paridade? ===")
    print("(instrumento cheio, sem Brent no modelo)")
    print(fe[(~fe.com_brent) & (fe.h.isin([0, 1, 3]))].pivot_table(
        index="h", columns="regime", values=["coef", "F"]).round(2).to_string())
    print("\n(identificando pelo crack spread, com Brent no modelo)")
    print(fe[(fe.com_brent) & (fe.h.isin([0, 1, 3]))].pivot_table(
        index="h", columns="regime", values="F").round(2).to_string())

    print("\n=== SEGUNDO ESTAGIO: efeito causal do combustivel nos demais precos ===")
    for cb in [False, True]:
        print(f"\ncom_brent = {cb}  "
              f"({'crack spread' if cb else 'variacao cheia do Golfo'})")
        v = t[t.com_brent == cb]
        print(v[["desfecho", "h", "c_ols", "c_2sls", "ep_2sls", "t_2sls",
                 "observacoes"]].to_string(
            index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\ngravado: {SAIDA / 'onda4_4_paridade_iv.xlsx'}")


if __name__ == "__main__":
    main()
