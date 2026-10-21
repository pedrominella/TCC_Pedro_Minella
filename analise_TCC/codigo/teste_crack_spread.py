# -*- coding: utf-8 -*-
"""
CRACK SPREAD COMO INSTRUMENTO: GASOLINA E DIESEL

Ideia. O furacao no Golfo do Mexico atinge REFINO, nao producao de cru. Ele move
a margem entre derivado e petroleo, e nao o preco do petroleo. Subtraindo o Brent
da cotacao do derivado no Golfo sobra exatamente essa margem: choque no preco do
combustivel ORTOGONAL ao choque de petroleo que ja e regressor do modelo.

Por que dois produtos. A margem da gasolina nao move o diesel domestico (F de
0,04 no teste anterior), e isso nao e falha do metodo, e produto errado. Gasolina
move consumidor final; DIESEL move custo de producao, via frete. Se ha segunda
volta detectavel, o encadeamento e:

    margem do diesel -> diesel na bomba -> frete -> preco de servico e de bem
                                                    que NAO e combustivel

Construcao, para cada produto:
    crack    = derivado no Golfo (US$/galao x 42) - Brent (US$/barril)
    cs_dif   variacao mensal do crack, padronizada
    cs_ort   residuo de dlog(derivado) sobre dlog(Brent), ortogonal por construcao

O que se reporta:
  1. primeiro estagio, o instrumento move o preco domestico do combustivel?
  2. forma reduzida sobre precos que NAO sao combustivel, que dispensa primeiro
     estagio e e o objeto da segunda volta.

Fontes. Gasolina: EIA, EER_EPMRU_PF4_RGC_DPG, mensal desde 1986.
        Diesel ULSD: EIA, EER_EPD2DXL0_PF4_RGC_DPG, mensal desde jun/2006.
O ULSD so vira padrao nos EUA em 2006, entao a serie do diesel comeca ali. Como
o teste que interessa e no regime de paridade, isso nao custa nada.

Uso: python teste_crack_spread.py
"""
import sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import lp_base_nova as L

RAIZ = Path(__file__).resolve().parent.parent.parent
DADOS = RAIZ / "base_de_dados"
GAL = 42.0
# Data do anuncio da PPI corrigida em 01/10/2026: a Petrobras anunciou a politica em 14/10/2016 e os
# novos precos valeram a partir de 15/10/2016 (Agencia Brasil, 14/10/2016). Nos dados mensais, out/2016.
CORTE = "2016-10-01"

PRODUTOS = {
    "gasolina": dict(arq=DADOS / "EER_EPMRU_PF4_RGC_DPGm.xls", endog="gasolina_consumidor"),
    "diesel":   dict(arq=DADOS / "EER_EPD2DXL0_PF4_RGC_DPGm.xls", endog="diesel"),
}


def serie_eia(arq):
    d = pd.read_excel(arq, sheet_name="Data 1", skiprows=2)
    d.columns = ["Data", "usd_gal"]
    d = d.dropna()
    d["Data"] = pd.to_datetime(d["Data"]).dt.to_period("M").dt.to_timestamp()
    d["usd_bbl"] = pd.to_numeric(d["usd_gal"], errors="coerce") * GAL
    return d[["Data", "usd_bbl"]]


def base():
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["pos"] = L.regime(b.Data, CORTE)
    dlb = 100.0 * np.log(pd.to_numeric(b["Preco_Barril"])).diff()

    print("CRACK SPREAD, derivado no Golfo menos Brent, US$/barril")
    for nome, cfg in PRODUTOS.items():
        s = serie_eia(cfg["arq"])
        b = b.merge(s.rename(columns={"usd_bbl": f"bbl_{nome}"}), on="Data", how="left")
        crack = b[f"bbl_{nome}"] - b["Preco_Barril"]
        b[f"cs_dif_{nome}"] = crack.diff()
        dlp = 100.0 * np.log(b[f"bbl_{nome}"]).diff()
        m = pd.concat([dlp, dlb], axis=1).dropna()
        r = sm.OLS(m.iloc[:, 0], sm.add_constant(m.iloc[:, 1])).fit()
        b[f"cs_ort_{nome}"] = np.nan
        b.loc[m.index, f"cs_ort_{nome}"] = r.resid.values
        for c in [f"cs_dif_{nome}", f"cs_ort_{nome}"]:
            b[c] = (b[c] - b[c].mean()) / b[c].std()
        val = b[b[f"bbl_{nome}"].notna()]
        print("  %-9s %s a %s | crack medio %5.1f | R2 sobre o Brent %.3f | "
              "correlacao de cs_ort com o choque %+.3f"
              % (nome, val.Data.min().strftime("%Y-%m"), val.Data.max().strftime("%Y-%m"),
                 crack.mean(), r.rsquared,
                 b[[f"cs_ort_{nome}", "choque_petroleo"]].dropna().corr().iloc[0, 1]))

    # series de comercializaveis e nao comercializaveis, ausentes do carregador
    for col, nome in [("Var_IPCA_Comer", "comercializaveis"),
                      ("Var_IPCA_NComer", "naocomercializaveis")]:
        v = pd.to_numeric(b[col], errors="coerce")
        b[f"dlog_{nome}"] = v
        b[f"ln_{nome}"] = v.fillna(0).cumsum()
    chk = (b["dlog_ipca_geral"] - (b["ln_ipca_geral"] - b["ln_ipca_geral"].shift(1))).abs().max()
    print("  conferencia da identidade ln/dlog em serie existente: desvio %.1e" % chk)
    return b, dum, pand


def desenho(b, dum, pand, d, alvo, h):
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{alvo}"].shift(lag)
    c = []
    for lag in range(1, L.LAGS + 1):
        c += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
             [f"{x}_lag{lag}" for x in L.CONTROLES]
    return c + ["choque_petroleo"] + dum + pand(d, h)


def estima(b, dum, pand, inst, alvo, amostra, h=0):
    d = b.copy()
    c = desenho(b, dum, pand, d, alvo, h)
    d["__y"] = L.y_acum(d, alvo, h)
    am = d[["pos", "__y", inst] + c].replace([np.inf, -np.inf], np.nan).dropna()
    am = am[am.pos == 1] if amostra == "pos" else am[am.pos == 0] if amostra == "pre" else am
    if len(am) < 40:
        return None
    r = sm.OLS(am["__y"], sm.add_constant(am[[inst] + c], has_constant="add")).fit(
        cov_type="HAC", cov_kwds=L.hac_kw(h))
    f = float(r.wald_test("%s = 0" % inst, scalar=True).statistic)
    return dict(F=f, coef=r.params[inst], ep=r.bse[inst], p=r.pvalues[inst], n=int(r.nobs))


def est(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def main():
    b, dum, pand = base()

    print("\n" + "=" * 84)
    print("1. PRIMEIRO ESTAGIO   o instrumento move o preco domestico do combustivel?")
    print("   F robusto sobre o instrumento excluido, h = 0, controles do modelo principal")
    print("=" * 84)
    print("%-10s %-22s %-9s %8s %10s %8s %6s" %
          ("produto", "endogena", "amostra", "F", "coef", "ep", "n"))
    for nome, cfg in PRODUTOS.items():
        for am, rot in [("pos", "pos-2016"), ("pre", "pre-2016")]:
            o = estima(b, dum, pand, f"cs_ort_{nome}", cfg["endog"], am)
            if o is None:
                continue
            marca = "   <-- F > 10" if o["F"] > 10 else ""
            print("%-10s %-22s %-9s %8.2f %10.3f %8.3f %6d%s"
                  % (nome if am == "pos" else "", cfg["endog"] if am == "pos" else "",
                     rot, o["F"], o["coef"], o["ep"], o["n"], marca))
        print()

    print("=" * 84)
    print("2. FORMA REDUZIDA   o instrumento move precos que NAO sao combustivel?")
    print("   Dispensa primeiro estagio. E o objeto da segunda volta.")
    print("=" * 84)
    alvos = ["ipca_transportes", "naocomercializaveis", "comercializaveis", "ipca_geral"]
    for nome in PRODUTOS:
        print("\n  instrumento: margem de refino do %s" % nome.upper())
        print("  %-22s %-3s %22s %22s" % ("desfecho", "h", "pos-2016", "pre-2016"))
        for alvo in alvos:
            for h in [0, 1, 3]:
                cel = []
                for am in ["pos", "pre"]:
                    o = estima(b, dum, pand, f"cs_ort_{nome}", alvo, am, h)
                    cel.append("      n/d" if o is None
                               else "%8.4f%-3s(%.3f)" % (o["coef"], est(o["p"]), o["p"]))
                print("  %-22s %-3d %22s %22s"
                      % (alvo if h == 0 else "", h, cel[0], cel[1]))
            print()

    print("Leitura. F abaixo de 10 e instrumento fraco pelo criterio usual; abaixo de 5 a")
    print("via do IV nao se sustenta. Na forma reduzida, o que interessa e o CONTRASTE")
    print("entre regimes: sinal no pos e nada no pre e a assinatura da regra de paridade.")


if __name__ == "__main__":
    main()
