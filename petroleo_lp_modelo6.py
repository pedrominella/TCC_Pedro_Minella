# -*- coding: utf-8 -*-
"""
petroleo_lp_modelo6.py
Modelo 6 - Local Projections para o TCC.

Versão corrigida:
- reconhece a coluna espectativa_inflacao;
- usa Brent em reais como choque principal;
- estima LP principal com 3 defasagens;
- gera robustez com 6 defasagens e deixa 12 lags opcional;
- divide Petrobras em 3 regimes: 2003-2014, 2015-2022 e 2023+;
- estima Wald entre regimes;
- estima LP-IV com Oil Supply News Shock, se a base estiver disponível;
- exporta tabelas e resumo.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

ARQUIVO_IPCA = "IPCA.xlsx"
ARQUIVO_SELIC = "STP-20260429165342557.csv"
ARQUIVO_STRINGENCY = "Stringency_index.csv"
ARQUIVO_OIL_NEWS = "oilSupplyNewsShocks_2025M06.xlsx"

DATA_INICIO = "2003-01-01"
H_MAX = 24
H_PRINCIPAL = 12
LAGS_PRINCIPAL = 3
LAGS_ROBUSTEZ = [6]
RODAR_ROBUSTEZ_12_LAGS = False
HORIZONTES_RESUMO = [3, 6, 12, 24]
CONF = 0.90
Z = stats.norm.ppf(1 - (1 - CONF) / 2)
MIN_OBS = 60

OUT = Path("output_petroleo_lp_modelo6")
TAB = OUT / "tabelas"
RES = OUT / "resumos"
for p in [OUT, TAB, RES]:
    p.mkdir(parents=True, exist_ok=True)


def localizar(nome):
    candidatos = [Path(nome), Path.cwd() / nome]
    try:
        candidatos.append(Path(__file__).resolve().parent / nome)
    except Exception:
        pass
    home = Path.home()
    candidatos += [
        home / "OneDrive" / "Documentos" / "TCC" / nome,
        home / "OneDrive" / "Documentos" / "TCC_python" / nome,
        home / "Documents" / "TCC" / nome,
        home / "Documents" / "TCC_python" / nome,
    ]
    for c in candidatos:
        if c.exists():
            return c
    return None


def norm(s):
    s = str(s).lower().strip()
    for a, b in {"á":"a","à":"a","ã":"a","â":"a","é":"e","ê":"e","í":"i","ó":"o","ô":"o","õ":"o","ú":"u","ç":"c"}.items():
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "", s)


def achar(df, candidatos, obrig=True, nome=""):
    mapa = {norm(c): c for c in df.columns}
    for cand in candidatos:
        if norm(cand) in mapa:
            return mapa[norm(cand)]
    for cand in candidatos:
        nc = norm(cand)
        for k, v in mapa.items():
            if nc and (nc in k or k in nc):
                return v
    if obrig:
        raise ValueError(f"Nao encontrei coluna {nome or candidatos}. Colunas: {list(df.columns)}")
    return None


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def dlog100(s):
    s = num(s).where(num(s) > 0)
    return 100 * (np.log(s) - np.log(s.shift(1)))


def diff_se_precisa(s):
    s = num(s)
    med = s.dropna().median()
    return dlog100(s) if pd.notna(med) and abs(med) > 20 else s


def padronizar(s):
    s = num(s)
    sd = s.std(skipna=True)
    return (s - s.mean(skipna=True)) / sd if pd.notna(sd) and sd != 0 else s * np.nan


def hac(h):
    return max(3, h + 1)


def ic(coef, se):
    return coef - Z * se, coef + Z * se


def lags(df, var, n):
    out = []
    for L in range(1, n + 1):
        c = f"{var}_lag{L}"
        df[c] = df[var].shift(L)
        out.append(c)
    return out


def y_h(df, y, h):
    cols = []
    for j in range(h + 1):
        c = f"{y}_lead{j}"
        df[c] = df[y].shift(-j)
        cols.append(c)
    df[f"y_h{h}"] = df[cols].sum(axis=1, min_count=h + 1)
    return df


def ols_hac(y, X, h):
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac(h)})


def importar_selic():
    arq = localizar(ARQUIVO_SELIC)
    if arq is None:
        return None
    df = pd.read_csv(arq, sep=";", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    cdata = achar(df, ["Data"], True, "data selic")
    cval = [c for c in df.columns if c != cdata][0]
    df[cdata] = pd.to_datetime(df[cdata], dayfirst=True, errors="coerce")
    df[cval] = num(df[cval])
    df = df.dropna(subset=[cdata])
    df["Data"] = df[cdata].dt.to_period("M").dt.to_timestamp()
    print("[OK] Selic diaria importada.")
    return df.groupby("Data", as_index=False)[cval].mean().rename(columns={cval: "selic_diaria_media_mensal"})


def importar_stringency():
    arq = localizar(ARQUIVO_STRINGENCY)
    if arq is None:
        return None
    df = pd.read_csv(arq, sep=";", encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "CountryCode" in df.columns:
        df = df[df["CountryCode"].astype(str).str.upper().eq("BRA")].copy()
    if "Jurisdiction" in df.columns:
        tmp = df[df["Jurisdiction"].astype(str).str.upper().eq("NAT_TOTAL")].copy()
        if not tmp.empty:
            df = tmp
    cdata = achar(df, ["Date", "Data"], True, "data stringency")
    cstr = achar(df, ["StringencyIndex_Average", "StringencyIndex_Average_ForDisplay", "Stringency"], True, "stringency")
    df[cdata] = pd.to_datetime(df[cdata].astype(str), format="%Y%m%d", errors="coerce")
    df[cstr] = num(df[cstr])
    df = df.dropna(subset=[cdata])
    df["Data"] = df[cdata].dt.to_period("M").dt.to_timestamp()
    print("[OK] Stringency importado.")
    return df.groupby("Data", as_index=False)[cstr].mean().rename(columns={cstr: "stringency_externo"})


def parse_m(x):
    if pd.isna(x):
        return pd.NaT
    sx = str(x).strip()
    if "M" in sx:
        a, m = sx.split("M")
        return pd.Timestamp(int(a), int(m), 1)
    return pd.to_datetime(sx, errors="coerce")


def importar_news():
    arq = localizar(ARQUIVO_OIL_NEWS)
    if arq is None:
        return None
    xl = pd.ExcelFile(arq)
    aba = "Monthly" if "Monthly" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(arq, sheet_name=aba)
    df.columns = [str(c).strip() for c in df.columns]
    cdata = achar(df, ["Date", "Data"], True, "data news")
    cnews = achar(df, ["Oil supply news shock", "Oil Supply News Shock"], True, "news")
    df["Data"] = df[cdata].apply(parse_m)
    df["oil_supply_news_shock"] = num(df[cnews])
    df = df.dropna(subset=["Data"])
    df["oil_supply_news_shock_std"] = padronizar(df["oil_supply_news_shock"])
    print("[OK] Oil Supply News Shock importado.")
    return df[["Data", "oil_supply_news_shock", "oil_supply_news_shock_std"]]


def preparar_base():
    arq = localizar(ARQUIVO_IPCA)
    if arq is None:
        raise FileNotFoundError("IPCA.xlsx nao encontrado.")
    df = pd.read_excel(arq)
    df.columns = [str(c).strip() for c in df.columns]
    cdata = achar(df, ["Data", "Date"], True, "data")
    df[cdata] = pd.to_datetime(df[cdata], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[cdata]).copy()
    df["Data"] = df[cdata].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("Data").drop_duplicates("Data", keep="last")

    for ext in [importar_selic(), importar_stringency(), importar_news()]:
        if ext is not None:
            df = df.merge(ext, on="Data", how="left")

    col_pet = achar(df, ["Preco_Barril", "Brent", "Petroleo"], True, "petroleo")
    col_cambio = achar(df, ["Cambio", "Câmbio", "Dolar"], True, "cambio")
    col_ativ = achar(df, ["Atividade", "IBC_BR", "IBC-Br"], True, "atividade")
    col_ipca = achar(df, ["Var_IPCA_Geral", "IPCA_Geral", "IPCA_Geral_nivel"], True, "ipca geral")
    col_trans = achar(df, ["Var_IPCA_Trans", "IPCA_Trans_nivel", "IPCA_Transportes"], True, "ipca transportes")
    col_ref = achar(df, ["Var_GasolinaABrasil_media", "GasolinaABrasil_media_nivel", "GasolinaABrasil_media"], True, "refinaria")
    col_gas = achar(df, ["Var_Gasolina", "Gasolina_nivel", "Gasolina"], True, "gasolina")
    col_eta = achar(df, ["Var_Etanol", "Etanol_nivel", "Etanol"], True, "etanol")
    col_die = achar(df, ["Var_Oleo_diesel", "Oleo_diesel_nivel", "Diesel"], True, "diesel")
    col_sel = achar(df, ["selic_diaria_media_mensal", "Selic.1", "Selic"], False, "selic")
    col_exp = achar(df, ["espectativa_inflacao", "Expectativa_Inflacao", "Expectativa_Inflação", "Focus_IPCA_12m"], False, "expectativa")
    col_str = achar(df, ["stringency_externo", "Stringency", "Stringency_Index"], False, "stringency")

    print("\nColunas usadas:")
    for k, v in {"petroleo": col_pet, "cambio": col_cambio, "expectativa": col_exp, "selic": col_sel, "stringency": col_str}.items():
        print(f"- {k}: {v}")

    df["brent_usd_nivel"] = num(df[col_pet])
    df["cambio_nivel"] = num(df[col_cambio])
    df["brent_brl_nivel"] = df["brent_usd_nivel"] * df["cambio_nivel"]
    df["dln_petroleo_brl"] = dlog100(df["brent_brl_nivel"])
    df["dln_petroleo_usd"] = dlog100(df["brent_usd_nivel"])
    df["dln_cambio"] = dlog100(df["cambio_nivel"])
    df["dln_atividade"] = dlog100(df[col_ativ])
    df["dln_gasolina_refinaria"] = diff_se_precisa(df[col_ref])
    df["dln_gasolina"] = diff_se_precisa(df[col_gas])
    df["dln_etanol"] = diff_se_precisa(df[col_eta])
    df["dln_diesel"] = diff_se_precisa(df[col_die])
    df["ipca_geral_mensal"] = diff_se_precisa(df[col_ipca])
    df["ipca_transporte_mensal"] = diff_se_precisa(df[col_trans])
    df["selic_controle"] = num(df[col_sel]) if col_sel is not None else 0.0
    df["expectativa_controle"] = num(df[col_exp]) if col_exp is not None else 0.0
    df["stringency_controle"] = num(df[col_str]).fillna(0.0) if col_str is not None else 0.0
    print(f"[OK] Expectativa de inflacao usada: {col_exp}" if col_exp else "[AVISO] Expectativa nao encontrada.")

    df["mes"] = df["Data"].dt.month
    dummies = pd.get_dummies(df["mes"], prefix="mes", drop_first=True, dtype=float)
    df = pd.concat([df, dummies], axis=1)
    dummy_cols = list(dummies.columns)

    df["regime_2003_2014"] = ((df["Data"] >= "2003-01-01") & (df["Data"] <= "2014-12-01")).astype(int)
    df["regime_2015_2022"] = ((df["Data"] >= "2015-01-01") & (df["Data"] <= "2022-12-01")).astype(int)
    df["regime_2023_2026"] = (df["Data"] >= "2023-01-01").astype(int)
    df = df[df["Data"] >= pd.to_datetime(DATA_INICIO)].reset_index(drop=True)
    df.to_excel(TAB / "base_transformada_modelo6.xlsx", index=False)
    print(f"[OK] Base final: {df['Data'].min().date()} ate {df['Data'].max().date()} | N={len(df)}")
    return df, dummy_cols


def controles(dummy_cols):
    return ["dln_atividade", "selic_controle", "expectativa_controle", "stringency_controle"] + dummy_cols


def lp(df, y, shock, xctrl, lags_n=3, horizontes=None, nome="lp"):
    horizontes = list(range(H_MAX + 1)) if horizontes is None else horizontes
    base = df.copy()
    sh = shock + "_std"
    base[sh] = padronizar(base[shock])
    regs = []
    regs += lags(base, y, lags_n)
    regs += lags(base, shock, lags_n)
    for c in xctrl:
        regs.append(c)
        if not c.startswith("mes_"):
            regs += lags(base, c, lags_n)
    out = []
    for h in horizontes:
        tmp = y_h(base.copy(), y, h)
        xs = [sh] + [c for c in regs if c in tmp.columns]
        reg = tmp[[f"y_h{h}"] + xs].replace([np.inf, -np.inf], np.nan).dropna()
        if len(reg) < max(MIN_OBS, len(xs) + 10):
            out.append({"modelo": nome, "h": h, "coef": np.nan, "ci_low": np.nan, "ci_high": np.nan, "nobs": len(reg), "impulso": shock, "resposta": y, "lags": lags_n})
            continue
        r = ols_hac(reg[f"y_h{h}"], reg[xs], h)
        coef = r.params.get(sh, np.nan); se = r.bse.get(sh, np.nan); low, high = ic(coef, se)
        out.append({"modelo": nome, "h": h, "coef": coef, "se": se, "pvalor": r.pvalues.get(sh, np.nan), "ci_low": low, "ci_high": high, "significativo_90": bool((low > 0) or (high < 0)), "nobs": int(r.nobs), "impulso": shock, "resposta": y, "lags": lags_n})
    return pd.DataFrame(out)


def lp_regimes(df, y, shock, xctrl, lags_n=3, horizontes=HORIZONTES_RESUMO, nome="regimes"):
    base = df.copy(); sh = shock + "_std"; base[sh] = padronizar(base[shock])
    regs_map = {"2003_2014":"regime_2003_2014", "2015_2022":"regime_2015_2022", "2023_2026":"regime_2023_2026"}
    inters = []
    for rn, rc in regs_map.items():
        it = f"{sh}_x_{rn}"; base[it] = base[sh] * base[rc]; inters.append(it)
    regs = lags(base, y, lags_n) + lags(base, shock, lags_n)
    for c in xctrl:
        regs.append(c)
        if not c.startswith("mes_"):
            regs += lags(base, c, lags_n)
    res, walds = [], []
    for h in horizontes:
        tmp = y_h(base.copy(), y, h); xs = inters + [c for c in regs if c in tmp.columns]
        reg = tmp[[f"y_h{h}"] + xs].replace([np.inf, -np.inf], np.nan).dropna()
        if len(reg) < max(MIN_OBS, len(xs)+10):
            continue
        r = ols_hac(reg[f"y_h{h}"], reg[xs], h)
        for rn in regs_map:
            it = f"{sh}_x_{rn}"; coef = r.params.get(it, np.nan); se = r.bse.get(it, np.nan); low, high = ic(coef, se)
            res.append({"modelo": nome, "impulso": shock, "resposta": y, "regime": rn, "h": h, "coef": coef, "se": se, "pvalor": r.pvalues.get(it, np.nan), "ci_low": low, "ci_high": high, "significativo_90": bool((low>0) or (high<0)), "nobs": int(r.nobs)})
        names = list(r.params.index)
        for a,b in [("2003_2014","2015_2022"),("2015_2022","2023_2026"),("2003_2014","2023_2026")]:
            ia, ib = f"{sh}_x_{a}", f"{sh}_x_{b}"
            if ia in names and ib in names:
                R = np.zeros((1,len(names))); R[0,names.index(ia)] = 1; R[0,names.index(ib)] = -1
                wt = r.wald_test(R, scalar=True)
                walds.append({"modelo": nome, "impulso": shock, "resposta": y, "h": h, "comparacao": f"{a} vs {b}", "pvalor_wald": float(wt.pvalue), "diferenca_coef": float(r.params[ia]-r.params[ib]), "nobs": int(r.nobs)})
    return pd.DataFrame(res), pd.DataFrame(walds)


def lp_iv(df, y, endog, instr, xctrl, lags_n=3, horizontes=HORIZONTES_RESUMO, nome="lpiv"):
    base = df.copy(); end = endog + "_std"; ins = instr + "_std"
    base[end] = padronizar(base[endog]); base[ins] = padronizar(base[instr])
    regs = lags(base, y, lags_n) + lags(base, endog, lags_n) + lags(base, instr, lags_n)
    for c in xctrl:
        regs.append(c)
        if not c.startswith("mes_"):
            regs += lags(base, c, lags_n)
    out, fsout = [], []
    for h in horizontes:
        tmp = y_h(base.copy(), y, h); xs = [c for c in regs if c in tmp.columns]
        reg = tmp[[f"y_h{h}", end, ins] + xs].replace([np.inf, -np.inf], np.nan).dropna()
        if len(reg) < max(MIN_OBS, len(xs)+10):
            continue
        X1 = sm.add_constant(reg[[ins] + xs], has_constant="add")
        fs = sm.OLS(reg[end], X1).fit(cov_type="HC1")
        try:
            ft = fs.f_test(f"{ins}=0"); fstat = float(ft.fvalue); fp = float(ft.pvalue)
        except Exception:
            fstat = float(fs.tvalues.get(ins, np.nan)**2); fp = float(fs.pvalues.get(ins, np.nan))
        reg = reg.copy(); reg[end + "_hat"] = fs.predict(X1)
        r = ols_hac(reg[f"y_h{h}"], reg[[end + "_hat"] + xs], h)
        coef = r.params.get(end + "_hat", np.nan); se = r.bse.get(end + "_hat", np.nan); low, high = ic(coef, se)
        out.append({"modelo": nome, "impulso_instrumentado": endog, "instrumento": instr, "resposta": y, "h": h, "coef_iv": coef, "se_iv": se, "pvalor_iv": r.pvalues.get(end + "_hat", np.nan), "ci_low": low, "ci_high": high, "significativo_90": bool((low>0) or (high<0)), "first_stage_f": fstat, "first_stage_pvalor": fp, "nobs": int(r.nobs)})
        fsout.append({"modelo": nome, "resposta": y, "h": h, "first_stage_f": fstat, "first_stage_pvalor": fp, "r2_primeiro_estagio": fs.rsquared, "nobs": int(fs.nobs)})
    return pd.DataFrame(out), pd.DataFrame(fsout)


def salvar(lista, nome):
    if not lista:
        return pd.DataFrame()
    df = pd.concat(lista, ignore_index=True)
    df.to_excel(TAB / f"{nome}.xlsx", index=False)
    df.to_csv(TAB / f"{nome}.csv", index=False, encoding="utf-8-sig")
    return df


def executar():
    print("="*90); print("MODELO 6 - LOCAL PROJECTIONS"); print("="*90)
    df, dummy_cols = preparar_base(); xctrl = controles(dummy_cols)
    pet = "dln_petroleo_brl"
    fuels = ["dln_diesel", "dln_gasolina", "dln_etanol", "dln_gasolina_refinaria"]
    ipcas = ["ipca_geral_mensal", "ipca_transporte_mensal"]

    res_lp = []
    rel = [(pet, f) for f in fuels] + [(f, i) for f in fuels for i in ipcas] + [(pet, i) for i in ipcas]
    print("\nEstimando LP principal com 3 defasagens...")
    for sh, y in rel:
        print(f"- {sh} -> {y}")
        res_lp.append(lp(df, y, sh, xctrl, LAGS_PRINCIPAL, None, f"principal_{sh}_para_{y}"))
    lp_df = salvar(res_lp, "resultados_lp_principal_3_lags")
    lp_df[lp_df["h"].isin([3,6,12,24])].to_excel(TAB / "tabela_lp_principal_h3_h6_h12_h24.xlsx", index=False)
    lp_df[lp_df["h"].eq(H_PRINCIPAL)].to_excel(TAB / "ranking_lp_h12.xlsx", index=False)

    print("\nEstimando robustez com 6 defasagens...")
    res_rob = []
    for sh,y in [(pet,"dln_diesel"),(pet,"dln_gasolina_refinaria"),(pet,"ipca_transporte_mensal"),("dln_gasolina_refinaria","ipca_transporte_mensal")]:
        print(f"- robustez: {sh} -> {y}")
        res_rob.append(lp(df, y, sh, xctrl, 6, HORIZONTES_RESUMO, f"robustez_6_{sh}_para_{y}"))
    salvar(res_rob, "resultados_lp_robustez_6_lags")

    print("\nEstimando 3 regimes Petrobras...")
    res_reg, res_wald = [], []
    for sh,y in [(pet,"dln_diesel"),(pet,"dln_gasolina_refinaria"),(pet,"ipca_transporte_mensal"),(pet,"ipca_geral_mensal"),("dln_gasolina_refinaria","ipca_transporte_mensal"),("dln_gasolina","ipca_transporte_mensal"),("dln_etanol","ipca_transporte_mensal")]:
        print(f"- regimes: {sh} -> {y}")
        a,b = lp_regimes(df, y, sh, xctrl, 3, HORIZONTES_RESUMO, f"regimes_{sh}_para_{y}")
        res_reg.append(a); res_wald.append(b)
    salvar(res_reg, "resultados_lp_regimes_petrobras_3_periodos")
    salvar(res_wald, "testes_wald_regimes_petrobras_3_periodos")

    if "oil_supply_news_shock_std" in df.columns:
        print("\nEstimando LP-IV com Oil Supply News Shock...")
        res_iv, res_fs = [], []
        for y in ["dln_diesel", "dln_gasolina_refinaria", "ipca_transporte_mensal", "ipca_geral_mensal"]:
            print(f"- LP-IV: {pet} -> {y}")
            a,b = lp_iv(df, y, pet, "oil_supply_news_shock_std", xctrl, 3, HORIZONTES_RESUMO, f"lpiv_{pet}_para_{y}")
            res_iv.append(a); res_fs.append(b)
        salvar(res_iv, "resultados_lp_iv_oil_supply_news")
        salvar(res_fs, "primeiro_estagio_lp_iv_oil_supply_news")

    texto = "Modelo 6 finalizado. Choque principal: Brent em reais. LP principal: 3 defasagens. Regimes Petrobras: 2003-2014, 2015-2022 e 2023+. LP-IV com Oil Supply News Shock apenas como robustez.\n"
    (RES / "resumo_modelo6_para_tcc.txt").write_text(texto, encoding="utf-8")
    print("\nMODELO 6 FINALIZADO")
    print(f"Arquivos salvos em: {OUT.resolve()}")


if __name__ == "__main__":
    executar()
