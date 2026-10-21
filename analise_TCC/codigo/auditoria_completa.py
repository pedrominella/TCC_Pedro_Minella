# -*- coding: utf-8 -*-
"""
AUDITORIA COMPLETA DO TCC  --  30 CHECAGENS

Complementa auditoria_360.py, que cobre o documento (familia E). Este arquivo
cobre dados, especificacao, identificacao e inferencia, que sao onde o erro
silencioso mora.

Cada checagem devolve OK, ATENCAO ou CRITICO e um detalhe com o numero que
sustenta o veredito. Nenhuma checagem diz "esta certo" sem mostrar a conta.

  A  DADOS               1..8    integridade, buracos, saltos, unidades
  B  ESPECIFICACAO       9..15   graus de liberdade, colinearidade, estabilidade
  C  IDENTIFICACAO      16..22   quanto o resultado depende de escolhas e de casos
  D  INFERENCIA         23..27   residuo, testes multiplos, erro-padrao
  E  DOCUMENTO          28..30   delega a auditoria_360

Uso:
  python auditoria_completa.py              todas
  python auditoria_completa.py --rapido     pula jackknife e deixa-um-ano
  python auditoria_completa.py 13 18 25     so essas
"""
import sys, re, time, warnings, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lp_base_nova as L
import onda5_alternativos as A

ART = AQUI.parent
RES = ART / "resultados"
ALVO, H_ALVO = "ipca_geral", 0
RESULTADOS = []


def reg(n, familia, nome, veredito, detalhe):
    RESULTADOS.append({"n": n, "familia": familia, "checagem": nome,
                       "veredito": veredito, "detalhe": detalhe})
    cor = {"OK": "[ok]      ", "ATENCAO": "[ATENCAO] ", "CRITICO": "[CRITICO] "}[veredito]
    print("  %2d %s%-46s %s" % (n, cor, nome, detalhe))


def _dados():
    if not hasattr(_dados, "c"):
        b, dum = A.base()
        _dados.c = (b, dum)
    return _dados.c


def _ajuste(nome=ALVO, h=H_ALVO, b=None, dum=None):
    b0, d0 = _dados()
    y, X, cols = A.monta(b if b is not None else b0, dum if dum is not None else d0, nome, h)
    return y, X, cols


# ══════════════════════════════════════════════ A. DADOS
def c01():
    b, _ = _dados()
    ruim = []
    for nome in list(L.VARIAVEIS) + ["Cambio", "Atividade", "Expectativa_inflacao"]:
        col = f"dlog_{nome}" if nome in L.VARIAVEIS else nome
        if col not in b.columns:
            continue
        s = pd.to_numeric(b[col], errors="coerce")
        v = s.notna()
        if not v.any():
            continue
        miolo = int(s.loc[v.idxmax():v[::-1].idxmax()].isna().sum())
        if miolo:
            ruim.append("%s(%d)" % (nome, miolo))
    reg(1, "A", "buracos no miolo das series",
        "CRITICO" if ruim else "OK",
        ", ".join(ruim) if ruim else "nenhum NaN interior em 9 series")


def c02():
    fontes = [f for f in AQUI.glob("*.py")
              if not f.name.startswith("auditoria")
              and "fillna" in f.read_text(encoding="utf-8", errors="ignore")]
    det = []
    for f in fontes:
        n = len(re.findall(r"fillna\(", f.read_text(encoding="utf-8", errors="ignore")))
        det.append("%s(%d)" % (f.name, n))
    reg(2, "A", "preenchimento silencioso (fillna)",
        "ATENCAO" if fontes else "OK",
        (", ".join(det) + " -- confira se so toca a 1a diferenca") if fontes else "nenhum")


def c03():
    b, _ = _dados()
    d = pd.to_datetime(b.Data).sort_values()
    dup = int(d.duplicated().sum())
    esperado = pd.date_range(d.min(), d.max(), freq="MS")
    falta = len(set(esperado) - set(d))
    reg(3, "A", "datas duplicadas ou meses faltando",
        "CRITICO" if (dup or falta) else "OK",
        "duplicadas %d, faltando %d, de %s a %s"
        % (dup, falta, d.min().strftime("%b/%Y"), d.max().strftime("%b/%Y")))


def c04():
    b, _ = _dados()
    piores = []
    for nome in L.VARIAVEIS:
        s = pd.to_numeric(b[f"dlog_{nome}"], errors="coerce").dropna()
        z = (s - s.mean()) / s.std()
        m = z.abs().max()
        if m > 5:
            piores.append("%s %.1f" % (nome, m))
    reg(4, "A", "saltos implausiveis (|z| > 5 d.p.)",
        "ATENCAO" if piores else "OK",
        ("; ".join(piores) + " -- confira se sao eventos reais") if piores
        else "nenhuma variacao acima de 5 desvios")


def c05():
    b, _ = _dados()
    faixa = {"Preco_Barril": (5, 250), "Cambio": (1, 12), "Atividade": (40, 200),
             "Expectativa_inflacao": (0, 30), "meta_selic_sgs432": (0, 60)}
    ruim = []
    for c, (lo, hi) in faixa.items():
        if c not in b.columns:
            continue
        s = pd.to_numeric(b[c], errors="coerce")
        if s.min() < lo or s.max() > hi:
            ruim.append("%s [%.1f, %.1f]" % (c, s.min(), s.max()))
    reg(5, "A", "unidades e escala dentro do plausivel",
        "CRITICO" if ruim else "OK",
        "; ".join(ruim) if ruim else "5 series dentro das faixas esperadas")


def c06():
    b, _ = _dados()
    ini, fim = pd.to_datetime(b.Data).min(), pd.to_datetime(b.Data).max()
    esp_i, esp_f = pd.Timestamp(L.DATA_INICIO), pd.Timestamp(L.DATA_FIM)
    ok = ini <= esp_i and fim >= esp_f
    reg(6, "A", "cobertura temporal cobre a amostra declarada",
        "OK" if ok else "CRITICO",
        "%s a %s (declarado %s a %s)" % (ini.strftime("%b/%Y"), fim.strftime("%b/%Y"),
                                         esp_i.strftime("%b/%Y"), esp_f.strftime("%b/%Y")))


def c07():
    b, _ = _dados()
    congeladas = []
    for nome in L.VARIAVEIS:
        s = pd.to_numeric(b[f"dlog_{nome}"], errors="coerce")
        z = (s.fillna(1) == 0).astype(int)
        run = maxrun = 0
        for v in z:
            run = run + 1 if v else 0
            maxrun = max(maxrun, run)
        if maxrun >= 4:
            congeladas.append("%s %d meses" % (nome, maxrun))
    reg(7, "A", "series congeladas (variacao zero repetida)",
        "ATENCAO" if congeladas else "OK",
        ("; ".join(congeladas) + " -- preco administrado ou dado faltando?")
        if congeladas else "nenhuma sequencia de 4+ meses sem variacao")


def c08():
    b, _ = _dados()
    s = pd.to_numeric(b["Atividade"], errors="coerce")
    d = 100 * np.log(s).diff()
    t = pd.DataFrame({"m": pd.to_datetime(b.Data).dt.month, "d": d}).dropna()
    X = pd.get_dummies(t.m, prefix="m", drop_first=True, dtype=float)
    r = sm.OLS(t.d, sm.add_constant(X)).fit()
    tem = r.f_pvalue < 0.05
    reg(8, "A", "atividade ja vem dessazonalizada",
        "ATENCAO" if tem else "OK",
        "F=%.2f p=%.3f -> %s" % (r.fvalue, r.f_pvalue,
                                 "TEM sazonalidade" if tem else "sem sazonalidade, coerente"))


# ══════════════════════════════════════════════ B. ESPECIFICACAO
def c09():
    y, X, _ = _ajuste()
    b, _ = _dados()
    npos = int((b.regime_pos == 1).sum())
    razao = X.shape[0] / X.shape[1]
    reg(9, "B", "graus de liberdade (n / K)",
        "ATENCAO" if razao < 5 else "OK",
        "n=%d K=%d razao=%.1f; regime posterior tem %d obs" % (X.shape[0], X.shape[1], razao, npos))


def c10():
    y, X, _ = _ajuste()
    Xs = X / np.maximum(np.abs(X).max(axis=0), 1e-12)
    c = float(np.linalg.cond(Xs))
    reg(10, "B", "colinearidade (numero de condicao)",
        "ATENCAO" if c > 1000 else "OK", "cond = %.0f (limiar 1000)" % c)


def c11():
    from statsmodels.tsa.stattools import adfuller
    b, _ = _dados()
    lim = []
    for c in ["selic", "expectativa", "dlog_ipca_geral", "choque_petroleo"]:
        x = pd.to_numeric(b[c], errors="coerce").dropna()
        p = adfuller(x, autolag="AIC")[1]
        if p > 0.05:
            lim.append("%s p=%.3f" % (c, p))
    reg(11, "B", "grau de integracao das series",
        "ATENCAO" if lim else "OK",
        ("no limite: " + "; ".join(lim) + " -- justifica a defasagem aumentada")
        if lim else "todas rejeitam raiz unitaria")


def c12():
    b, dum = _dados()
    y, X, cols = _ajuste()
    pos = X[:, cols.index("regime_pos")]
    inter = X[:, [cols.index(c) for c in dum]] * pos[:, None]
    X2 = np.column_stack([X, inter])
    keep = X2.std(axis=0) > 0
    r2 = sm.OLS(y, X2[:, keep]).fit()
    idx = [i for i in range(X.shape[1], X2.shape[1]) if keep[i]]
    pos_map = np.cumsum(keep) - 1
    R = np.zeros((len(idx), keep.sum()))
    for k, i in enumerate(idx):
        R[k, pos_map[i]] = 1
    f = r2.f_test(R)
    muda = float(f.pvalue) < 0.05
    # o que importa: o coeficiente muda?
    b1 = sm.OLS(y, X).fit().params[cols.index("choque_pos")]
    b2 = r2.params[pos_map[cols.index("choque_pos")]]
    reg(12, "B", "sazonalidade estavel entre regimes",
        "ATENCAO" if muda else "OK",
        "F=%.2f p=%.3f; coeficiente muda de %.4f para %.4f (%+.4f)"
        % (float(f.fvalue), float(f.pvalue), b1, b2, b2 - b1))


def c13():
    y, X, cols = _ajuste()
    r = sm.OLS(y, X).fit()
    pos = X[:, cols.index("regime_pos")].astype(bool)
    v0, v1 = r.resid[~pos].var(ddof=1), r.resid[pos].var(ddof=1)
    F = v1 / v0
    p = 2 * min(stats.f.cdf(F, pos.sum() - 1, (~pos).sum() - 1),
                1 - stats.f.cdf(F, pos.sum() - 1, (~pos).sum() - 1))
    reg(13, "B", "variancia residual estavel entre regimes",
        "ATENCAO" if p < 0.05 else "OK",
        "var pos/pre = %.2f, p=%.4f -- HAC cobre a inferencia; vale reportar" % (F, p))


def c14():
    y, X, cols = _ajuste()
    zero = [c for i, c in enumerate(cols) if X[:, i].std() == 0 and c != "const"]
    reg(14, "B", "colunas degeneradas na matriz de desenho",
        "CRITICO" if zero else "OK",
        (", ".join(zero[:6]) + " -- remova antes de estimar") if zero
        else "nenhuma coluna sem variacao")


def c15():
    b, dum = _dados()
    vals = []
    orig = L.LAGS
    for p in [1, 2, 3, 4, 6]:
        L.LAGS = p
        try:
            y, X, cols = A.monta(b, dum, ALVO, H_ALVO)
            vals.append((p, sm.OLS(y, X).fit().params[cols.index("choque_pos")]))
        except Exception:
            pass
    L.LAGS = orig
    v = [x for _, x in vals]
    amp = max(v) - min(v)
    reg(15, "B", "sensibilidade ao numero de defasagens",
        "ATENCAO" if amp > 0.05 else "OK",
        "p=1..6 -> " + ", ".join("%.3f" % x for x in v) + " (amplitude %.3f)" % amp)


# ══════════════════════════════════════════════ C. IDENTIFICACAO
def c16():
    b, dum = _dados()
    out = []
    for m in [-6, -3, 0, 3, 6]:
        d = pd.Timestamp(L.DATA_CORTE_FINAL) + pd.DateOffset(months=m)
        bb = b.copy()
        bb["regime_pos"] = L.regime(bb.Data, d)
        bb["choque_pre"] = bb["choque_petroleo"] * (1 - bb["regime_pos"])
        bb["choque_pos"] = bb["choque_petroleo"] * bb["regime_pos"]
        y, X, cols = A.monta(bb, dum, ALVO, H_ALVO)
        out.append(sm.OLS(y, X).fit().params[cols.index("choque_pos")])
    amp = max(out) - min(out)
    reg(16, "C", "sensibilidade a data de corte (+-6 meses)",
        "ATENCAO" if amp > 0.05 else "OK",
        ", ".join("%.3f" % x for x in out) + " (amplitude %.3f)" % amp)


def c17(rapido=False):
    if rapido:
        reg(17, "C", "jackknife de observacao", "OK", "pulado (--rapido)"); return
    y, X, cols = _ajuste()
    ip = cols.index("choque_pos")
    base = sm.OLS(y, X).fit().params[ip]
    dif = np.array([sm.OLS(np.delete(y, i), np.delete(X, i, 0)).fit().params[ip] - base
                    for i in range(len(y))])
    pior = dif[np.argmax(np.abs(dif))]
    inverte = bool(np.any(base + dif <= 0))
    reg(17, "C", "nenhuma observacao isolada sustenta o resultado",
        "CRITICO" if inverte else ("ATENCAO" if abs(pior / base) > 0.25 else "OK"),
        "maior efeito %+.1f%%; sinal inverte: %s" % (100 * pior / base, inverte))


def c18(rapido=False):
    if rapido:
        reg(18, "C", "deixa-um-ano-de-fora", "OK", "pulado (--rapido)"); return
    b, dum = _dados()
    y, X, cols = _ajuste()
    ip = cols.index("choque_pos")
    base = sm.OLS(y, X).fit().params[ip]
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{ALVO}"].shift(lag)
    pand = L.controles_pandemia(d, H_ALVO, modo="janela")
    regs = ["choque_pre", "choque_pos", "regime_pos"]
    for lag in range(1, L.LAGS + 1):
        regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                [f"{c}_lag{lag}" for c in L.CONTROLES]
    regs += list(dum) + pand
    d["__y"] = L.y_acum(d, ALVO, H_ALVO)
    am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
    piores = []
    for ano in range(2017, 2026):
        m = am.Data.dt.year != ano
        Xs = sm.add_constant(am.loc[m, regs].astype(float))
        Xs = Xs.loc[:, Xs.std() > 0]
        v = sm.OLS(am.loc[m, "__y"].astype(float), Xs).fit().params.get("choque_pos", np.nan)
        piores.append((ano, 100 * (v - base) / base))
    ano, pct = max(piores, key=lambda x: abs(x[1]))
    reg(18, "C", "deixa-um-ano-de-fora",
        "ATENCAO" if abs(pct) > 25 else "OK",
        "ano mais influente: %d (%+.1f%%); demais abaixo de %.0f%%"
        % (ano, pct, max(abs(p) for a, p in piores if a != ano)))


def c19():
    b, dum = _dados()
    y, X, cols = _ajuste()
    ip = [i for i, c in enumerate(cols) if c.startswith("pand_")]
    Xs = X[:, [i for i in range(len(cols)) if i not in ip]]
    lev = np.diag(Xs @ np.linalg.pinv(Xs.T @ Xs) @ Xs.T)
    med = lev.mean()
    alto = int((lev > 3 * med).sum())
    reg(19, "C", "pontos de alavanca (sem as dummies de pandemia)",
        "ATENCAO" if alto else "OK",
        "%d obs com alavancagem > 3x a media (%.3f); maxima %.3f" % (alto, med, lev.max()))


def c20():
    b, _ = _dados()
    d = b[["choque_petroleo", "regime_pos"]].dropna()
    p0 = (d[d.regime_pos == 0].choque_petroleo > 0).mean()
    p1 = (d[d.regime_pos == 1].choque_petroleo > 0).mean()
    reg(20, "C", "composicao do choque entre regimes",
        "ATENCAO" if abs(p1 - p0) > 0.15 else "OK",
        "altas: %.1f%% antes, %.1f%% depois (diferenca %.1f p.p.)"
        % (100 * p0, 100 * p1, 100 * (p1 - p0)))


def c21():
    b, dum = _dados()
    bb = b.copy()
    bb["__ch"] = bb["choque_petroleo"].shift(-3)      # choque do futuro: placebo
    bb["choque_pre"] = bb["__ch"] * (1 - bb["regime_pos"])
    bb["choque_pos"] = bb["__ch"] * bb["regime_pos"]
    y, X, cols = A.monta(bb, dum, ALVO, H_ALVO)
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds=L.hac_kw(H_ALVO))
    p = r.pvalues[cols.index("choque_pos")]
    reg(21, "C", "placebo: choque futuro nao deve explicar o passado",
        "CRITICO" if p < 0.10 else "OK",
        "coef %.4f, p=%.3f" % (r.params[cols.index("choque_pos")], p))


def c22():
    f = RES / "modelo_11_alternativos" / "robusta_lad.xlsx"
    if not f.exists():
        reg(22, "C", "estimadores alternativos concordam", "ATENCAO", "rode onda5_alternativos"); return
    d = pd.read_excel(f)
    z = d[(d.variavel == ALVO) & (d.h == H_ALVO) & (d.pandemia == "com dummies")]
    v = dict(zip(z.estimador, z.coef_pos))
    amp = max(v.values()) - min(v.values())
    reg(22, "C", "estimadores alternativos concordam",
        "ATENCAO" if amp > 0.08 else "OK",
        ", ".join("%s %.3f" % (k, x) for k, x in v.items()) + " (amplitude %.3f)" % amp)


# ══════════════════════════════════════════════ D. INFERENCIA
def c23():
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    y, X, cols = _ajuste(h=0)
    r = sm.OLS(y, X).fit()
    lm, p, _, _ = acorr_breusch_godfrey(r, nlags=2)
    reg(23, "D", "autocorrelacao residual em h=0 (nao ha induzida)",
        "ATENCAO" if p < 0.05 else "OK",
        "Breusch-Godfrey(2) p=%.3f" % p)


def c24():
    y, X, _ = _ajuste()
    r = sm.OLS(y, X).fit()
    jb = stats.jarque_bera(r.resid)
    k = stats.kurtosis(r.resid, fisher=False)
    reg(24, "D", "normalidade dos residuos",
        "ATENCAO" if jb[1] < 0.05 else "OK",
        "JB p=%.4f, curtose %.2f -- inferencia e assintotica/bootstrap" % (jb[1], k))


def c25():
    from statsmodels.stats.multitest import multipletests
    f = RES / "onda3_3_regimes_data_escolhida.xlsx"
    if not f.exists():
        reg(25, "D", "correcao para testes multiplos", "ATENCAO", "planilha ausente"); return
    d = pd.read_excel(f)
    d = d[d.data_corte.astype(str).str.startswith("2017-03")]
    p = d.p_pos.values
    sem = int((p < 0.10).sum())
    bh = int(multipletests(p, alpha=0.10, method="fdr_bh")[0].sum())
    holm = int(multipletests(p, alpha=0.10, method="holm")[0].sum())
    reg(25, "D", "correcao para testes multiplos (%d testes)" % len(p),
        "ATENCAO",
        "sem correcao %d, Benjamini-Hochberg %d, Holm %d -- reporte BH" % (sem, bh, holm))


def c26():
    y, X, cols = _ajuste()
    ip = cols.index("choque_pos")
    m = sm.OLS(y, X)
    a = m.fit(cov_type="HAC", cov_kwds=L.hac_kw(H_ALVO))
    c = m.fit(cov_type="HC1")
    e = m.fit(cov_type="HAC", cov_kwds={"maxlags": 2 * (H_ALVO + 1), "use_correction": True})
    eps = [a.bse[ip], c.bse[ip], e.bse[ip]]
    amp = (max(eps) - min(eps)) / min(eps)
    reg(26, "D", "escolha do erro-padrao muda a conclusao?",
        "ATENCAO" if amp > 0.25 else "OK",
        "HAC %.4f, EHW %.4f, HAC2 %.4f (dispersao %.1f%%)" % (*eps, 100 * amp))


def c27():
    # "fixa semente" inclui a constante nomeada (SEMENTE = 20260830; default_rng(SEMENTE)),
    # que era o padrao usado na pasta e que a primeira versao desta checagem nao via.
    USA = r"np\.random\.(?!default_rng)|default_rng\(|\.choice\(|permutation\("
    FIXA = r"seed|SEMENTE|default_rng\(\s*[A-Za-z_0-9]"
    falta = []
    for f in sorted(AQUI.glob("*.py")):
        if f.name.startswith("auditoria"):
            continue
        s = f.read_text(encoding="utf-8", errors="ignore")
        if re.search(USA, s) and not re.search(FIXA, s):
            falta.append(f.name)
    reg(27, "D", "sementes aleatorias fixadas",
        "CRITICO" if falta else "OK",
        ", ".join(falta) if falta else "todos os scripts que sorteiam fixam semente")


# ══════════════════════════════════════════════ E. DOCUMENTO
def c28_30():
    r = subprocess.run([sys.executable, str(AQUI / "auditoria_360.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    saida = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"RESUMO: (\d+) achado", saida)
    ncrit = int(m.group(1)) if m else -1
    for n, nome in [(28, "rastreabilidade dos numeros do documento"),
                    (29, "cadeia de amostra figura -> script -> planilha"),
                    (30, "cronologia, remissoes, listas e travessoes")]:
        reg(n, "E", nome, "OK" if ncrit == 0 else "CRITICO",
            "auditoria_360: %d critico(s)" % ncrit if ncrit >= 0 else "nao consegui rodar")


# ══════════════════════════════════════════════
FAMILIA = {"A": "DADOS", "B": "ESPECIFICACAO", "C": "IDENTIFICACAO",
           "D": "INFERENCIA", "E": "DOCUMENTO"}
CHECAGENS = {1: c01, 2: c02, 3: c03, 4: c04, 5: c05, 6: c06, 7: c07, 8: c08,
             9: c09, 10: c10, 11: c11, 12: c12, 13: c13, 14: c14, 15: c15,
             16: c16, 17: c17, 18: c18, 19: c19, 20: c20, 21: c21, 22: c22,
             23: c23, 24: c24, 25: c25, 26: c26, 27: c27, 28: c28_30}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rapido = "--rapido" in sys.argv
    pedidas = [int(a) for a in args] if args else sorted(CHECAGENS)
    print("AUDITORIA COMPLETA  |  %s  |  30 checagens\n" % time.strftime("%d/%m/%Y %H:%M"))
    fam = None
    t0 = time.time()
    for n in pedidas:
        fn = CHECAGENS.get(n)
        if fn is None:
            continue
        f = {1: "A", 9: "B", 16: "C", 23: "D", 28: "E"}
        for k, v in f.items():
            if n == k:
                print("\n--- %s. %s" % (v, FAMILIA[v]))
        try:
            fn(rapido) if fn in (c17, c18) else fn()
        except Exception as e:
            reg(n, "?", getattr(fn, "__name__", str(n)), "CRITICO",
                "a checagem falhou: %s: %s" % (type(e).__name__, e))
    t = pd.DataFrame(RESULTADOS)
    t.to_excel(RES / "auditoria_completa.xlsx", index=False)
    print("\n" + "=" * 78)
    for v in ["CRITICO", "ATENCAO", "OK"]:
        print("  %-8s %d" % (v, int((t.veredito == v).sum())))
    print("  tempo: %.0fs | planilha: resultados/auditoria_completa.xlsx" % (time.time() - t0))
    print("=" * 78)
    raise SystemExit(1 if (t.veredito == "CRITICO").any() else 0)


if __name__ == "__main__":
    main()
