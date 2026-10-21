# -*- coding: utf-8 -*-
"""
Acrescenta ao Apêndice K do TCC a subseção K.7, com o VAR recursivo por regime (08/10/2026).
Números de resultados/modelo_11_alternativos/var_cholesky.xlsx e var_cholesky_bandas.xlsx
(onda5_alternativos.py e fig_var_cholesky.py); o desvio-padrão do choque em cada regime é
recalculado aqui com a mesma especificação. Tudo é conferido com assert antes de gravar.
Backup feito antes: historico/3_versoes_e_backups/backups_do_TCC/_backup_TCC_pre_var_apendice.docx
"""
import copy, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import docx
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lp_base_nova as L
import onda5_alternativos as A

RAIZ = AQUI.parents[1]
ARQ = RAIZ / "TCC_PEDRO_MINELLA.docx"
RES = RAIZ / "analise_TCC" / "resultados" / "modelo_11_alternativos"


def v(x, casas=3):
    return (f"%.{casas}f" % x).replace(".", ",").replace("-0,000", "0,000")


# ---------------------------------------------------------------- números
c = pd.read_excel(RES / "var_cholesky.xlsx", sheet_name="comparacao")
bd = pd.read_excel(RES / "var_cholesky_bandas.xlsx")


def val(var, h, reg):
    return float(c[(c.variavel == var) & (c.h == h)][reg].iloc[0])


def bnd(var, reg, h):
    r = bd[(bd.variavel == var) & (bd.regime == reg) & (bd.h == h)].iloc[0]
    return float(r.banda_inf), float(r.banda_sup)


ip0_pre, ip0_pos = val("ipca_geral", 0, "pre"), val("ipca_geral", 0, "pos")
ip3_pre, ip3_pos = val("ipca_geral", 3, "pre"), val("ipca_geral", 3, "pos")
ip6_pos = val("ipca_geral", 6, "pos")
ref0_pre, ref0_pos = val("gasolina_refinaria", 0, "pre"), val("gasolina_refinaria", 0, "pos")
con0_pre, con0_pos = val("gasolina_consumidor", 0, "pre"), val("gasolina_consumidor", 0, "pos")
b0, b3 = bnd("ipca_geral", "pos", 0), bnd("ipca_geral", "pos", 3)
g = bd[bd.variavel == "ipca_geral"]
pre_inclui_zero = bool(((g[g.regime == "pre"].banda_inf <= 0) & (g[g.regime == "pre"].banda_sup >= 0)).all())
pos_exclui_zero = bool((g[g.regime == "pos"].banda_inf > 0).all())

# desvio-padrão do resíduo da equação do petróleo em cada regime (tamanho do choque de Cholesky)
b, _ = A.base()
dp = {}
for rg in (0, 1):
    d = b[b.regime_pos == rg]
    cols = ["choque_petroleo", "dlog_cambio", "dlog_atividade", "dlog_ipca_geral", "selic", "expectativa"]
    m = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    dp[rg] = float(np.sqrt(VAR(m.values).fit(L.LAGS, trend="c").sigma_u[0, 0]))

assert [v(ip0_pre), v(ip0_pos), v(ip3_pre), v(ip3_pos), v(ip6_pos)] == ["0,000", "0,094", "0,028", "0,335", "0,380"]
assert [v(x) for x in (*b0, *b3)] == ["0,061", "0,110", "0,156", "0,418"]
assert pre_inclui_zero and pos_exclui_zero
assert [v(ref0_pre), v(ref0_pos), v(con0_pre), v(con0_pos)] == ["0,271", "4,756", "0,253", "1,137"]
assert [v(dp[0]), v(dp[1])] == ["0,789", "1,136"]
u0, u3, u3_pre = ip0_pos / dp[1], ip3_pos / dp[1], ip3_pre / dp[0]
assert [v(u0), v(u3), v(u3_pre)] == ["0,083", "0,295", "0,036"]

TITULO = "K.7 Vetor autorregressivo com identificação recursiva"
PAR1 = (
    "Como verificação de método, estimou-se um vetor autorregressivo em cada regime, com as mesmas variáveis do "
    "modelo principal, ordenadas como choque do petróleo, câmbio, atividade, preço de interesse, Selic e expectativa "
    "de inflação, as mesmas três defasagens e identificação recursiva com o petróleo em primeiro. A resposta é a "
    "acumulada a um choque de um desvio-padrão, com banda de 90% por bootstrap selvagem de resíduos com pesos de "
    "Rademacher e 400 réplicas. O padrão do Capítulo 4 se "
    f"reproduz. No IPCA Geral, a resposta no mês do choque é de {v(ip0_pre)} no regime anterior e de {v(ip0_pos)} no "
    f"posterior, com banda de {v(b0[0])} a {v(b0[1])}; em três meses, de {v(ip3_pre)} e de {v(ip3_pos)}, com banda "
    f"de {v(b3[0])} a {v(b3[1])} no posterior. A banda do regime anterior inclui o zero em todos os horizontes, e a "
    "do posterior o exclui em todos. Nos preços de combustíveis o contraste se repete: no impacto, a gasolina na "
    f"refinaria passa de {v(ref0_pre)} para {v(ref0_pos)}, e a gasolina ao consumidor, de {v(con0_pre)} para "
    f"{v(con0_pos)}."
)
PAR2 = (
    "Duas diferenças em relação à especificação principal limitam a comparação de magnitudes. O vetor "
    "autorregressivo não inclui as indicadoras de sazonalidade nem as de pandemia, e o choque de identificação "
    "recursiva tem o tamanho do desvio-padrão do resíduo da equação do petróleo em cada regime, de "
    f"{v(dp[0])} no anterior e de {v(dp[1])} no posterior. Por unidade de choque, a resposta do IPCA Geral no "
    f"regime posterior é de {v(u0)} no impacto e de {v(u3)} em três meses, contra {v(u3_pre)} em três meses no "
    "anterior, de modo que o contraste não decorre do tamanho do choque. No impacto, a resposta é menor que a da "
    f"projeção local, {v(ip0_pos)} contra 0,176, e mais persistente, chegando a {v(ip6_pos)} em seis meses, o que "
    "reflete a dinâmica imposta pelas defasagens do sistema. O exercício confirma o padrão entre regimes, e "
    "não a magnitude, que permanece a da projeção local."
)

# ---------------------------------------------------------------- documento
d = docx.Document(ARQ)
P = d.paragraphs
assert P[616].text == "K.6 Dinâmica específica a cada regime" and P[616].style.name == "Heading 3"
assert P[617].text.endswith("e não a de repasse nulo que se tornou positivo.")
assert P[619].text.startswith("APÊNDICE L")
assert not any(p.text.startswith("K.7") for p in P), "K.7 já existe"


def clonar_apos(modelo, ancora, texto):
    novo = copy.deepcopy(modelo._p)
    ancora._p.addnext(novo)
    par = docx.text.paragraph.Paragraph(novo, ancora._parent)
    par.runs[0].text = texto
    for r in par.runs[1:]:
        r.text = ""
    return par


h = clonar_apos(P[616], P[617], TITULO)
p1 = clonar_apos(P[617], h, PAR1)
p2 = clonar_apos(P[617], p1, PAR2)

d.save(ARQ)
print("salvo", ARQ)
