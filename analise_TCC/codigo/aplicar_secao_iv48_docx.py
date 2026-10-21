# -*- coding: utf-8 -*-
"""
Seção 4.8, identificação por variável instrumental, com os resultados de 14/10/2026 (iv_reestimacao.py e
iv_lp_modelos.py), como alteração controlada, autor "Claude".

Parte do arquivo de revisão anterior à subseção 4.7.6 (backup _backup_revisao_pre_secao_iv.docx), que ela
substitui. Faz:
  1  seção nova 4.8 antes da atual 4.8, com as equações (15) e (16) no formato de equação do Word e as
     Figuras 13 (F por horizonte) e 14 (LP-IV com a margem do diesel)
  2  renumeração: 4.8 -> 4.9 e 4.9 -> 4.10 (títulos, sumário e remissões); Figuras 13 e 14 -> 15 e 16
     (lista, legendas e remissões); linha nova no sumário com marcador e campo de página, e duas linhas
     novas na lista de figuras
  3  coerência: resumo, abstract, introdução (dentro da inserção pendente), 3.2, 3.4, 3.7, 4.5, conclusão e
     Apêndices C, G e K.1, que diziam não haver instrumento forte, passam a distinguir o choque (sem
     instrumento forte) do preço do diesel (instrumento forte a partir do primeiro mês, depois de 2017)
  4  correções numéricas já feitas antes na 4.7.6: F do choque estrutural de Känzig até 251 e largura do
     conjunto de Anderson-Rubin de 1,3 a 7,5 vezes a do intervalo de dois estágios

Uso: python aplicar_secao_iv48_docx.py entrada saida [pag_secao]
"""
import copy, random, sys
from pathlib import Path
import numpy as np, pandas as pd
import docx
from docx.oxml.ns import qn
from PIL import Image

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from revisao_track import Revisor
from omml import Eq

RES = AQUI.parent / "resultados"; FIGS = AQUI.parent / "figuras"
ENTRADA, SAIDA = Path(sys.argv[1]), Path(sys.argv[2])
PAG = sys.argv[3] if len(sys.argv) > 3 else "42"
W14_PARAID = "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def br(x, c=3):
    return f"{x:.{c}f}".replace(".", ",").replace("-", "−")


# ------------------------------------------------------------------ números, conferidos nos resultados
t = pd.read_excel(RES / "iv_lp_modelos.xlsx")
q = lambda ins, y, reg, h: t[(t.instrumento == ins) & (t.desfecho == y) & (t.regime == reg) & (t.h == h)].iloc[0]
dz = lambda y, h: q("margem_diesel", y, "pos", h)
assert [br(dz("ipca_geral", h).beta) for h in (1, 2)] == ["0,034", "0,045"]
assert [br(dz("ipca_geral", 1).ar_inf), br(dz("ipca_geral", 1).ar_sup)] == ["0,001", "0,079"]
assert [br(dz("ipca_geral", 2).ar_inf), br(dz("ipca_geral", 2).ar_sup)] == ["0,009", "0,091"]
assert dz("ipca_geral", 3).ar_inf < 0
assert [br(dz("ipca_transportes", h).beta) for h in (1, 2, 3)] == ["0,186", "0,222", "0,110"]
assert all(dz("ipca_transportes", h).ar_inf > 0 for h in (1, 2, 3))
assert [br(dz("ipca_ex_comb_central", h).beta) for h in (1, 6)] == ["0,009", "0,034"]
assert all(dz("ipca_ex_comb_central", h).ar_inf < 0 < dz("ipca_ex_comb_central", h).ar_sup for h in range(1, 7))
n5, n6 = dz("ipca_naocomercializaveis", 5), dz("ipca_naocomercializaveis", 6)
assert (br(n5.beta), br(n6.beta), br(n5.ar_inf), br(n5.ar_sup)) == ("0,059", "0,066", "−0,001", "0,123")
assert n5.ic_inf > 0 and n5.boot_inf > 0
for y in t.desfecho.unique():           # MQO dentro do Anderson-Rubin nos três primeiros meses
    if y in ("ipca_geral",) or True:
        for h in range(0, 4):
            r = dz(y, h); assert r.ar_inf <= r.beta_ols <= r.ar_sup, (y, h)
fd = t[(t.instrumento == "margem_diesel") & (t.regime == "pos") & (t.h >= 1)].F
assert (br(fd.min(), 1), br(fd.max(), 1)) == ("10,1", "31,3")
fdi = t[(t.instrumento == "margem_diesel") & (t.desfecho == "ipca_geral")]
assert br(fdi[(fdi.regime == "pos") & (fdi.h == 0)].F.iloc[0], 1) == "6,9"
assert (br(fdi[(fdi.regime == "pos") & (fdi.h >= 1)].F.min(), 1), br(fdi[(fdi.regime == "pos") & (fdi.h >= 1)].F.max(), 1)) == ("19,0", "31,3")
assert br(fdi[fdi.regime == "pre"].F.max(), 1) == "4,6"
assert br(t[(t.instrumento == "margem_gasolina") & (t.regime == "pos")].F.max(), 2) == "9,99"
k = t[t.instrumento == "surpresa_kanzig"]
assert (br(k.F.min(), 1), br(k.F.max(), 1)) == ("2,2", "6,6")
kp0 = k[(k.regime == "pos") & (k.h == 0)].iloc[0]
assert (br(kp0.ar_inf, 2), br(kp0.ar_sup, 2)) == ("−3,18", "0,04") and kp0.ar_sup < 0.176 and kp0.ar_sup < kp0.beta_ols
assert br(kp0.beta_ols) == "0,160"
ka = k[k.regime == "all"]
assert all(r.ar_inf <= r.beta_ols <= r.ar_sup for r in ka.itertuples())
g = lambda y, h: q("margem_gasolina", y, "pos", h)
assert [br(g("ipca_geral", h).beta) for h in (1, 2)] == ["0,044", "0,054"] and all(g("ipca_geral", h).ar_inf > 0 for h in (1, 2))
assert [br(g("ipca_transportes", h).beta, 2) for h in (0, 1, 2)] == ["0,25", "0,26", "0,26"]

TITULO = "4.8 Identificação por variável instrumental"
P1 = ("As estimativas anteriores tratam o choque do petróleo como endógeno e leem a parcela associada aos "
      "combustíveis como decomposição contábil (seções 3.4 e 3.7). Esta seção examina as duas etapas por variável "
      "instrumental. Para o choque, o instrumento é a surpresa de oferta de Känzig (2021), a variação dos preços "
      "futuros do petróleo em torno de anúncios da OPEP, diferente de zero em apenas 90 dos 270 meses. Para o "
      "preço doméstico dos combustíveis, são as margens de refino da gasolina e do diesel na Costa do Golfo dos "
      "Estados Unidos, isto é, a cotação do derivado menos a do Brent, ortogonalizada em relação ao choque. A "
      "lógica é que interrupções de refino, em geral climáticas, movem a margem sem mover o óleo cru e alcançariam "
      "o Brasil pela regra de paridade. A estimação é por projeções locais com variável instrumental, na "
      "especificação da seção 3.4:")
P3 = ("A Figura 13 mostra a estatística F do primeiro estágio por horizonte. A surpresa de Känzig fica entre 2,2 e "
      "6,6 em qualquer amostra e horizonte, e a margem da gasolina, abaixo de 10 em todos os desfechos e horizontes, "
      "com máximo de 9,99. A margem do diesel é a exceção: depois de março de 2017, o F vai de 19,0 a 31,3 entre um "
      "e seis meses após o choque, embora fique em 6,9 no próprio mês, e com três defasagens continua acima de 10 "
      "de um a seis meses; antes de 2017, não passa de 4,6. O contraste entre regimes é o que a regra de paridade "
      "prevê: a margem internacional do diesel só alcança o preço doméstico quando ele segue a referência externa. "
      "Para o choque do petróleo, portanto, não há instrumento forte, e as estimativas da seção 4.3 continuam "
      "lidas como associações condicionais.")
P4 = ("A Figura 14 apresenta as estimativas com a margem do diesel no regime posterior, em que o instrumento é "
      "forte a partir do primeiro mês, com F entre 10,1 e 31,3 nos cinco desfechos. Uma alta de 1 ponto "
      "logarítmico no diesel induzida pela margem eleva o IPCA Geral em 0,034 ponto em um mês e em 0,045 em dois, "
      "com intervalos de Anderson-Rubin de [0,001; 0,079] e [0,009; 0,091], e o efeito deixa de ser distinguível "
      "de zero a partir do terceiro mês. No IPCA Transportes, as respostas são de 0,186, 0,222 e 0,110 em um, dois "
      "e três meses, todas com intervalos que excluem o zero. Nos três primeiros meses, as estimativas por mínimos "
      "quadrados da mesma equação ficam dentro dos intervalos de Anderson-Rubin em todos os desfechos, o que não "
      "indica viés grande nesse elo.")
P5 = ("Nos preços que não são combustível, a identificação não encontra efeito. O IPCA sem combustíveis responde "
      "entre 0,009 e 0,034 de um a seis meses, sem significância, e os comercializáveis ficam próximos de zero. Os "
      "não comercializáveis sobem devagar, até 0,066 em seis meses; em cinco meses, o efeito de 0,059 tem "
      "intervalos convencional e por bootstrap que excluem o zero, mas o de Anderson-Rubin, de −0,001 a 0,123, o "
      "inclui. Há uma ressalva de interpretação: a margem do diesel move também a gasolina. As duas margens têm "
      "correlação de 0,31 depois de 2017, e a do diesel tem efeito positivo, embora fraco, sobre a gasolina ao "
      "consumidor, com F entre 2,6 e 5,3 até três meses. Parte da resposta do IPCA Geral e do IPCA Transportes "
      "passa, portanto, pela gasolina, o que explica por que a resposta do IPCA Geral, de 0,034 a 0,045, é muito "
      "maior do que o peso do diesel na cesta, de cerca de 0,15%. O exercício identifica o efeito de altas de "
      "combustíveis induzidas pela margem do diesel, e não o do diesel isoladamente.")
P6 = ("Com a margem da gasolina, os intervalos de Anderson-Rubin são limitados apenas até dois meses, quando "
      "indicam resposta do IPCA Geral de 0,044 em um mês e 0,054 em dois e do IPCA Transportes de 0,25 a 0,26, sem efeito "
      "sobre os demais preços; a partir de três meses, deixam de informar. Com a surpresa de Känzig, as estimativas "
      "do IPCA Geral oscilam em torno de zero, com intervalos de Anderson-Rubin amplos, que contêm as estimativas "
      "por mínimos quadrados na amostra completa em todos os horizontes. No regime posterior, porém, o intervalo do "
      "mês do choque, de −3,18 a 0,04, exclui a resposta de 0,176 da Tabela 2 e a de 0,160 da mesma equação por "
      "mínimos quadrados. Como o instrumento isola choques de oferta, enquanto a variação do Brent mistura oferta e "
      "demanda, a divergência é compatível com a hipótese de que parte da resposta de impacto por mínimos quadrados "
      "reflita choques de demanda, possibilidade que a seção 3.4 admite. O resultado vale para um único horizonte e "
      "para um instrumento nulo na maioria dos meses, e pede cautela.")
P7 = ("Em síntese, a variável instrumental confirma, a partir do primeiro mês e no regime posterior, o efeito dos "
      "combustíveis sobre o IPCA Geral e o IPCA Transportes, e não encontra efeito sobre os demais preços em até "
      "seis meses. Para o choque do petróleo e para o mês do impacto, não há instrumento forte, e a leitura da "
      "seção 3.4 permanece.")
LEG13 = "Figura 13 – Força dos instrumentos: estatística F do primeiro estágio por horizonte"
NOTA13 = ("Nota: F robusto (Newey-West) do instrumento excluído no primeiro estágio, no desenho do IPCA Geral; para "
          "a surpresa de Känzig, a variável instrumentada é o choque do petróleo, e para as margens, o preço do "
          "combustível acumulado de t−1 a t+h. Linha tracejada: limiar convencional de 10. Antes e depois "
          "referem-se a março de 2017.")
LEG14 = "Figura 14 – Projeções locais com variável instrumental: margem de refino do diesel, regime posterior"
NOTA14 = ("Nota: efeito de uma alta de 1 ponto logarítmico no preço do diesel induzida pela margem de refino, em "
          "pontos logarítmicos do desfecho acumulado. Barras: intervalos de 90% convencional e de Anderson-Rubin; "
          "linha tracejada: mínimos quadrados da mesma equação; fundo cinza: F do primeiro estágio abaixo de 10.")
assert not any(c in P1 + P3 + P4 + P5 + P6 + P7 + NOTA13 + NOTA14 for c in "—")

# ------------------------------------------------------------------ documento
doc = docx.Document(ENTRADA)
R = Revisor(doc)
corpo = doc.element.body
E = Eq(corpo)
usados = {e.get(W14_PARAID) for e in corpo.iter() if e.get(W14_PARAID)}
rnd = random.Random(20261016)


def paraid():
    while True:
        v = "%08X" % rnd.randrange(1, 0x7FFFFFFF)
        if v not in usados:
            usados.add(v); return v


def vis(p):
    return "".join(x.text or "" for x in p.iter(qn("w:t")) if qn("w:del") not in {a.tag for a in x.iterancestors()}
                   and MATH + "oMath" not in {a.tag for a in x.iterancestors()})


def achar(inicio):
    r = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith(inicio)]
    assert len(r) == 1, (inicio[:50], len(r)); return r[0]


def marca_ins(p):
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = p.makeelement(qn("w:pPr"), {}); p.insert(0, ppr)
    rpr = ppr.find(qn("w:rPr"))
    if rpr is None:
        rpr = ppr.makeelement(qn("w:rPr"), {}); ppr.append(rpr)
    rpr.insert(0, R._marca("w:ins"))


def run_de(modelo_run, texto=None, tab=False):
    r = R._run_novo(modelo_run, texto if texto is not None else "")
    if texto is None:
        r.remove(r.find(qn("w:t")))
    if tab:
        r.append(r.makeelement(qn("w:tab"), {}))
    return r


def novo_par(modelo, segmentos):
    """segmentos: textos, ('tab',) ou elementos m:oMath; tudo numa inserção."""
    p = corpo.makeelement(qn("w:p"), {}); p.set(W14_PARAID, paraid())
    ppr = copy.deepcopy(modelo.find(qn("w:pPr")))
    for v in ppr.findall(qn("w:rPr")):
        ppr.remove(v)
    p.append(ppr); marca_ins(p)
    mr = modelo.find(".//" + qn("w:r"))
    ins = R._marca("w:ins")
    for s in segmentos:
        if isinstance(s, str):
            ins.append(R._run_novo(mr, s))
        elif isinstance(s, tuple):
            ins.append(run_de(mr, tab=True))
        else:
            ins.append(s)
    p.append(ins)
    return p


_docpr = [max(int(x.get("id")) for x in corpo.iter(WP + "docPr"))]


def par_figura(modelo_img, arquivo, descr):
    p = copy.deepcopy(modelo_img); p.set(W14_PARAID, paraid())
    rid, _ = doc.part.get_or_add_image(str(arquivo))
    w, h = Image.open(arquivo).size
    for b in p.iter(A + "blip"):
        b.set(RNS + "embed", rid)
    for ext in list(p.iter(WP + "extent")) + list(p.iter(A + "ext")):
        ext.set("cy", str(int(int(ext.get("cx")) * h / w)))
    _docpr[0] += 1
    for pr in p.iter(WP + "docPr"):
        pr.set("id", str(_docpr[0])); pr.set("name", arquivo.stem); pr.set("descr", descr)
    for r in list(p.findall(qn("w:r"))):
        i = R._marca("w:ins"); r.addprevious(i); i.append(r)
    marca_ins(p)
    return p


# atalhos de equação
P_ = lambda x: E.d(x)
CH = lambda x: E.d(x, "{", "}")
RT = lambda s: E.r(s, reto=True)
TX = lambda s: E.r(s, palavra=True)
M = lambda *x: E.omath(*x)

# modelos
def titulo(inicio):
    r = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith(inicio) and p.find(qn("w:pPr") + "/" + qn("w:pStyle")) is not None
         and p.find(qn("w:pPr") + "/" + qn("w:pStyle")).get(qn("w:val")).startswith("Heading")]
    assert len(r) == 1, (inicio, len(r)); return r[0]


cab48 = titulo("4.8 O resultado em perspectiva internacional")
assert cab48.find(qn("w:pPr") + "/" + qn("w:pStyle")).get(qn("w:val")) == "Heading2"
corpo_mod = achar("Acumulam-se, por ano civil, as contribuições")
eq_mod = [p for p in corpo.iter(qn("w:p")) if p.find(MATH + "oMath") is not None and "(13)" in vis(p)][0]
leg12 = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith("Figura 12 – Contribuição")]
assert len(leg12) == 2
lista12, leg12 = leg12
img12 = leg12.getnext(); nota12 = img12.getnext(); fonte12 = nota12.getnext()
assert img12.find(".//" + qn("w:drawing")) is not None and vis(nota12).startswith("Nota:") and vis(fonte12).startswith("Fonte:")
ancora = cab48.getprevious()
assert vis(ancora).startswith("Nos subgrupos sem combustíveis")

# ------------------------------------------------------------------ 1 seção nova
bm_id = str(max(int(b.get(qn("w:id"))) for b in corpo.iter(qn("w:bookmarkStart"))) + 1)
BM = "_TocReviewed_iv48"
tit = novo_par(cab48, [TITULO])
ins_t = tit.find(qn("w:ins"))
ins_t.addprevious(tit.makeelement(qn("w:bookmarkStart"), {qn("w:id"): bm_id, qn("w:name"): BM}))
ins_t.addnext(tit.makeelement(qn("w:bookmarkEnd"), {qn("w:id"): bm_id}))

eq15 = M("y", P_("t,h"), "=α+", E.sub("β", "r"), P_("h"), "·D", P_("t,h"), "+", TX("controles"), "+ε", P_("t,h"))
eq16 = M("D", P_("t,h"), "=a+", E.sub("π", RT("pré")), P_("h"), "·m", P_("t"), "·1", CH("t<τ"), "+",
         E.sub("π", RT("pós")), P_("h"), "·m", P_("t"), "·1", CH("t≥τ"), "+", TX("controles"), "+v", P_("t,h"))
p2 = ["Em (15) e (16), ", M("y", P_("t,h")), " é a variação acumulada do desfecho, definida como em (2); ",
      M("D", P_("t,h")), " é a variação acumulada do preço do combustível de ", M("t−1"), " a ", M("t+h"), "; ",
      M("m", P_("t")), " é a margem de refino; e ", M("r"), " indica o regime. Os controles são os do modelo "
      "principal, com o choque do Brent também com coeficiente próprio em cada regime, porque a margem é ortogonal "
      "ao Brent apenas na amostra inteira. O coeficiente ", M(E.sub("β", RT("pós")), P_("h")), " mede, em pontos "
      "logarítmicos, a resposta do desfecho a uma alta de 1 ponto logarítmico no preço do combustível induzida pela "
      "margem. Para o choque do petróleo, a equação é a mesma, com o choque no lugar de ", M("D", P_("t,h")),
      " e a surpresa de Känzig no lugar de ", M("m", P_("t")), ". O número de defasagens é o indicado pelo critério "
      "de Akaike acrescido de uma, como no modelo principal: cinco para o diesel e duas para os demais "
      "instrumentos. O erro-padrão é de Newey-West com banda automática, e cada estimativa traz o intervalo de 90% "
      "de Anderson-Rubin, robusto a instrumento fraco (ANDREWS; STOCK; SUN, 2019), além do intervalo por bootstrap "
      "selvagem em blocos de doze meses, com 999 réplicas. O limiar convencional da estatística F do primeiro "
      "estágio é 10 (STAIGER; STOCK, 1997)."]

novos = [tit, novo_par(corpo_mod, [P1]),
         novo_par(eq_mod, [("tab",), eq15, ("tab",), "(15)"]),
         novo_par(eq_mod, [("tab",), eq16, ("tab",), "(16)"]),
         novo_par(corpo_mod, p2), novo_par(corpo_mod, [P3]),
         novo_par(leg12, [LEG13]), par_figura(img12, FIGS / "iv_F_por_horizonte.png",
                                              "Estatística F do primeiro estágio por horizonte, por instrumento e regime"),
         novo_par(nota12, [NOTA13]), novo_par(fonte12, ["Fonte: Elaboração própria."]),
         novo_par(corpo_mod, [P4]),
         novo_par(leg12, [LEG14]), par_figura(img12, FIGS / "iv_lp_diesel_tcc.png",
                                              "Projeções locais com variável instrumental, margem do diesel, regime posterior"),
         novo_par(nota12, [NOTA14]), novo_par(fonte12, ["Fonte: Elaboração própria."]),
         novo_par(corpo_mod, [P5]), novo_par(corpo_mod, [P6]), novo_par(corpo_mod, [P7])]
for p in novos:
    ancora.addnext(p); ancora = p

# ------------------------------------------------------------------ 2 renumeração
R.substituir(cab48, "4.8", "4.9")
R.substituir(titulo("4.9 O resultado à luz da literatura"), "4.9", "4.10")
toc48 = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith("4.8 O resultado em perspectiva") and p.find(qn("w:hyperlink")) is not None][0]
toc49 = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith("4.9 O resultado à luz") and p.find(qn("w:hyperlink")) is not None][0]
nova_toc = copy.deepcopy(toc48); nova_toc.set(W14_PARAID, paraid())
hl = nova_toc.find(qn("w:hyperlink")); hl.set(qn("w:anchor"), BM)
textos = [x for x in hl.iter(qn("w:t"))]
textos[0].text = TITULO; textos[-1].text = PAG
for it in hl.iter(qn("w:instrText")):
    it.text = f" PAGEREF {BM} \\h "
for r in list(hl.findall(qn("w:r"))):
    i = R._marca("w:ins"); r.addprevious(i); i.append(r)
marca_ins(nova_toc)
toc48.addprevious(nova_toc)
R.substituir(toc48, "4.8", "4.9")
R.substituir(toc49, "4.9", "4.10")
R.substituir(R.paragrafo("Resta registrar o que a hipótese não cobre"), "a seção 4.8 oferece", "a seção 4.9 oferece")
R.substituir(R.paragrafo("Este apêndice reporta os números do exercício comparado"), "da seção 4.8.", "da seção 4.9.")
R.substituir(R.paragrafo("A figura deste apêndice acompanha a seção 4.8"), "a seção 4.8", "a seção 4.9")
nota_f13 = [p for p in corpo.iter(qn("w:p")) if "com as adaptações da seção 4.8" in vis(p)]
assert len(nota_f13) == 1
R.substituir(nota_f13[0], "da seção 4.8", "da seção 4.9")
# figuras 13 e 14 -> 15 e 16
cands13 = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith("Figura 13 – O repasse brasileiro")]
cands14 = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith("Figura 14 – Composição dos choques")]
assert len(cands13) == 2 and len(cands14) == 2
for p in cands13:
    R.substituir(p, "Figura 13", "Figura 15")
for p in cands14:
    R.substituir(p, "Figura 14", "Figura 16")
R.substituir([p for p in corpo.iter(qn("w:p")) if "e na Figura 13, no Apêndice N" in vis(p)][0],
             "na Figura 13, no Apêndice N", "na Figura 15, no Apêndice N")
R.substituir([p for p in corpo.iter(qn("w:p")) if vis(p).startswith("A Figura 14 não mostra")][0], "A Figura 14", "A Figura 16")
anc = lista12
for leg in (LEG13, LEG14):
    nl = novo_par(lista12, [leg]); anc.addnext(nl); anc = nl

# ------------------------------------------------------------------ 3 coerência
p = R.paragrafo("Este trabalho estima o repasse do petróleo")
R.substituir(p, "Como nenhum instrumento externo testado se mostrou forte, o choque é tratado como endógeno e os "
                "coeficientes, como associações condicionais.",
             "Como nenhum instrumento externo para o choque do petróleo se mostrou forte, o choque é tratado como "
             "endógeno e os coeficientes, como associações condicionais. Instrumentado pela margem de refino "
             "internacional, o preço do diesel afeta o IPCA Geral nos primeiros meses depois de 2017, sem efeito "
             "detectável sobre os demais preços.")
p = R.paragrafo("This study estimates oil price pass-through")
R.substituir(p, "Since none of the external instruments tested proved strong, the shock is treated as endogenous and "
                "the coefficients as conditional associations.",
             "Since no external instrument for the oil shock proved strong, the shock is treated as endogenous and "
             "the coefficients as conditional associations. Instrumented by the international refining margin, the "
             "diesel price affects headline inflation in the first months after 2017, with no detectable effect on "
             "other prices.")


def editar_insercao(trecho, novo):
    """Texto que eu mesmo inseri e ainda está pendente: muda dentro da própria inserção."""
    alvos = [x for ins in corpo.iter(qn("w:ins")) if ins.get(qn("w:author")) == R.autor
             for x in ins.iter(qn("w:t")) if trecho in (x.text or "")]
    assert len(alvos) == 1, (trecho[:40], len(alvos))
    alvos[0].text = alvos[0].text.replace(trecho, novo)


editar_insercao("Como os instrumentos externos testados se mostraram fracos, o choque é tratado como endógeno,",
                "Como o instrumento externo disponível para o choque se mostrou fraco, o choque é tratado como "
                "endógeno,")
editar_insercao("sob a hipótese de vieses comparáveis nos dois períodos.",
                "sob a hipótese de vieses comparáveis nos dois períodos. O elo entre combustíveis e inflação é testado "
                "também por variável instrumental, com a margem de refino do diesel.")
editar_insercao("que as duas construções usadas não sejam confirmações independentes.",
                "que as duas construções usadas não sejam confirmações independentes. A identificação por variável "
                "instrumental, possível para o preço do diesel a partir do primeiro mês depois de 2017, confirma o "
                "efeito dos combustíveis sobre o IPCA Geral e o IPCA Transportes nos primeiros meses e não encontra "
                "efeito sobre os demais preços em até seis meses.")

p = R.paragrafo("Uma limitação da medida adotada é que o modelo não separa choques")
R.substituir(p, "A tentativa de identificação por instrumento externo, nos Apêndices G e K.1, não encontrou instrumento "
                "forte o bastante para estimação em dois estágios.",
             "A tentativa de instrumentar o choque com a surpresa de oferta de Känzig (2021) não encontrou instrumento "
             "forte (seção 4.8).")
p = R.paragrafo("Convém explicitar o que esta especificação identifica")
R.substituir(p, "mas nenhum dos instrumentos testados se mostrou forte o bastante. A surpresa de oferta de Känzig (2021) "
                "tem estatística F de primeiro estágio entre 2,9 e 6,8, e as margens de refino da gasolina e do diesel "
                "ficam entre 0,25 e 5,45, todas abaixo do limiar convencional de 10 (STAIGER; STOCK, 1997), como "
                "reportam os Apêndices G e K.1.",
             "mas o instrumento disponível se mostrou fraco: a surpresa de oferta de Känzig (2021) tem estatística F "
             "de primeiro estágio entre 2,2 e 6,6, abaixo do limiar convencional de 10 (STAIGER; STOCK, 1997), como "
             "mostra a seção 4.8. As margens de refino internacionais, examinadas na mesma seção, instrumentam o preço "
             "doméstico dos combustíveis, e não o choque.")
p = R.paragrafo("Registre-se a limitação de identificação.")
R.substituir(p, "não é causalmente identificada, pois não se dispõe de instrumento específico para esse preço.",
             "não é causalmente identificada nesta decomposição; a seção 4.8 a examina por variável instrumental, com "
             "a margem de refino do diesel, forte apenas a partir do primeiro mês e no regime posterior.")
p = [x for x in corpo.iter(qn("w:p")) if "Vale aqui a limitação de identificação da seção 3.7" in vis(x)][0]
R.inserir_depois(p, "a leitura do mediador.", " A seção 4.8 examina essa etapa por variável instrumental.")
p = R.paragrafo("A transmissão mensal concentra-se nos dois primeiros meses.")
R.inserir_depois(p, "com diferença significativa.",
                 " Por variável instrumental, com a margem de refino do diesel, o efeito dos combustíveis sobre o IPCA "
                 "Geral e o IPCA Transportes aparece nos primeiros meses depois de 2017, sem efeito detectável sobre os "
                 "demais preços em até seis meses.")
p = R.paragrafo("A interpretação enfrenta limites:", aceita_revisao=True)
R.substituir(p, "pois nenhum instrumento testado alcançou estatística F de 10,",
             "pois o instrumento disponível para ele não alcançou estatística F de 10,")
R.substituir(p, "o mediador não é causalmente identificado;",
             "o mediador só é identificado por variável instrumental a partir do primeiro mês e no regime posterior, e "
             "o instrumento move também a gasolina;")
p = R.paragrafo("As extensões incluem")
R.substituir(p, "identificar o mediador por um instrumento que satisfaça relevância e exclusão,",
             "estender a identificação do mediador, aqui limitada aos meses seguintes ao choque e ao regime de "
             "paridade, com um instrumento que separe o diesel da gasolina,")

# apêndices
p = R.paragrafo("O Apêndice K.1 reporta a projeção sobre o choque de notícia de oferta")
R.substituir(p, "o conjunto de Anderson-Rubin é aproximadamente 80% mais amplo nos casos reportados e nunca exclui o zero.",
             "o conjunto de Anderson-Rubin é sempre limitado, tem de 1,3 a 7,5 vezes a largura do intervalo de dois "
             "estágios (mediana de 1,6) e nunca exclui o zero.")
p = R.paragrafo("A outra série do arquivo, o choque estrutural")
R.substituir(p, "entre 114 e 276", "entre 114 e 251")
R.substituir(p, "os conjuntos de Anderson-Rubin são 80% mais amplos e não excluem o zero.",
             "os conjuntos de Anderson-Rubin têm de 1,3 a 7,5 vezes a largura do intervalo de dois estágios e não "
             "excluem o zero.")
p = R.paragrafo("O trabalho trata o choque do petróleo como endógeno e não adota variável instrumental")
R.substituir(p, "e não adota variável instrumental, porque nenhum dos instrumentos testados se mostrou forte o bastante "
                "(seção 3.4). Este apêndice reporta as estatísticas desses instrumentos, para documentar a tentativa e "
                "permitir que o leitor avalie a decisão.",
             "e não o instrumenta, porque o instrumento disponível para ele se mostrou fraco (seções 3.4 e 4.8). Este "
             "apêndice reporta as estatísticas dos instrumentos na especificação original, com subamostras separadas "
             "pelo anúncio de outubro de 2016, três defasagens e banda de Newey-West de um mês, para documentar a "
             "tentativa; a seção 4.8 os reestima com a especificação do modelo principal.")
p = R.paragrafo("Nenhum dos instrumentos alcança o limiar convencional.")
R.substituir(p, "Nenhum dos instrumentos alcança o limiar convencional. O único candidato que o supera é",
             "Nessa especificação, nenhum dos instrumentos alcança o limiar convencional; com a do modelo principal, a "
             "margem do diesel o supera depois de 2017, a partir do primeiro mês (seção 4.8). Entre os demais "
             "candidatos, o único que o supera é")
p = R.paragrafo("Verificação análoga foi conduzida com instrumentos construídos")
R.substituir(p, "Nenhum dos dois instrumentos, contudo, alcança o limiar convencional de 10, e a margem do diesel não "
                "apresenta primeiro estágio em regime algum.",
             "Nessa especificação, com subamostras separadas pelo anúncio de outubro de 2016, nenhum dos dois alcança o "
             "limiar convencional de 10; com a especificação do modelo principal, a margem do diesel o supera depois "
             "de 2017, a partir do primeiro mês, como mostra a seção 4.8.")
R.substituir(p, "A conclusão coincide com a do parágrafo anterior: nesta amostra não se dispõe de instrumento externo "
                "com potência suficiente para inferência por dois estágios,",
             "Para o choque do petróleo, a conclusão coincide com a do parágrafo anterior: não se dispõe de instrumento "
             "externo com potência suficiente para inferência por dois estágios,")
p = [x for x in corpo.iter(qn("w:p")) if "A implicação para a agenda futura é que a via instrumental" in vis(x)][0]
R.substituir(p, "A implicação para a agenda futura é que a via instrumental exige mudança da unidade de observação, e "
                "mesmo assim não garante instrumento forte.",
             "A experiência deste trabalho acrescenta que, em séries agregadas, a força de um instrumento pode depender "
             "do regime de preços: a margem do diesel só se torna forte sob a paridade (seção 4.8).")

doc.save(SAIDA)
print("salvo", SAIDA)
