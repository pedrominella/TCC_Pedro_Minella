# -*- coding: utf-8 -*-
"""
Seção 4.11, implicações de política, a pedido do autor depois do comentário do prof. Raphael Gouvea
(15/10/2026): o repasse maior sob a paridade não torna preferível a regra anterior. Alteração controlada,
autor "Claude". Também: linha no sumário (com marcador e campo de página) e remissão na conclusão.

Números usados, todos já no trabalho: perdas de cerca de R$ 98 bilhões (Almeida, Oliveira e Losekann,
2015, Introdução); diferenças de 1,72 e 1,15 ponto em 2021 e 2022 (seção 4.7.5); 0,005 e 0,176 no impacto
(Tabela 2); segunda volta não significativa em três meses (4.5.1) e sem efeito sobre os demais preços em
até seis meses na variável instrumental (4.8); subvenção ao diesel em 2018 e desonerações de 2022 (K.5);
fundo de estabilização do Peru (Apêndice I).

Uso: python aplicar_implicacoes_docx.py entrada saida [pagina]
"""
import copy, random, sys
from pathlib import Path
import docx
from docx.oxml.ns import qn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from revisao_track import Revisor

ENTRADA, SAIDA = Path(sys.argv[1]), Path(sys.argv[2])
PAG = sys.argv[3] if len(sys.argv) > 3 else "46"
W14_PARAID = "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"

TITULO = "4.11 Implicações de política"
TEXTO = [
    "Os resultados não autorizam a conclusão de que a regra anterior fosse preferível. Um repasse menor ao IPCA "
    "não significa que o custo do choque tenha desaparecido: entre 2011 e 2014, ele foi absorvido pela Petrobras, "
    "cujas perdas Almeida, Oliveira e Losekann (2015) estimam em cerca de R$ 98 bilhões, e reapareceu quando os "
    "preços foram corrigidos. A contenção adia a inflação e transfere seu custo para a empresa e, indiretamente, "
    "para o acionista controlador, além de enfraquecer o sinal de preço para consumo e investimento. O trabalho "
    "mede quanto do choque chega ao índice sob cada regra; não mede custo fiscal, eficiência ou bem-estar, e, com o "
    "choque tratado como endógeno, suas conclusões normativas pedem ainda mais cautela.",

    "As formas de amortecer o repasse também diferem entre si. Na contenção por empresa estatal, como entre 2011 e "
    "2014, o subsídio é implícito e fica fora do orçamento. Na subvenção explícita, como a do óleo diesel em 2018, o "
    "custo recai sobre o Tesouro, de forma transparente e temporária. Na redução de tributos, como em 2022, o "
    "governo abre mão de arrecadação. Os fundos de estabilização, como o mantido pelo Peru (Apêndice I), procuram "
    "suavizar o preço sem subsídio permanente, compensando altas e quedas ao longo do tempo. Todos reduzem o "
    "repasse de curto prazo, mas diferem em custo fiscal, transparência e duração. A literatura de corte "
    "transversal costuma reuni-los numa medida única, como o subsídio em proporção do PIB de Choi et al. (2018); o "
    "caso brasileiro mostra que a distinção importa, porque os três primeiros se sucederam no mesmo país em pouco "
    "mais de uma década.",

    "Para a política monetária, a implicação mais direta é que o repasse do petróleo não é parâmetro estrutural da "
    "economia, e sim atributo da regra de preços vigente. Sob a paridade, um choque típico passa a mover o IPCA já "
    "no mês em que ocorre, com resposta de 0,176 ponto contra 0,005 antes, e o exercício de equilíbrio parcial da "
    "seção 4.7.5 atribui à diferença entre as regras 1,72 ponto da inflação de 2021 e 1,15 da de 2022. Uma "
    "calibragem baseada na média da amostra inteira subestima o efeito sob a paridade e o superestima sob a "
    "contenção. Mudanças de regra, como a adoção da paridade em 2016 e seu encerramento em maio de 2023, pedem, "
    "portanto, recalibragem do repasse usado nas projeções.",

    "A natureza do repasse importa tanto quanto a magnitude. Se o choque atinge apenas os preços de combustíveis, "
    "ele altera preços relativos e tende a se dissipar, e a política monetária pode, em princípio, olhar através "
    "dele; se contamina os demais preços e as expectativas, exige reação. A evidência do trabalho sobre a segunda "
    "volta é frágil: a decomposição da seção 4.5.1 indica participação crescente dos demais preços na estimativa "
    "pontual, sem significância em três meses, e a variável instrumental da seção 4.8 não detecta efeito sobre eles "
    "em até seis meses. O quadro é compatível com um repasse concentrado nos próprios combustíveis, mas não permite "
    "descartar efeitos de segunda ordem em horizontes mais longos.",

    "Em síntese, a paridade torna o repasse maior e mais rápido, mas vinculado a uma regra explícita, e desloca o "
    "custo do choque da empresa e do Tesouro para os preços. A escolha da regra envolve, portanto, uma troca entre "
    "a volatilidade da inflação de curto prazo e custos fiscais e alocativos, que este trabalho ajuda a "
    "dimensionar, mas não resolve.",
]
assert not any(c in "".join(TEXTO) for c in "—–")

doc = docx.Document(ENTRADA)
R = Revisor(doc)
corpo = doc.element.body
usados = {e.get(W14_PARAID) for e in corpo.iter() if e.get(W14_PARAID)}
rnd = random.Random(20261017)


def paraid():
    while True:
        v = "%08X" % rnd.randrange(1, 0x7FFFFFFF)
        if v not in usados:
            usados.add(v); return v


def vis(p):
    return "".join(t.text or "" for t in p.iter(qn("w:t")) if qn("w:del") not in {a.tag for a in t.iterancestors()})


def estilo(p):
    s = p.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return s.get(qn("w:val")) if s is not None else ""


def um(lista, rot):
    assert len(lista) == 1, (rot, len(lista)); return lista[0]


def marca_ins(p):
    ppr = p.find(qn("w:pPr"))
    rpr = ppr.find(qn("w:rPr"))
    if rpr is None:
        rpr = ppr.makeelement(qn("w:rPr"), {}); ppr.append(rpr)
    rpr.insert(0, R._marca("w:ins"))


def novo_par(modelo, texto):
    p = corpo.makeelement(qn("w:p"), {}); p.set(W14_PARAID, paraid())
    ppr = copy.deepcopy(modelo.find(qn("w:pPr")))
    for v in ppr.findall(qn("w:rPr")):
        ppr.remove(v)
    p.append(ppr); marca_ins(p)
    ins = R._marca("w:ins"); ins.append(R._run_novo(modelo.find(".//" + qn("w:r")), texto)); p.append(ins)
    return p


P = list(corpo.iter(qn("w:p")))
cab_mod = um([p for p in P if estilo(p) == "Heading2" and vis(p).startswith("4.7 Robustez")], "4.7")
corpo_mod = um([p for p in P if vis(p).startswith("Na estimação país a país de Choi")], "corpo")
ultimo = um([p for p in P if vis(p).startswith("A parcela associada aos combustíveis, de 0,23 ponto")], "fim da 4.10")

bm_id = str(max(int(b.get(qn("w:id"))) for b in corpo.iter(qn("w:bookmarkStart"))) + 1)
BM = "_TocReviewed_pol411"
tit = novo_par(cab_mod, TITULO)
ins_t = tit.find(qn("w:ins"))
ins_t.addprevious(tit.makeelement(qn("w:bookmarkStart"), {qn("w:id"): bm_id, qn("w:name"): BM}))
ins_t.addnext(tit.makeelement(qn("w:bookmarkEnd"), {qn("w:id"): bm_id}))
ancora = ultimo
for el in [tit] + [novo_par(corpo_mod, t) for t in TEXTO]:
    ancora.addnext(el); ancora = el

# sumário: cópia da linha da 4.7 (nível 2, sem revisões), antes da linha do Capítulo 5
toc47 = um([p for p in P if estilo(p) == "TOC2" and vis(p).startswith("4.7 Robustez")], "toc 4.7")
toc5 = um([p for p in P if estilo(p) == "TOC1" and vis(p).startswith("5 CONCLUS")], "toc 5")
nova = copy.deepcopy(toc47); nova.set(W14_PARAID, paraid())
hl = nova.find(qn("w:hyperlink")); hl.set(qn("w:anchor"), BM)
ts = list(hl.iter(qn("w:t"))); ts[0].text = TITULO; ts[-1].text = PAG
for it in hl.iter(qn("w:instrText")):
    it.text = f" PAGEREF {BM} \\h "
for r in list(hl.findall(qn("w:r"))):
    i = R._marca("w:ins"); r.addprevious(i); i.append(r)
marca_ins(nova)
toc5.addprevious(nova)

# conclusão
p = um([x for x in P if "A mudança da política de preços deve ser considerada na calibragem do repasse." in vis(x)], "conclusão")
R.inserir_depois(p, "A mudança da política de preços deve ser considerada na calibragem do repasse.",
                 " A seção 4.11 discute as implicações de política, inclusive por que o repasse menor sob a "
                 "contenção não torna a regra anterior preferível.")

doc.save(SAIDA)
print("salvo", SAIDA)
