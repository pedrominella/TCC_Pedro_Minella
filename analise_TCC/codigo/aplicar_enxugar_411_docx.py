# -*- coding: utf-8 -*-
"""
Enxuga a seção 4.11 (implicações de política) para o corpo voltar a 40 páginas, a pedido do autor (20/10/2026).
A seção é inserção minha ainda pendente, então o texto muda dentro da própria inserção, sem tocar no texto do
autor: sai o parágrafo final de síntese, que repetia o primeiro (custo deslocado da empresa e do Tesouro para os
preços), o parágrafo sobre as formas de amortecer o repasse fica mais curto, com os mesmos exemplos, e duas
formulações do primeiro parágrafo encurtam. Como alteração controlada, saem dois parágrafos vazios antes da
conclusão, um deles um título de nível 2 vazio que abria uma página em branco.

Uso: python aplicar_enxugar_411_docx.py entrada saida
"""
import sys
from pathlib import Path
import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor

ENT, SAI = sys.argv[1], sys.argv[2]
doc = docx.Document(ENT); R = Revisor(doc); corpo = doc.element.body

P2_NOVO = ("As formas de amortecer o repasse também diferem: a contenção por empresa estatal, como entre 2011 e 2014, "
           "é subsídio implícito, fora do orçamento; a subvenção ao óleo diesel em 2018 e a redução de tributos em "
           "2022 têm custo fiscal explícito; e os fundos de estabilização, como o do Peru (Apêndice I), compensam "
           "altas e quedas. Medidas únicas, como o subsídio em proporção do PIB de Choi et al. (2018), não captam "
           "essas diferenças de custo, transparência e duração.")
assert not any(c in P2_NOVO for c in "—–")
# no primeiro parágrafo, duas formulações mais curtas, para a quebra de página antes da conclusão caber na
# mesma folha (sem elas, sobra 1 pt)
P1_TROCAS = [("Os resultados não autorizam a conclusão de que a regra anterior fosse preferível.",
              "Os resultados não indicam que a regra anterior fosse preferível."),
             ("transfere seu custo para a empresa e, indiretamente, para o acionista controlador,",
              "transfere seu custo para a empresa e seu acionista controlador,")]


def minha(p):
    """Parágrafo inteiro inserido por mim e ainda pendente: devolve o w:ins único com o texto."""
    marca = p.find(qn("w:pPr") + "/" + qn("w:rPr") + "/" + qn("w:ins"))
    filhos = [c for c in p if c.tag != qn("w:pPr")]
    assert marca is not None and marca.get(qn("w:author")) == R.autor
    assert len(filhos) == 1 and filhos[0].tag == qn("w:ins") and filhos[0].get(qn("w:author")) == R.autor
    return filhos[0]


def achar(inicio):
    r = [p for p in corpo.iter(qn("w:p")) if "".join(t.text or "" for t in p.iter(qn("w:t"))).startswith(inicio)]
    assert len(r) == 1, (inicio[:40], len(r)); return r[0]


p1 = achar("Os resultados não autorizam a conclusão de que a regra anterior fosse preferível.")
ts = list(minha(p1).iter(qn("w:t")))
assert len(ts) == 1
for velho, novo in P1_TROCAS:
    assert ts[0].text.count(velho) == 1, velho
    ts[0].text = ts[0].text.replace(velho, novo)

p2 = achar("As formas de amortecer o repasse também diferem entre si.")
ts = list(minha(p2).iter(qn("w:t")))
assert len(ts) == 1
ts[0].text = P2_NOVO

p5 = achar("Em síntese, a paridade torna o repasse maior e mais rápido")
minha(p5)
fim = p5.getprevious()
p5.getparent().remove(p5)

# Entre o fim do capítulo 4 e a conclusão (título com quebra de página antes) há três parágrafos vazios, o do
# meio com estilo de título de nível 2, resto do título da antiga 4.10 Síntese. Esse título vazio não cabe no
# pé da p. 49 e abre uma página só para ele. Ficam excluídas, como alteração controlada, as marcas dos dois
# primeiros, que se fundem ao terceiro, de estilo normal.
vazios = [fim.getnext(), fim.getnext().getnext(), fim.getnext().getnext().getnext()]
estilos = [(v.find(qn("w:pPr") + "/" + qn("w:pStyle")).get(qn("w:val"))
            if v.find(qn("w:pPr") + "/" + qn("w:pStyle")) is not None else "") for v in vazios]
assert estilos == ["", "Heading2", ""], estilos
assert all(len(v) == 1 and v[0].tag == qn("w:pPr") for v in vazios)
cab5 = vazios[2].getnext()
assert "".join(t.text for t in cab5.iter(qn("w:t"))) == "5 CONCLUSÃO"
for v in vazios[:2]:
    v.find(qn("w:pPr") + "/" + qn("w:rPr")).insert(0, R._marca("w:del"))

doc.save(SAI); print("salvo", SAI)
