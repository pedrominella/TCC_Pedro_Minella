# -*- coding: utf-8 -*-
"""
Correção da classificação do etanol, como alteração controlada (15/10/2026).

Fonte: Banco Central do Brasil, Estudo Especial 69/2019 ("Atualizações da estrutura de ponderação do IPCA e
repercussão nas suas classificações"), Tabela 5, estrutura em vigor desde janeiro de 2020: monitorados
incluem gasolina, óleo diesel e gás veicular, mas não o etanol; comercializáveis são calculados por exclusão
de não comercializáveis e monitorados, e o etanol não está na lista de não comercializáveis. Logo, o etanol
integra os comercializáveis. Figueiredo e Ferreira (2002), já nas referências, listavam o álcool entre os
monitorados.

O texto dizia que comercializáveis e não comercializáveis excluem combustíveis por construção; isso só vale
para os não comercializáveis.

Uso: python aplicar_etanol_docx.py entrada saida
"""
import copy, random, sys
from pathlib import Path
import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor

ENT, SAI = sys.argv[1], sys.argv[2]
W14_PARAID = "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
doc = docx.Document(ENT); R = Revisor(doc); corpo = doc.element.body


def vis(p):
    return "".join(t.text or "" for t in p.iter(qn("w:t")) if qn("w:del") not in {a.tag for a in t.iterancestors()})


def com(trecho, n=1):
    r = [p for p in corpo.iter(qn("w:p")) if trecho in vis(p)]
    assert len(r) == n, (trecho[:50], len(r)); return r


for p in com("e desfechos que excluem combustíveis", 2):
    R.substituir(p, "desfechos que excluem combustíveis", "desfechos que excluem gasolina e diesel")
for p in com("subgrupos que excluem combustíveis", 2):
    R.substituir(p, "subgrupos que excluem combustíveis", "subgrupos que excluem gasolina e diesel")
p, = com("pela propriedade de excluírem combustíveis por construção")
R.substituir(p, "pela propriedade de excluírem combustíveis por construção, já que estes são classificados como "
                "preços monitorados.",
             "porque excluem gasolina e óleo diesel, classificados como preços monitorados. O etanol, porém, é preço "
             "livre na classificação do Banco Central em vigor desde 2020 e integra os comercializáveis (BANCO "
             "CENTRAL DO BRASIL, 2019), embora figurasse entre os monitorados na classificação descrita por "
             "Figueiredo e Ferreira (2002); só os não comercializáveis excluem todos os combustíveis.")
p, = com("que excluem combustíveis por construção. Uma resposta nesses subgrupos")
R.substituir(p, "que excluem combustíveis por construção. Uma resposta nesses subgrupos representa variação de preços "
                "fora dos combustíveis, sem que seja preciso subtrair coisa alguma.",
             "que excluem gasolina e óleo diesel; os não comercializáveis excluem também o etanol, que integra os "
             "comercializáveis (seção 3.1). Uma resposta dos não comercializáveis representa, portanto, variação de "
             "preços fora dos combustíveis, sem que seja preciso subtrair coisa alguma; a dos comercializáveis inclui "
             "a do etanol.")
p, = com("D e E: subgrupos sem combustíveis")
R.substituir(p, "D e E: subgrupos sem combustíveis,", "D e E: subgrupos sem gasolina e diesel (os comercializáveis incluem "
                                                     "o etanol),")
p, = com("Nos subgrupos sem combustíveis, os testes de igualdade")
R.substituir(p, "Nos subgrupos sem combustíveis,", "Nos subgrupos de comercializáveis e não comercializáveis,")
p, = com("nos subgrupos de comercializáveis e não comercializáveis, que excluem combustíveis.")
R.substituir(p, "que excluem combustíveis.", "que excluem gasolina e diesel (o etanol integra os comercializáveis).")
p, = com("Combustíveis são classificados como preços monitorados e não integram nenhum dos dois subgrupos.")
R.substituir(p, "Combustíveis são classificados como preços monitorados e não integram nenhum dos dois subgrupos.",
             "Gasolina e óleo diesel são preços monitorados e não integram nenhum dos dois subgrupos; o etanol, preço "
             "livre, integra os comercializáveis (BANCO CENTRAL DO BRASIL, 2019).")

# seção 4.8, inserção minha ainda pendente
alvos = [t for ins in corpo.iter(qn("w:ins")) if ins.get(qn("w:author")) == "Claude"
         for t in ins.iter(qn("w:t")) if "e os comercializáveis ficam próximos de zero" in (t.text or "")]
assert len(alvos) == 1
alvos[0].text = alvos[0].text.replace("e os comercializáveis ficam próximos de zero",
                                      "e os comercializáveis, que incluem o etanol, ficam próximos de zero")

# referência nova, antes da primeira do Banco Central
ref, = com("BANCO CENTRAL DO BRASIL. Exchange rate pass-through")
novo = copy.deepcopy(ref)
usados = {e.get(W14_PARAID) for e in corpo.iter() if e.get(W14_PARAID)}
rnd = random.Random(20261018)
v = "%08X" % rnd.randrange(1, 0x7FFFFFFF)
while v in usados:
    v = "%08X" % rnd.randrange(1, 0x7FFFFFFF)
novo.set(W14_PARAID, v)
modelo = novo.find(".//" + qn("w:r"))
for el in list(novo):
    if el.tag != qn("w:pPr"):
        novo.remove(el)
ppr = novo.find(qn("w:pPr"))
for x in ppr.findall(qn("w:rPr")):
    ppr.remove(x)
rpr = ppr.makeelement(qn("w:rPr"), {}); rpr.append(R._marca("w:ins")); ppr.append(rpr)
ins = R._marca("w:ins")
ins.append(R._run_novo(modelo, "BANCO CENTRAL DO BRASIL. Atualizações da estrutura de ponderação do IPCA e "
                               "repercussão nas suas classificações. In: Relatório de Inflação, Brasília, dez. 2019. "
                               "Estudo Especial n. 69/2019."))
novo.append(ins)
ref.addprevious(novo)

doc.save(SAI); print("salvo", SAI)
