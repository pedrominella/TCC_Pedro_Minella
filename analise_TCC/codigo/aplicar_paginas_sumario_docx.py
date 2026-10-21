# -*- coding: utf-8 -*-
"""
Corrige os números de página do sumário, como alteração controlada (15/10/2026).

As páginas vêm de toc_paginas.tsv, gerado pelo Word com todas as alterações aceitas: para cada linha do
sumário, o marcador do título correspondente e a página em que ele cai (Range.Information(3)). O número
exibido em cada linha é o resultado de um campo PAGEREF; aqui o texto antigo fica marcado como excluído e o
novo como inserido, dentro do próprio resultado do campo. Linhas que eu mesmo inseri e ainda estão pendentes
são corrigidas dentro da inserção.

Uso: python aplicar_paginas_sumario_docx.py entrada saida toc_paginas.tsv
"""
import copy, sys
from pathlib import Path
import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor

ENTRADA, SAIDA, TSV = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pag = {}
for l in open(TSV, encoding="utf-8-sig"):
    a, p, _ = l.rstrip("\n").split("\t", 2)
    pag[a] = p

doc = docx.Document(ENTRADA)
R = Revisor(doc)
n = 0
for p in doc.element.body.iter(qn("w:p")):
    hl = p.find(qn("w:hyperlink"))
    if hl is None or not (hl.get(qn("w:anchor")) or "").startswith("_Toc"):
        continue
    novo = pag[hl.get(qn("w:anchor"))]
    vivos = [t for t in hl.iter(qn("w:t")) if qn("w:del") not in {x.tag for x in t.iterancestors()}]
    t = vivos[-1]
    assert t.text.strip().isdigit(), t.text
    if t.text == novo:
        continue
    run = t.getparent()
    dentro_ins = [x for x in run.iterancestors() if x.tag == qn("w:ins")]
    if dentro_ins:                       # linha inserida por mim, ainda pendente
        assert dentro_ins[0].get(qn("w:author")) == R.autor
        t.text = novo
    else:
        d = R._marca("w:del"); run.addprevious(d); d.append(run)
        for x in run.findall(qn("w:t")):
            x.tag = qn("w:delText")
        i = R._marca("w:ins"); i.append(R._run_novo(run, novo)); d.addnext(i)
    n += 1
doc.save(SAIDA)
print("linhas corrigidas:", n, "| salvo", SAIDA)
