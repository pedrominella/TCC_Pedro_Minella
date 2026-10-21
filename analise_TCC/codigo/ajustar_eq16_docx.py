# -*- coding: utf-8 -*-
"""Quebra a equação (16) da seção 4.8 em duas linhas (ela não cabia numa só), dentro da inserção pendente."""
import sys
from pathlib import Path
import docx
from docx.oxml.ns import qn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor
from omml import Eq
ENT, SAI = sys.argv[1], sys.argv[2]
MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
doc = docx.Document(ENT); R = Revisor(doc); corpo = doc.element.body; E = Eq(corpo)
alvo = [p for p in corpo.iter(qn("w:p")) if any((t.text or "") == "(16)" for t in p.iter(qn("w:t")))
        and p.find(".//" + MATH + "oMath") is not None]
assert len(alvo) == 1
p = alvo[0]; ins = p.find(qn("w:ins")); assert ins.get(qn("w:author")) == "Claude"
runs = ins.findall(qn("w:r")); modelo = runs[0]
for el in list(ins):
    ins.remove(el)
P_ = lambda x: E.d(x); CH = lambda x: E.d(x, "{", "}"); RT = lambda s: E.r(s, reto=True); TX = lambda s: E.r(s, palavra=True)
def r_tab(br=False):
    r = R._run_novo(modelo, ""); r.remove(r.find(qn("w:t")))
    if br:
        r.append(r.makeelement(qn("w:br"), {}))
    r.append(r.makeelement(qn("w:tab"), {}))
    return r
l1 = E.omath("D", P_("t,h"), "=a+", E.sub("π", RT("pré")), P_("h"), "·m", P_("t"), "·1", CH("t<τ"))
l2 = E.omath("+", E.sub("π", RT("pós")), P_("h"), "·m", P_("t"), "·1", CH("t≥τ"), "+", TX("controles"), "+v", P_("t,h"))
for el in [r_tab(), l1, r_tab(br=True), l2, r_tab(), R._run_novo(modelo, "(16)")]:
    ins.append(el)
doc.save(SAI); print("salvo", SAI)
