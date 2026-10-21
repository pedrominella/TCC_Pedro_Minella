# -*- coding: utf-8 -*-
"""
Folha de aprovação: o autor ainda não tem o nome do examinador (20/10/2026). O marcador "[NOME DO EXAMINADOR]" sai
como exclusão controlada; ficam a linha de assinatura e "IBMEC (Examinador)", e o parágrafo vazio mantém a
paginação. Quando houver o nome, basta escrevê-lo no lugar.

O comentário do ChatGPT que lembra de preencher o nome, e a minha resposta, estavam ancorados no marcador e
sumiriam ao aceitar a exclusão; passam para "IBMEC (Examinador)", e a minha resposta é atualizada.

Uso: python aplicar_examinador_vazio_docx.py entrada saida
"""
import shutil, sys, tempfile, zipfile
from pathlib import Path
import docx
from docx.oxml.ns import qn
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor

ENT, SAI = sys.argv[1], sys.argv[2]
RESPOSTA = ("Pendente, decisão sua: o nome do segundo examinador ainda não foi definido. O marcador [NOME DO "
            "EXAMINADOR] saiu e a linha de assinatura ficou em branco; escreva o nome acima de \"IBMEC (Examinador)\" "
            "quando a banca for divulgada pelo coordenador de TCC, como prevê o plano de ensino.")

doc = docx.Document(ENT); R = Revisor(doc)
p = R.paragrafo("[NOME DO EXAMINADOR]")
alvo = p.getnext()
assert "".join(t.text or "" for t in alvo.iter(qn("w:t"))) == "IBMEC (Examinador)"
ids = sorted({e.get(qn("w:id")) for e in p.iter(qn("w:commentRangeStart"))})
assert ids == ["0", "1"], ids
R.substituir(p, "[NOME DO EXAMINADOR]", "")

# âncoras dos comentários: início antes do texto de "IBMEC (Examinador)", fim e referência depois
inicios = [e for e in p.iter(qn("w:commentRangeStart"))]
fins = {e.get(qn("w:id")): e for e in p.iter(qn("w:commentRangeEnd"))}
refs = {r.find(qn("w:commentReference")).get(qn("w:id")): r for r in p.findall(qn("w:r"))
        if r.find(qn("w:commentReference")) is not None}
assert sorted(fins) == ids and sorted(refs) == ids
runs = alvo.findall(qn("w:r"))
for e in inicios:
    runs[0].addprevious(e)
ancora = runs[-1]
for i in ids:
    ancora.addnext(fins[i]); fins[i].addnext(refs[i]); ancora = refs[i]
doc.save(SAI)

# minha resposta (comentário 1, autor Claude) no comments.xml
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
tmp = Path(tempfile.mkdtemp()) / "x.docx"
with zipfile.ZipFile(SAI) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
    for item in zi.infolist():
        dados = zi.read(item.filename)
        if item.filename == "word/comments.xml":
            raiz = etree.fromstring(dados)
            c, = [c for c in raiz.iter(W + "comment") if c.get(W + "id") == "1"]
            assert c.get(W + "author") == "Claude"
            ts = list(c.iter(W + "t"))
            assert len(ts) == 1 and ts[0].text.startswith("Pendente, decisão sua: o segundo examinador")
            ts[0].text = RESPOSTA
            dados = etree.tostring(raiz, xml_declaration=True, encoding="UTF-8", standalone=True)
        zo.writestr(item, dados)
shutil.move(str(tmp), SAI)
print("salvo", SAI)
