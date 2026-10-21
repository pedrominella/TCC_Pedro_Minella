# -*- coding: utf-8 -*-
"""
Figura 13 (F do primeiro estágio por horizonte) sai da seção 4.8 e vai para o fim do Apêndice G, para o corpo
voltar a 40 páginas (15/10/2026). Alteração controlada, autor "Claude".

A figura mantém o número, como as demais figuras citadas no corpo e postas em apêndice ("Figura 4, no
Apêndice L"). Os quatro parágrafos do bloco (legenda, imagem, nota e fonte) são inserção minha ainda pendente,
então são apenas deslocados, com as marcas de inserção. Também dentro das minhas inserções: a remissão da 4.8
passa a indicar o apêndice, o último parágrafo do Apêndice G passa a citar a figura e a nota avisa que a
especificação é a da seção 4.8, e não a das Tabelas G.1 e G.2. A imagem fica com 85% da largura, para que a
quebra de página antes do Apêndice H não crie uma folha em branco.

Uso: python aplicar_fig13_apendice_g_docx.py entrada saida
"""
import sys
from pathlib import Path
import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor

ENT, SAI = sys.argv[1], sys.argv[2]
doc = docx.Document(ENT); R = Revisor(doc); corpo = doc.element.body


def vis(p):
    return "".join(t.text or "" for t in p.iter(qn("w:t")) if qn("w:del") not in {a.tag for a in t.iterancestors()})


def so_minha_insercao(p):
    marca = p.find(qn("w:pPr") + "/" + qn("w:rPr") + "/" + qn("w:ins"))
    filhos = [c for c in p if c.tag != qn("w:pPr")]
    return (marca is not None and marca.get(qn("w:author")) == R.autor and filhos
            and all(c.tag == qn("w:ins") and c.get(qn("w:author")) == R.autor for c in filhos))


def editar_insercao(trecho, novo):
    alvos = [t for ins in corpo.iter(qn("w:ins")) if ins.get(qn("w:author")) == R.autor
             for t in ins.iter(qn("w:t")) if trecho in (t.text or "")]
    assert len(alvos) == 1, (trecho[:40], len(alvos))
    alvos[0].text = alvos[0].text.replace(trecho, novo)


leg = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith("Figura 13 – Força dos instrumentos")
       and p.getnext() is not None and p.getnext().find(".//" + qn("w:drawing")) is not None]
assert len(leg) == 1
leg = leg[0]; img = leg.getnext(); nota = img.getnext(); fonte = nota.getnext()
assert vis(nota).startswith("Nota: F robusto (Newey-West)") and vis(fonte) == "Fonte: Elaboração própria."
bloco = [leg, img, nota, fonte]
assert all(so_minha_insercao(p) for p in bloco)

fim_g = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith("Nessa especificação, nenhum dos instrumentos alcança")]
assert len(fim_g) == 1
fim_g = fim_g[0]
quebra = fim_g.getnext()
assert vis(quebra) == "" and quebra.find(".//" + qn("w:br")).get(qn("w:type")) == "page"
assert vis(quebra.getnext()).startswith("APÊNDICE H")

ancora = fim_g
for p in bloco:
    ancora.addnext(p); ancora = p

# em largura cheia, o bloco termina a 4 pt da margem inferior e o parágrafo com a quebra de página antes do
# Apêndice H passa para a folha seguinte, que fica em branco; a 85% da largura, ele cabe
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
for ext in [e for e in list(img.iter(WP + "extent")) + list(img.iter(A + "ext")) if e.get("cx")]:
    ext.set("cx", str(round(int(ext.get("cx")) * 0.85))); ext.set("cy", str(round(int(ext.get("cy")) * 0.85)))

editar_insercao("A Figura 13 mostra a estatística F", "A Figura 13, no Apêndice G, mostra a estatística F")
editar_insercao("a partir do primeiro mês (seção 4.8). Entre os demais",
                "a partir do primeiro mês (seção 4.8 e Figura 13). Entre os demais")
editar_insercao("Antes e depois referem-se a março de 2017.",
                "Antes e depois referem-se a março de 2017. A especificação é a da seção 4.8, e não a das Tabelas "
                "G.1 e G.2.")
doc.save(SAI); print("salvo", SAI)
