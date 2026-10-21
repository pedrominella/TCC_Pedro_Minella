# -*- coding: utf-8 -*-
"""
Figura 5 em linha e numeração das figuras pela ordem da primeira citação (14/10/2026).

1) Figura 5: troca a imagem de barras pela de linha (fig_distribuicao_temporal.py) e ajusta a nota.
2) Renumera para que a ordem siga a primeira citação no texto, regra declarada no Apêndice L:
     4 -> 3   Resposta dos preços de combustíveis (corpo, citada antes)
     3 -> 4   Resposta do IPCA Transportes (Apêndice L)
     9 -> 8   Primeira e segunda volta (corpo, citada antes)
     8 -> 9   Os dois caminhos do choque (Apêndice L)
    13 -> 12  Contrafactual por ano civil (corpo, 4.7.5, que passa a citá-la)
    14 -> 13  O repasse brasileiro na distribuição internacional (citada na 4.8)
    12 -> 14  Composição dos choques (citada só no Apêndice K)
   e reordena a lista de figuras.
3) Acrescenta na 4.7.5 a citação da figura do contrafactual.

Uso: python aplicar_figuras_ordem_docx.py [arquivo_entrada arquivo_saida]  (sem argumentos: o TCC)
Backup feito antes: historico/3_versoes_e_backups/backups_do_TCC/_backup_TCC_pre_ordem_figuras.docx
"""
import io, re, sys
from pathlib import Path
import docx
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 2 else RAIZ / "TCC_PEDRO_MINELLA.docx"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else ENTRADA
FIG5 = RAIZ / "analise_TCC" / "figuras" / "distribuicao_temporal_mensal.png"
MAPA = {4: 3, 3: 4, 9: 8, 8: 9, 13: 12, 14: 13, 12: 14}

doc = docx.Document(ENTRADA)
P = doc.paragraphs


def titulo(n):
    """Título (sem número) da legenda de corpo da figura n, antes da troca."""
    for i, p in enumerate(P[140:], start=140):
        m = re.match(rf"Figura {n} – (.*)", p.text)
        if m and P[i + 1]._p.xpath(".//a:blip"):
            return m.group(1)
    raise ValueError(n)


# conferência do estado de partida
esperado = {3: "Resposta do IPCA Transportes", 4: "Resposta dos preços de combustíveis",
            8: "Os dois caminhos do choque", 9: "Primeira e segunda volta", 12: "Composição dos choques",
            13: "Contribuição dos choques de petróleo ao IPCA", 14: "O repasse brasileiro"}
for n, ini in esperado.items():
    assert titulo(n).startswith(ini), (n, titulo(n))

# ---------------------------------------------------------------- 1) renumeração (um só passe)
PADRAO = re.compile(r"\b(Figuras?)(\s+)(\d+(?:(?:,\s*|\s+e\s+)\d+)*)")


def renum(m):
    return m.group(1) + m.group(2) + re.sub(r"\d+", lambda k: str(MAPA.get(int(k.group()), int(k.group()))), m.group(3))


def todos():
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                yield from c.paragraphs


vistos, n_mud = set(), 0
for p in todos():
    if id(p._p) in vistos:
        continue
    vistos.add(id(p._p))
    alvo = PADRAO.sub(renum, p.text)
    if alvo != p.text:
        for r in p.runs:
            r.text = PADRAO.sub(renum, r.text)
        assert p.text == alvo, ("citação partida entre trechos", p.text[:100])
        n_mud += 1
print("parágrafos renumerados:", n_mud)

# ---------------------------------------------------------------- 2) reordenar a lista de figuras
P = doc.paragraphs
lista = [(i, p) for i, p in enumerate(P[:200]) if re.match(r"Figura \d+ –", p.text)]
idx = [i for i, _ in lista]
assert idx == list(range(idx[0], idx[0] + len(idx))), "a lista de figuras não é contígua"
elems = sorted(((int(re.match(r"Figura (\d+)", p.text).group(1)), p._p) for _, p in lista), key=lambda t: t[0])
assert [n for n, _ in elems] == list(range(1, 15)), [n for n, _ in elems]
ancora = lista[0][1]._p.getprevious()
for _, el in elems:
    el.getparent().remove(el)
for _, el in elems:
    ancora.addnext(el)
    ancora = el

# ---------------------------------------------------------------- 3) Figura 5 em linha
P = doc.paragraphs
[i5] = [i for i, p in enumerate(P[140:], start=140) if p.text.startswith("Figura 5 – ") and P[i + 1]._p.xpath(".//a:blip")]
pimg = P[i5 + 1]
rid = pimg._p.xpath(".//a:blip/@r:embed")[0]
novo = FIG5.read_bytes()
doc.part.related_parts[rid]._blob = novo
w, h = Image.open(io.BytesIO(novo)).size
for ext in pimg._p.xpath(".//wp:extent") + pimg._p.xpath(".//a:ext"):
    ext.set("cy", str(int(int(ext.get("cx")) * h / w)))
nota = P[i5 + 2]
r = nota.runs
assert r[0].text.endswith("As barras representam a resposta da inflação do mês ") and r[1].text == "t+h", [x.text for x in r]
r[0].text = r[0].text.replace("As barras representam", "A linha representa")
assert r[2].text.startswith(", estimada separadamente para cada mês, com intervalo de 90%; em verde")
r[2].text = ", estimada separadamente para cada mês, com faixa de 90%. A resposta acumulada está na Figura 6."

# ---------------------------------------------------------------- 4) citação do contrafactual na 4.7.5
[p20] = [p for p in P if p.text.startswith("Em 2020 o sinal se inverte.")]
assert "Figura" not in p20.text
p20.runs[-1].text = p20.runs[-1].text + " A Figura 12 mostra as trajetórias mensais dos três anos."

# ---------------------------------------------------------------- 5) Figura 8 em cinco painéis (fig_primeira_segunda_volta.py)
FIG8 = RAIZ / "analise_TCC" / "figuras" / "voltas_paineis.png"
P = doc.paragraphs
[i8] = [i for i, p in enumerate(P[140:], start=140)
        if p.text == "Figura 8 – Primeira e segunda volta, e desfechos que excluem combustíveis" and P[i + 1]._p.xpath(".//a:blip")]
pimg8, nota8 = P[i8 + 1], P[i8 + 2]
rid8 = pimg8._p.xpath(".//a:blip/@r:embed")[0]
novo8 = FIG8.read_bytes()
doc.part.related_parts[rid8]._blob = novo8
w8, h8 = Image.open(io.BytesIO(novo8)).size
for ext in pimg8._p.xpath(".//wp:extent") + pimg8._p.xpath(".//a:ext"):
    ext.set("cy", str(int(int(ext.get("cx")) * h8 / w8)))
for pr in pimg8._p.xpath(".//wp:docPr"):
    pr.set("descr", "Efeito total, primeira e segunda volta e subgrupos sem combustíveis, regime posterior a março de 2017")
assert nota8.text.startswith("Nota: ambos os painéis referem-se ao regime posterior"), nota8.text[:60]
nota8.runs[0].text = (
    "Nota: regime posterior a março de 2017, com faixas de 90%, exceto no painel B, cuja faixa é a dos três "
    "cenários de pesos da cesta. Os painéis A a C decompõem o IPCA Geral no desenho comum da Tabela 5, de modo "
    "que o efeito total é a soma da primeira com a segunda volta; os painéis D e E trazem os subgrupos que "
    "excluem combustíveis por construção, com a resposta do IPCA Geral como referência.")
for r in nota8.runs[1:]:
    r.text = ""
[p_cit] = [p for p in P if "o painel direito da Figura 8 a apresenta" in p.text]
for r in p_cit.runs:
    if "o painel direito da Figura 8 a apresenta" in r.text:
        r.text = r.text.replace("o painel direito da Figura 8 a apresenta", "os painéis D e E da Figura 8 a apresentam")
        break
else:
    raise ValueError("citação do painel partida entre trechos")

doc.save(SAIDA)
print("salvo", SAIDA)
