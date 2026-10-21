# -*- coding: utf-8 -*-
"""
Terminologia: "primeira volta" e "segunda volta" passam a "primeiro efeito" e "segundo efeito", padrão definido
pelo autor (20/10/2026). Alteração controlada, autor "Claude".

Não é troca cega: a mudança de gênero pede concordância ("a segunda volta não é significativa" passa a "o
segundo efeito não é significativo"; "a tabela não a reporta" passa a "não o reporta"), e expressões como
"efeitos de segunda volta" ou "é quase inteiramente de segunda volta" são reescritas. A introdução define os dois
termos na primeira ocorrência. "O máximo volta a novembro de 2016" é verbo e fica. As Figuras 8 e 9, que trazem
os termos dentro da imagem, são trocadas pelas versões v2 (fig_primeira_segunda_volta.py e
fig_voltas_e_residuos.py, com os mesmos números).

Também, na mesma passagem:
  - a Figura 9 volta a ser citada no texto (a remissão do mestre saiu no corte do ChatGPT; a ABNT exige citação)
  - a nota da Tabela 5 tinha a faixa do regime anterior desatualizada (0,006 a 0,098, p de 0,196 a 0,982); em
    onda2_7b_voltas_por_regime.xlsx, cenário central, horizontes de 0 a 12: total de −0,066 a 0,102 e p de
    0,196 a 0,997

Texto dentro de inserção minha ainda pendente muda dentro da própria inserção.

Uso: python aplicar_primeiro_segundo_efeito_docx.py entrada saida
"""
import copy, sys
from pathlib import Path
import docx
from docx.oxml.ns import qn
from PIL import Image

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from revisao_track import Revisor

ENT, SAI = sys.argv[1], sys.argv[2]
FIGS = AQUI.parent / "figuras"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

doc = docx.Document(ENT); R = Revisor(doc); corpo = doc.element.body


def vis(p):
    return "".join(t.text or "" for t in p.iter(qn("w:t")) if qn("w:del") not in {a.tag for a in t.iterancestors()})


def pars(trecho, n=1, exato=False):
    r = [p for p in corpo.iter(qn("w:p")) if (vis(p) == trecho if exato else trecho in vis(p))]
    assert len(r) == n, (trecho[:50], len(r)); return r


def trocar(p, antigo, novo):
    """Troca controlada; se o trecho está numa inserção minha pendente, muda dentro dela."""
    livre = "".join(R._texto(r) for r in R._runs(p))
    if antigo in livre:
        R.substituir(p, antigo, novo); return
    alvos = [t for ins in p.iter(qn("w:ins")) if ins.get(qn("w:author")) == R.autor
             for t in ins.iter(qn("w:t")) if antigo in (t.text or "")]
    assert len(alvos) == 1, (antigo[:50], len(alvos))
    alvos[0].text = alvos[0].text.replace(antigo, novo)


# ------------------------------------------------------------------ listas, legendas, sumário e título
for p in pars("Figura 8 – Primeira e segunda volta", 2):
    trocar(p, "Primeira e segunda volta", "Primeiro e segundo efeito")
for p in pars("Tabela 5 – Decomposição do repasse ao IPCA Geral em primeira e segunda volta", 2):
    trocar(p, "em primeira e segunda volta", "em primeiro e segundo efeito")
for p in pars("3.7 Decomposição do canal e efeitos de segunda volta", 2):      # sumário e título
    trocar(p, "e efeitos de segunda volta", "e segundo efeito")

# ------------------------------------------------------------------ texto
p, = pars("Medir exige separar o efeito aritmético, que decorre")
trocar(p, "o efeito aritmético, que decorre", "o efeito aritmético, ou primeiro efeito, que decorre")
trocar(p, "do efeito econômico de segunda volta, que é", "do efeito econômico, ou segundo efeito, que é")
p, = pars("dentro daquele, entre primeira e segunda volta.")
trocar(p, "entre primeira e segunda volta.", "entre primeiro e segundo efeito.")
p, = pars("torna-se majoritariamente de segunda volta a partir do terceiro mês")
trocar(p, "torna-se majoritariamente de segunda volta a partir do terceiro mês, ainda que a segunda volta não seja "
          "significativa", "passa a vir majoritariamente do segundo efeito a partir do terceiro mês, ainda que este não "
          "seja significativo")
p, = pars("insumos é segunda volta, por atingir")
trocar(p, "insumos é segunda volta, por atingir", "insumos é segundo efeito, por atingir")
trocar(p, "estimar diretamente a segunda volta, com", "estimar diretamente o segundo efeito, com")
p, = pars("a segunda obtém o efeito de segunda volta por resíduo")
trocar(p, "obtém o efeito de segunda volta por resíduo", "obtém o segundo efeito por resíduo")
p, = pars("O Primeiro Efeito representa 63% do efeito no impacto")
trocar(p, "O Primeiro Efeito representa", "O primeiro efeito representa")
trocar(p, "quando a segunda volta é 0,158 ponto.", "quando o segundo efeito é 0,158 ponto (Figura 9, no Apêndice L).")
trocar(p, "mas a segunda volta não é significativa em três meses", "mas o segundo efeito não é significativo em três meses")
p, = pars("não constitui confirmação independente da segunda volta.")
trocar(p, "independente da segunda volta.", "independente do segundo efeito.")
p, = pars("A evidência de segunda volta decorre da decomposição")
trocar(p, "A evidência de segunda volta decorre", "A evidência do segundo efeito decorre")
p, = pars("Segunda volta: elasticidade de 0,015")
trocar(p, "Segunda volta:", "Segundo efeito:")
p, = pars("Segunda volta: 87% do efeito em seis meses")
trocar(p, "Segunda volta:", "Segundo efeito:")
p, = pars("a parcela estimada de segunda volta chega a 87%")
# "é de" em vez de "chega a": com "do segundo efeito chega a", o parágrafo ganha uma linha e o título da 4.11
# deixa de caber no pé da p. 48, levando o fim do capítulo para a p. 50
trocar(p, "a parcela estimada de segunda volta chega a 87%", "a parcela estimada do segundo efeito é de 87%")
p, = pars("A evidência do trabalho sobre a segunda volta é frágil")
trocar(p, "sobre a segunda volta é frágil", "sobre o segundo efeito é frágil")
p, = pars("A participação da segunda volta cresce na decomposição pontual")
trocar(p, "A participação da segunda volta cresce", "A participação do segundo efeito cresce")
p, = pars("decomposição entre primeira e segunda volta da seção 4.5.1")
trocar(p, "entre primeira e segunda volta da seção 4.5.1", "entre primeiro e segundo efeito da seção 4.5.1")
p, = pars("seu efeito sobre o IPCA é quase inteiramente de segunda volta, por frete.")
trocar(p, "seu efeito sobre o IPCA é quase inteiramente de segunda volta, por frete.",
       "seu impacto sobre o IPCA vem quase inteiramente do segundo efeito, por frete.")

# ------------------------------------------------------------------ Tabela 5: cabeçalho e nota
for velho, novo in [("Primeira volta", "Primeiro efeito"), ("Segunda volta", "Segundo efeito"),
                    ("% primeira volta", "% primeiro efeito")]:
    p, = pars(velho, exato=True)
    assert any(a.tag == qn("w:tbl") for a in p.iterancestors())
    trocar(p, velho, novo)
p, = pars("Os asteriscos referem-se à segunda volta, única coluna")
trocar(p, "Os asteriscos referem-se à segunda volta,", "Os asteriscos referem-se ao segundo efeito,")
trocar(p, "efeito total, primeira e segunda volta é exata.", "efeito total, primeiro e segundo efeito é exata.")
trocar(p, "situa-se entre 0,006 e 0,098 e a segunda volta não é significativa em horizonte algum, com p entre 0,196 "
          "e 0,982, razão pela qual a tabela não a reporta.",
       "situa-se entre −0,066 e 0,102 e o segundo efeito não é significativo em horizonte algum, com p entre 0,196 e "
       "0,997, razão pela qual a tabela não o reporta.")

# ------------------------------------------------------------------ Figura 9: nota
p, = pars("A primeira volta é a contribuição de composição da cesta; a segunda, o encarecimento")
trocar(p, "A primeira volta é a contribuição de composição da cesta; a segunda, o encarecimento",
       "O primeiro efeito é a contribuição de composição da cesta; o segundo, o encarecimento")


# ------------------------------------------------------------------ Figuras 8 e 9: imagens v2
def trocar_imagem(legenda, arquivo, nome, descr):
    leg = [p for p in corpo.iter(qn("w:p")) if vis(p).startswith(legenda)
           and p.getnext() is not None and p.getnext().find(".//" + qn("w:drawing")) is not None]
    assert len(leg) == 1, legenda
    pimg = leg[0].getnext()
    runs = [r for r in pimg.iter(qn("w:r")) if r.find(".//" + qn("w:drawing")) is not None]
    assert len(runs) == 1 and not {a.tag for a in runs[0].iterancestors()} & {qn("w:ins"), qn("w:del")}
    velho = runs[0]
    rid, _ = doc.part.get_or_add_image(str(arquivo))
    novo = copy.deepcopy(velho)
    for blip in novo.iter(A + "blip"):
        blip.set(RNS + "embed", rid)
    larg, alt = Image.open(arquivo).size
    for ext in [e for e in list(novo.iter(WP + "extent")) + list(novo.iter(A + "ext")) if e.get("cx")]:
        ext.set("cy", str(int(int(ext.get("cx")) * alt / larg)))
    maior = max(int(x.get("id")) for x in corpo.iter(WP + "docPr"))
    for pr in novo.iter(WP + "docPr"):
        pr.set("id", str(maior + 1)); pr.set("name", nome); pr.set("descr", descr)
    d = R.apagar_elemento(velho)
    i = R._marca("w:ins"); i.append(novo); d.addnext(i)


trocar_imagem("Figura 8 – Primeiro e segundo efeito", FIGS / "voltas_paineis_v2.png", "Figura 8 v2",
              "Efeito total, primeiro efeito, segundo efeito, comercializáveis e não comercializáveis, regime posterior")
trocar_imagem("Figura 9 – Os dois caminhos do choque", FIGS / "voltas_esquema_v2.png", "Figura 9 v2",
              "Os dois caminhos do choque até o IPCA e a participação do primeiro e do segundo efeito por horizonte")

# nada de "volta" no sentido antigo pode restar
import re
sobra = [vis(p) for p in corpo.iter(qn("w:p")) if re.search(r"(?i)\bvoltas?(\b|\d)", vis(p))]
assert all("o máximo volta a novembro" in s for s in sobra), [s[:80] for s in sobra]
doc.save(SAI); print("salvo", SAI)
