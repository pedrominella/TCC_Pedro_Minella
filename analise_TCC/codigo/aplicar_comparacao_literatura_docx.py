# -*- coding: utf-8 -*-
"""
Insere a seção 4.9 "O resultado à luz da literatura" no TCC, com a Tabela 8, e renumera a
Síntese para 4.10 (08/10/2026).

Todos os números dos outros trabalhos foram conferidos nos PDFs/textos de
referencias/conferencia_05out2026/fontes e referencias/artigos_pdf:
  Choi et al. (2018): resumo (0,4 p.p. por 10%), seção 5 (cerca de 0,1 p.p. instantâneo, 2000-2015),
                      Tabela 6 (EUA 0,025; Brasil -0,008, último dos 71) e Tabela A.4 (Brasil 0,012 em USD).
  Blanchard e Galí (2007): seção 4 (impacto na inflação ao consumidor semelhante antes e depois de 1984).
  Känzig (2021): 0,4% no pico por 10%; repasse de cerca de 4,5% ao CPI e de 35% à energia após um ano.
  Baba e Lee (2022): núcleo 0,002 no primeiro trimestre, 0,03 em três anos.
  López et al. (2026): efeitos indiretos de cerca de 75% da resposta acumulada em três anos.
  Gelos e Ustyugova (2017): dispersão maior do repasse de combustíveis em economias em desenvolvimento,
                      possivelmente por controles de preço e subsídios.
  Banco Central do Brasil (2022): 0,20 a 0,27 p.p. (2021) e 0,31 a 0,43 p.p. (2022) por 10% em reais.
Números deste trabalho: Tabela 2, onda2_bandas_placebo_final.xlsx (amostra completa),
onda2_4 (canal) e onda2_7b_voltas_por_regime.xlsx (primeira e segunda volta), reescalados por 10/10,7.

Backup antes: historico/3_versoes_e_backups/backups_do_TCC/_backup_TCC_pre_comparacao_literatura.docx
"""
import copy
from pathlib import Path
import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

RAIZ = Path(__file__).resolve().parents[2]
ARQ = RAIZ / "TCC_PEDRO_MINELLA.docx"
d = docx.Document(ARQ)
P = d.paragraphs


def conferir(i, inicio):
    assert P[i].text.startswith(inicio), (i, P[i].text[:80])


conferir(413, "4.8 O resultado em perspectiva internacional")
conferir(416, "A comparação cumpre duas funções")
conferir(417, "4.9 Síntese")
conferir(322, "Tabela 2 – ")
conferir(323, "Nota: coeficientes")
conferir(324, "Fonte: Elaboração própria.")
conferir(118, "Tabela 7 – ")


def clonar(par, texto):
    """Cópia de um parágrafo existente (estilo, recuo, fonte), com o texto trocado."""
    novo = copy.deepcopy(par._p)
    runs = novo.findall(qn("w:r"))
    for r in runs[1:]:
        novo.remove(r)
    for t in runs[0].findall(qn("w:t")):
        runs[0].remove(t)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = texto
    runs[0].append(t)
    return novo


CORPO = P[416]
TITULO = P[413]
LEGENDA, NOTA, FONTE = P[322], P[323], P[324]

titulo = "4.9 O resultado à luz da literatura"

p1 = ("Os números deste capítulo ganham sentido quando postos ao lado dos que a literatura revisada no "
      "Capítulo 2 reporta, para o Brasil e para outras economias. A comparação exige uma unidade comum. Como a "
      "maior parte daqueles trabalhos expressa o efeito por alta de 10% no petróleo, os coeficientes daqui, que "
      "correspondem a um choque de um desvio-padrão, equivalente a 10,7%, são reescalados pelo fator 10/10,7. A "
      "Tabela 8 reúne o resultado. Métodos, amostras e medidas de preço diferem entre os estudos, de modo que a "
      "comparação vale como ordem de grandeza, e não como teste de igualdade.")

legenda = "Tabela 8 – O repasse estimado e a literatura, por alta de 10% no preço do petróleo"

nota = ("Nota: os valores deste trabalho são os coeficientes das Tabelas 2, 4 e 5 e da seção 4.5.1 multiplicados "
        "por 10/10,7; elasticidade é a razão entre a variação do preço e a do petróleo. Antes e depois referem-se a "
        "março de 2017. Métodos, amostras e medidas de preço diferem entre os estudos, e a comparação é de ordem de "
        "grandeza.")
fonte = "Fonte: Elaboração própria a partir dos trabalhos citados."

p2 = ("A primeira constatação é que o regime posterior se parece com as economias da literatura internacional, e "
      "o anterior não. Choi et al. (2018) estimam, em painel de 72 economias entre 1970 e 2015, que uma alta de 10% "
      "eleva a inflação em cerca de 0,4 ponto percentual no ano do choque. O análogo desta amostra em doze meses é "
      "de 0,10 na especificação de amostra completa, mas de −0,06 no regime anterior e de 0,25 no posterior. Para os "
      "Estados Unidos, Känzig (2021) obtém, com choques de oferta identificados por anúncios da OPEP, elevação de "
      "0,4% no nível de preços ao consumidor no pico da resposta, o que corresponde a repasse de cerca de 4,5%; no "
      "regime posterior, o IPCA atinge 0,33 ponto no pico, um mês após o choque, repasse de 3,3%. Sob a paridade, a "
      "magnitude brasileira é, portanto, da mesma ordem da norte-americana e algo inferior à média histórica do "
      "painel de Choi et al. (2018), o que é compatível com o declínio do repasse ao longo do tempo que aqueles "
      "autores documentam.")

p3 = ("O mesmo artigo traz evidência específica sobre o Brasil que corrobora o resultado por via independente. Com "
      "dados mensais de 2000 a 2015, Choi et al. (2018) estimam o efeito instantâneo do petróleo sobre a inflação "
      "país a país, e o Brasil aparece em último lugar entre as 71 economias, com coeficiente de −0,008, equivalente "
      "a −0,08 ponto por alta de 10%, contra cerca de 0,1 ponto em média nas economias avançadas e nas em "
      "desenvolvimento e 0,25 nos Estados Unidos, o maior entre as avançadas. A janela daquele exercício coincide "
      "quase inteiramente com o regime anterior deste trabalho, em que o efeito no mês do choque é de 0,005 ponto "
      "por alta de 10%; no regime posterior ele passa a 0,16, entre a média internacional e o valor norte-americano. "
      "Registre-se que, medido o petróleo em dólares e não em moeda local, o coeficiente brasileiro daquele trabalho "
      "passa a 0,012, ao passo que o norte-americano permanece em 0,025, e que a especificação não inclui os "
      "controles domésticos usados aqui. A sensibilidade do coeficiente brasileiro à moeda de medida é compatível "
      "com a ausência de repasse sistemático naquele período.")

p4 = ("A comparação com os Estados Unidos também esclarece onde a mudança ocorreu. Blanchard e Galí (2007) mostram "
      "que, de antes para depois de 1984, quase todas as respostas norte-americanas ao petróleo se atenuaram, com "
      "uma exceção: a resposta da inflação ao consumidor no impacto permaneceu muito semelhante entre os períodos, o "
      "que os autores atribuem ao reflexo mecânico do petróleo no componente de energia do índice; o que diminuiu "
      "foi a persistência. No Brasil ocorre o inverso. O que muda entre os regimes é precisamente o impacto, que "
      "passa de 0,005 para 0,16 ponto por alta de 10%, e a primeira volta responde por 63% do efeito no mês do "
      "choque, como registra a seção 4.5.1. Onde o preço do combustível é de mercado, o elo mecânico é estável e o "
      "que varia ao longo do tempo é a persistência; onde o preço é administrado, é o próprio elo mecânico que a "
      "regra liga ou desliga. A gasolina ao consumidor ilustra o ponto: no regime posterior, sua elasticidade ao "
      "petróleo chega a 0,37 no mês seguinte ao choque, da mesma ordem dos cerca de 35% que Känzig (2021) reporta "
      "para o componente de energia do índice norte-americano após um ano.")

p5 = ("Quanto ao mecanismo, a literatura europeia oferece o termo de comparação para a decomposição da seção 4.5. "
      "Baba e Lee (2022) encontram repasse ao núcleo que parte de 0,002 no primeiro trimestre e chega a 0,03 em três "
      "anos, e López et al. (2026), para choques de gás na área do euro, atribuem aos efeitos indiretos cerca de 75% "
      "da resposta acumulada da inflação cheia ao fim de três anos. No regime posterior brasileiro, a segunda volta "
      "responde por 87% do efeito em seis meses e por 80% em doze, com elasticidade de 0,015 e 0,020 nesses "
      "horizontes. A direção é a mesma, com a parcela de custos crescendo até dominar, mas aqui o deslocamento se "
      "completa em meses, e não em anos. A comparação requer cautela redobrada: a segunda volta deste trabalho é "
      "obtida por subtração dos pesos da cesta, não é significativa em todos os horizontes e não equivale ao núcleo "
      "examinado por aqueles autores.")

p6 = ("A literatura de corte transversal antecipa o mecanismo que o caso brasileiro exibe ao longo do tempo. Gelos "
      "e Ustyugova (2017) observam que o repasse dos combustíveis é bem mais disperso entre as economias em "
      "desenvolvimento do que entre as avançadas e sugerem que a dispersão reflete o uso de controles de preço e de "
      "subsídios; Choi et al. (2018) e López et al. (2026) também associam o tamanho do repasse à regulação dos "
      "preços domésticos de energia; e a seção 4.8 mostra, com a base de Akcura (2025), que os países de preço "
      "desregulado repassam duas vezes e meia mais que os de preço regulado. O Brasil oferece a contrapartida que "
      "nenhum corte transversal pode oferecer: a mesma economia observada nos dois extremos, com a passagem de um a "
      "outro datada pelos próprios dados.")

p7 = ("Na literatura brasileira, o termo de comparação mais direto é o do Banco Central do Brasil (2022), que calibra "
      "o efeito de uma alta de 10% do petróleo em reais sobre o IPCA, via gasolina, entre 0,20 e 0,27 ponto com os "
      "pesos de 2021 e entre 0,31 e 0,43 com os de 2022. O canal dos combustíveis estimado aqui para o regime "
      "posterior, de 0,23 ponto por alta de 10% em três meses, situa-se dentro do primeiro intervalo, ao passo que o "
      "do regime anterior, de 0,003, fica muito abaixo dele. A proximidade deve ser lida com as ressalvas da seção "
      "4.5.2: a calibração mede o petróleo em reais e considera apenas a gasolina, enquanto o canal estimado usa o "
      "Brent em dólares, com o câmbio entre os controles, e inclui diesel e etanol. Ainda assim, ela indica que a "
      "hipótese de repasse integral à refinaria, na qual a calibração se apoia, descreve o regime posterior e não o "
      "anterior. Os trabalhos que documentaram a contenção por via contábil, Almeida, Oliveira e Losekann (2015) e "
      "Azevedo e Serigati (2015), descrevem o mecanismo pelo qual o repasse anterior ficou próximo de zero, e Jesus "
      "Jr., Uchôa e Fernandes (2025) concluem que manter o preço abaixo da referência pouco afeta a inflação, "
      "resultado coerente com o regime anterior estimado aqui. O que este trabalho acrescenta é a datação da "
      "mudança pelos próprios dados, a medida do canal e a extensão da amostra até 2025.")

p8 = ("Em conjunto, a literatura confirma a leitura do capítulo por três vias distintas. Sob a paridade, o Brasil "
      "exibe magnitudes da ordem das reportadas para os Estados Unidos e para o painel internacional; antes dela, "
      "aparecia em último lugar na comparação internacional de Choi et al. (2018); e a estimativa de amostra "
      "completa, de 0,10 ponto em doze meses, fica abaixo das referências internacionais e não descreve nenhum dos "
      "dois regimes. É a regra de formação de preços que explica por que o Brasil passou de um extremo a outro da "
      "distribuição internacional sem mudança correspondente em seu regime monetário.")

linhas = [
    ("Estudo", "Contexto", "Estimativa do estudo", "Este trabalho, mesma unidade"),
    ("Choi et al. (2018)", "72 economias, 1970 a 2015, dados anuais",
     "0,4 p.p. na inflação do ano do choque",
     "12 meses: 0,10 na amostra completa; −0,06 antes e 0,25 depois"),
    ("Choi et al. (2018), Tabela 6", "71 economias, dados mensais, 2000 a 2015",
     "Efeito instantâneo: cerca de 0,1 p.p. em média; EUA 0,25; Brasil −0,08, o último",
     "Mês do choque: 0,005 antes e 0,16 depois"),
    ("Blanchard e Galí (2007)", "Estados Unidos, antes e depois de 1984",
     "Impacto na inflação ao consumidor semelhante nos dois períodos; persistência menor depois",
     "É o impacto que muda: de 0,005 para 0,16"),
    ("Känzig (2021)", "Estados Unidos, choques de oferta da OPEP",
     "Preços ao consumidor +0,4% no pico (repasse de 4,5%); energia: 35% em um ano",
     "IPCA: 0,33 no pico (repasse de 3,3%); gasolina ao consumidor: elasticidade de 0,37 no mês seguinte, depois"),
    ("Baba e Lee (2022)", "Economias europeias",
     "Núcleo: elasticidade de 0,002 no primeiro trimestre a 0,03 em três anos",
     "Segunda volta: elasticidade de 0,015 em seis meses e 0,020 em doze, depois"),
    ("López et al. (2026)", "Área do euro, choques de gás",
     "Efeitos indiretos: cerca de 75% da resposta em três anos",
     "Segunda volta: 87% do efeito em seis meses e 80% em doze, depois"),
    ("Banco Central do Brasil (2022)", "Brasil, calibração via gasolina, petróleo em reais",
     "0,20 a 0,27 p.p. no IPCA com pesos de 2021; 0,31 a 0,43 com os de 2022",
     "Canal dos combustíveis em três meses: 0,003 antes e 0,23 depois"),
]
LARG = [1700, 1750, 2350, 2251]   # soma 8051 dxa, a mesma largura da Tabela 2


def celula(texto, larg, negrito, topo, base):
    tc = OxmlElement("w:tc")
    pr = OxmlElement("w:tcPr")
    w = OxmlElement("w:tcW"); w.set(qn("w:w"), str(larg)); w.set(qn("w:type"), "dxa"); pr.append(w)
    bd = OxmlElement("w:tcBorders")
    for lado, sz in (("top", topo), ("left", None), ("bottom", base), ("right", None)):
        e = OxmlElement(f"w:{lado}")
        if sz:
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        else:
            e.set(qn("w:val"), "nil")
        bd.append(e)
    pr.append(bd); tc.append(pr)
    p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    sp = OxmlElement("w:spacing"); sp.set(qn("w:after"), "20"); sp.set(qn("w:line"), "240"); sp.set(qn("w:lineRule"), "auto")
    ppr.append(sp); p.append(ppr)
    r = OxmlElement("w:r"); rpr = OxmlElement("w:rPr")
    if negrito:
        rpr.append(OxmlElement("w:b"))
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "18"); rpr.append(sz); r.append(rpr)
    t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = texto; r.append(t)
    p.append(r); tc.append(p)
    return tc


tbl = OxmlElement("w:tbl")
tbl.append(copy.deepcopy(d.tables[0]._tbl.tblPr))
grid = OxmlElement("w:tblGrid")
for lg in LARG:
    g = OxmlElement("w:gridCol"); g.set(qn("w:w"), str(lg)); grid.append(g)
tbl.append(grid)
for k, lin in enumerate(linhas):
    tr = OxmlElement("w:tr")
    trpr = OxmlElement("w:trPr"); trpr.append(OxmlElement("w:cantSplit"))
    if k == 0:
        trpr.append(OxmlElement("w:tblHeader"))
    tr.append(trpr)
    topo = 12 if k == 0 else None
    base = 6 if k == 0 else (12 if k == len(linhas) - 1 else None)
    for txt, lg in zip(lin, LARG):
        tr.append(celula(txt, lg, k == 0, topo, base))
    tbl.append(tr)

# monta a sequência e insere depois do último parágrafo da seção 4.8
seq = [clonar(TITULO, titulo), clonar(CORPO, p1), clonar(LEGENDA, legenda), tbl,
       clonar(NOTA, nota), clonar(FONTE, fonte),
       clonar(CORPO, p2), clonar(CORPO, p3), clonar(CORPO, p4), clonar(CORPO, p5),
       clonar(CORPO, p6), clonar(CORPO, p7), clonar(CORPO, p8)]
ancora = CORPO._p
for el in seq:
    ancora.addnext(el)
    ancora = el

# Síntese passa a 4.10
for r in P[417].runs:
    if "4.9" in r.text:
        r.text = r.text.replace("4.9", "4.10")
        break
assert P[417].text.startswith("4.10 Síntese"), P[417].text

# lista de tabelas
P[118]._p.addnext(clonar(P[118], legenda))

d.save(ARQ)
print("salvo", ARQ)
