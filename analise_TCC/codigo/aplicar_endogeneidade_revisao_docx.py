# -*- coding: utf-8 -*-
"""
Assume a endogeneidade do choque e justifica a ausência de instrumento forte (14/10/2026).

Pedido do autor: tratar o choque e os controles como endógenos, declarar que os coeficientes
são associações condicionais e justificar a não adoção de variável instrumental pela fraqueza
dos instrumentos testados. Todas as mudanças entram como alterações controladas (w:ins/w:del,
autor "Claude"), para serem aceitas ou rejeitadas uma a uma no Word.

Números usados, todos já reportados no trabalho ou nos resultados:
  F da surpresa de Känzig entre 2,9 e 6,8 (Apêndice K.1); F das margens de refino entre 0,25 e
  5,45 (Tabela G.1); F da cotação cheia da gasolina do Golfo no período da paridade, 13,65, e
  2,55 com o Brent como controle (resultados/onda4_4_paridade_iv.xlsx, primeiro_estagio, h = 0).

Uso: python aplicar_endogeneidade_revisao_docx.py [entrada saida]  (sem argumentos: o arquivo de revisão)
Backup: historico/3_versoes_e_backups/backups_do_TCC/_backup_revisao_pre_endogeneidade.docx
"""
import copy, sys
from datetime import datetime, timezone
from pathlib import Path
import docx
from docx.oxml.ns import qn

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 2 else RAIZ / "TCC_PEDRO_MINELLA_revisao_com_alteracoes.docx"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else ENTRADA
AUTOR = "Claude"
DATA = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

doc = docx.Document(ENTRADA)
body = doc.element.body
proximo_id = [max([int(e.get(qn("w:id"))) for e in body.iter() if e.get(qn("w:id")) is not None
                   and e.get(qn("w:id")).lstrip("-").isdigit()] + [0]) + 1000]


def _novo_id():
    proximo_id[0] += 1
    return str(proximo_id[0])


def _marca(tag):
    el = body.makeelement(qn(tag), {qn("w:id"): _novo_id(), qn("w:author"): AUTOR, qn("w:date"): DATA})
    return el


def _runs(p):
    """Runs de texto do parágrafo fora de revisões e de fórmulas, na ordem do documento."""
    saida = []
    for r in p.iter(qn("w:r")):
        anc = {a.tag for a in r.iterancestors()}
        if anc & {qn("w:ins"), qn("w:del"), "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath"}:
            continue
        if r.find(qn("w:t")) is not None:
            saida.append(r)
    return saida


def _texto(r):
    return "".join(t.text or "" for t in r.findall(qn("w:t")))


def _partir(r, k):
    """Divide o run r no caractere k; devolve (esquerda, direita)."""
    s = _texto(r)
    if k <= 0 or k >= len(s):
        return (None, r) if k <= 0 else (r, None)
    assert len(r.findall(qn("w:t"))) == 1, "run com vários w:t"
    novo = copy.deepcopy(r)
    r.find(qn("w:t")).text = s[:k]
    novo.find(qn("w:t")).text = s[k:]
    for t in (r.find(qn("w:t")), novo.find(qn("w:t"))):
        t.set(XML_SPACE, "preserve")
    r.addnext(novo)
    return r, novo


def _localizar(p, trecho):
    runs = _runs(p)
    texto = "".join(_texto(r) for r in runs)
    assert texto.count(trecho) == 1, ("trecho não único ou ausente", trecho[:60], texto.count(trecho))
    ini = texto.index(trecho)
    return runs, ini, ini + len(trecho)


def _isolar(p, ini, fim):
    """Garante fronteiras de run em ini e fim; devolve os runs que cobrem [ini, fim)."""
    for pos in (ini, fim):
        acc = 0
        for r in _runs(p):
            n = len(_texto(r))
            if acc < pos < acc + n:
                _partir(r, pos - acc)
                break
            acc += n
    dentro, acc = [], 0
    for r in _runs(p):
        n = len(_texto(r))
        if acc >= ini and acc + n <= fim and n:
            dentro.append(r)
        acc += n
    return dentro


def _run_inserido(modelo, texto):
    r = copy.deepcopy(modelo)
    for filho in list(r):
        if filho.tag != qn("w:rPr"):
            r.remove(filho)
    t = r.makeelement(qn("w:t"), {})
    t.text = texto
    t.set(XML_SPACE, "preserve")
    r.append(t)
    return r


def substituir(p, antigo, novo):
    _, ini, fim = _localizar(p, antigo)
    alvo = _isolar(p, ini, fim)
    d = _marca("w:del")
    alvo[0].addprevious(d)
    for r in alvo:
        d.append(r)
        for t in r.findall(qn("w:t")):
            t.tag = qn("w:delText")
            t.set(XML_SPACE, "preserve")
    if novo:
        i = _marca("w:ins")
        i.append(_run_inserido(alvo[0], novo))
        d.addnext(i)


def inserir_depois(p, ancora, novo):
    _, ini, fim = _localizar(p, ancora)
    alvo = _isolar(p, ini, fim)
    i = _marca("w:ins")
    i.append(_run_inserido(alvo[-1], novo))
    alvo[-1].addnext(i)


def paragrafo(inicio):
    achados = [p for p in doc.paragraphs if p.text.startswith(inicio)]
    assert len(achados) == 1, (inicio[:50], len(achados))
    p = achados[0]._p
    assert p.find(".//" + qn("w:ins")) is None and p.find(".//" + qn("w:del")) is None, ("já tem revisão", inicio[:50])
    return p


# ------------------------------------------------------------------ Resumo e abstract
p = paragrafo("Este trabalho estima o repasse do petróleo")
substituir(p, "Os resultados são compatíveis com mudança na transmissão após a paridade de importação, "
              "condicionados à comparabilidade dos vieses.",
           "Como nenhum instrumento externo testado se mostrou forte, o choque é tratado como endógeno e os "
           "coeficientes, como associações condicionais. Os resultados são compatíveis com mudança na "
           "transmissão após a paridade de importação; atribuí-la à regra de preços supõe vieses comparáveis "
           "entre regimes.")

p = paragrafo("This study estimates oil price pass-through")
substituir(p, "The results are consistent with changed transmission following import parity pricing, "
              "conditional on comparable biases.",
           "Since none of the external instruments tested proved strong, the shock is treated as endogenous and "
           "the coefficients as conditional associations. The results are consistent with changed transmission "
           "following import parity pricing; attributing it to the pricing rule assumes comparable biases "
           "across regimes.")

# ------------------------------------------------------------------ Introdução
p = paragrafo("Utilizam-se projeções locais (JORDÀ, 2005)")
inserir_depois(p, "e a resposta residual.",
               " Como os instrumentos externos testados se mostraram fracos, o choque é tratado como endógeno, e o "
               "contraste entre regimes é lido como efeito da regra sob a hipótese de vieses comparáveis nos dois "
               "períodos.")

# ------------------------------------------------------------------ 3.2
p = paragrafo("Uma limitação da medida adotada é que o modelo não separa choques")
substituir(p, "choques de oferta e de demanda. Ela é tratada na seção 4.6, e a resposta principal é que o viés pode "
              "se cancelar no contraste entre regimes se for semelhante nos dois períodos. A verificação por "
              "instrumento externo, no Apêndice K.1, é inconclusiva por falta de potência.",
           "choques de oferta e de demanda, de modo que o choque é tratado como endógeno. A seção 3.4 discute as "
           "consequências dessa escolha: os coeficientes medem associações condicionais, e o viés pode se cancelar "
           "no contraste entre regimes se for semelhante nos dois períodos. A tentativa de identificação por "
           "instrumento externo, nos Apêndices G e K.1, não encontrou instrumento forte o bastante para "
           "estimação em dois estágios.")

# ------------------------------------------------------------------ 3.4
p = paragrafo("Convém explicitar o que esta especificação identifica")
substituir(p, "Convém explicitar o que esta especificação identifica, dado que ela é estimada por mínimos quadrados "
              "ordinários e que o choque do petróleo, como qualquer variável macroeconômica, não é experimentalmente "
              "atribuído. O objeto de interesse do trabalho não é o nível de β_pós(h), e sim a diferença entre "
              "β_pós(h) e β_pré(h). A interpretação da diferença exige uma hipótese de identificação própria.",
           "Convém explicitar o que esta especificação identifica. O trabalho assume que o choque do petróleo é "
           "endógeno e que os controles não o tornam exógeno: a cotação do Brent responde à demanda mundial, que "
           "também afeta a inflação brasileira por outros canais, e câmbio e expectativas respondem ao próprio "
           "choque dentro do mês, razão pela qual entram apenas defasados. A "
           "resposta usual a esse problema seria instrumentar o choque, mas nenhum dos instrumentos testados se "
           "mostrou forte o bastante. A surpresa de oferta de Känzig (2021) tem estatística F de primeiro estágio "
           "entre 2,9 e 6,8, e as margens de refino da gasolina e do diesel ficam entre 0,25 e 5,45, todas abaixo "
           "do limiar convencional de 10 (STAIGER; STOCK, 1997), como reportam os Apêndices G e K.1. Com "
           "instrumento fraco, o estimador de dois estágios é viesado na direção do de mínimos quadrados e sua "
           "inferência convencional deixa de ser confiável. O trabalho mantém, por isso, a estimação por mínimos "
           "quadrados ordinários e não atribui interpretação causal aos coeficientes β_pré(h) e β_pós(h): eles "
           "medem a resposta média da inflação à variação observada do Brent, condicionada aos controles. A "
           "leitura da diferença entre eles como efeito da regra de preços exige uma hipótese adicional, "
           "explicitada a seguir.")

p = paragrafo("Suponha-se que o choque esteja correlacionado")
substituir(p, "Suponha-se que o choque esteja correlacionado com determinantes da inflação omitidos da equação, de "
              "modo que cada coeficiente estimado incorpore um termo de viés.",
           "Admitida a endogeneidade, o choque está correlacionado com determinantes da inflação omitidos da "
           "equação, e cada coeficiente estimado incorpora um termo de viés.")

p = paragrafo("O asterisco denota estimador")
substituir(p, "O asterisco denota estimador, e a sua ausência, o parâmetro populacional.",
           "O asterisco denota o limite em probabilidade do estimador, e a sua ausência, o parâmetro "
           "populacional; v_pré(h) e v_pós(h) são os vieses de cada regime.")
substituir(p, "Há uma analogia com diferenças em diferenças: a contaminação sobrevive em cada termo isolado e "
              "desaparece no contraste, desde que seja comum. Essa hipótese pode permitir identificar o contraste "
              "sem exigir exogeneidade de cada coeficiente: não se afirma que o choque seja exógeno, e sim que a "
              "endogeneidade que porventura exista não se alterou na data de corte.",
           "A igualdade dos vieses não decorre de usar a mesma medida de choque nos dois períodos: ela exige que "
           "a covariância entre o choque e os determinantes omitidos, em proporção à variância do choque, e o "
           "efeito desses determinantes sobre a inflação sejam os mesmos nos dois regimes. Trata-se de hipótese, "
           "e não de resultado. Sem ela, a diferença estimada continua a medir a mudança na associação entre o "
           "petróleo e a inflação doméstica, mas não pode ser atribuída exclusivamente à regra de preços. Não se "
           "afirma, portanto, que o choque seja exógeno, e sim que a leitura causal do contraste depende de a "
           "endogeneidade não se ter alterado na data de corte.")

p = paragrafo("Essa hipótese é examinada por evidências indiretas")
substituir(p, "Essa hipótese é examinada por evidências indiretas, mas não é plenamente testável com os exercícios "
              "disponíveis.",
           "As evidências disponíveis sobre ela são indiretas.")

p = paragrafo("Resta registrar o que a hipótese não cobre")
substituir(p, "que a estratégia não identifica com a mesma força que a diferença,",
           "que sob endogeneidade não tem interpretação causal,")

# ------------------------------------------------------------------ 4.6
p = paragrafo("A mudança doméstica na frequência de reajustes")
substituir(p, "A verificação externa de Känzig é inconclusiva (Apêndice K.1).",
           "Os instrumentos externos testados são fracos (Apêndices G e K.1), e por isso essa possibilidade não "
           "pode ser afastada por estimação em dois estágios.")

# ------------------------------------------------------------------ Conclusão
p = paragrafo("A interpretação enfrenta limites:")
substituir(p, "vieses comparáveis entre regimes são uma hipótese;",
           "o choque é tratado como endógeno, pois nenhum instrumento testado alcançou estatística F de 10, e os "
           "coeficientes medem associações condicionais; vieses comparáveis entre regimes, necessários para ler o "
           "contraste como efeito da regra, são uma hipótese;")

# ------------------------------------------------------------------ Apêndice G
p = paragrafo("O Apêndice K.1 registra que a identificação do trabalho")
substituir(p, "O Apêndice K.1 registra que a identificação do trabalho não repousa sobre variável instrumental. Este "
              "apêndice reporta as estatísticas dos instrumentos externos que foram testados e descartados,",
           "O trabalho trata o choque do petróleo como endógeno e não adota variável instrumental, porque nenhum "
           "dos instrumentos testados se mostrou forte o bastante (seção 3.4). Este apêndice reporta as "
           "estatísticas desses instrumentos,")

p = paragrafo("Nenhum dos instrumentos alcança o limiar convencional.")
inserir_depois(p, "Nenhum dos instrumentos alcança o limiar convencional.",
               " O único candidato que o supera é a cotação cheia da gasolina no Golfo, sem descontar o Brent, com "
               "F de 13,65 para a gasolina ao consumidor no período da paridade, de março de 2017 a maio de 2023. "
               "Ela carrega, porém, o próprio preço do petróleo, que alcança a inflação brasileira por outros "
               "canais além do combustível, e por isso não satisfaz a restrição de exclusão; com o Brent incluído "
               "como controle, a estatística cai para 2,55.")

# ------------------------------------------------------------------ Apêndice K.1
p = paragrafo("O exercício é, portanto, projeção sobre o choque estimado por Känzig")
substituir(p, "com o choque entrando como regressor e a identificação repousando sobre",
           "com o choque tratado como endógeno e a leitura causal repousando sobre")

p = paragrafo("Verificação análoga foi conduzida com instrumentos construídos")
substituir(p, "e a identificação repousa sobre a comparação entre regimes.",
           "e por isso o choque é tratado como endógeno, com a leitura causal restrita ao contraste entre "
           "regimes sob a hipótese da seção 3.4.")

doc.save(SAIDA)
print("salvo", SAIDA)
