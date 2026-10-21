# -*- coding: utf-8 -*-
"""
Nova subseção 4.7.6, identificação por variável instrumental, no corpo (14/10/2026).

Pedido do autor: com o corpo em cerca de 35 páginas, trazer para o capítulo de resultados a parte de
variável instrumental, que estava só nos Apêndices C, G e K.1. Tudo entra como alteração controlada,
autor "Claude". Os apêndices ficam com as tabelas e a figura de detalhe.

Números conferidos em 14/10/2026:
  Känzig, surpresa externa: F de 2,92 a 6,80; forma reduzida do IPCA Geral com p de 0,302 a 0,979;
    Anderson-Rubin sempre limitado, de 1,30 a 7,51 vezes a largura do intervalo de dois estágios,
    nunca exclui o zero (resultados/onda2_3_forma_reduzida_ar_final.xlsx); 90 de 270 meses não nulos
    (onda2_3_forma_reduzida_ar.py)
  Känzig, choque estrutural: F de 114,0 a 251,1; IPCA Geral 0,082 em h=1 e 0,133 em h=3 (mesmo arquivo)
  Margens de refino (teste_crack_spread.py, corte de outubro de 2016): gasolina F 5,45 depois e 1,48 antes;
    diesel 0,25 e 0,91; forma reduzida: Transportes 0,357** e 0,620**, IPCA Geral 0,111* em h=1
  Golfo cheio: F 13,65 no período da paridade, 2,55 com o Brent (onda4_4_paridade_iv.xlsx)
  Cardoso e Severnini (2023): F perto de 18 na amostra completa e abaixo de 10 em 2005-2009 e 2010-2016
Correções nos apêndices: "80% mais amplo" não corresponde ao arquivo (mediana 1,6 vez, faixa 1,3 a 7,5)
e o F do choque estrutural vai até 251, não 276.

Uso: python aplicar_secao_iv_docx.py [entrada saida] [pagina_do_sumario]
Backup: historico/3_versoes_e_backups/backups_do_TCC/_backup_revisao_pre_secao_iv.docx
"""
import copy, random, sys
from pathlib import Path
import pandas as pd
import docx
from docx.oxml.ns import qn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from revisao_track import Revisor

RAIZ = AQUI.parents[1]
RES = AQUI.parent / "resultados"
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 2 else RAIZ / "TCC_PEDRO_MINELLA_revisao_com_alteracoes.docx"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else ENTRADA
PAGINA = sys.argv[3] if len(sys.argv) > 3 else None
W14_PARAID = "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"

# ------------------------------------------------------------------ conferências
fr = pd.read_excel(RES / "onda2_3_forma_reduzida_ar_final.xlsx")
s = fr[fr.instrumento == "surpresa_externa"]; n = fr[fr.instrumento == "news_svar"]
assert (round(s.f_primeiro_estagio.min(), 1), round(s.f_primeiro_estagio.max(), 1)) == (2.9, 6.8)
ig = s[s.variavel == "ipca_geral"]
assert (round(ig.p_forma_reduzida.min(), 3), round(ig.p_forma_reduzida.max(), 3)) == (0.302, 0.979)
assert (s.ar_forma == "limitado").all() and not s.ar_exclui_zero.any()
assert (round(s.largura_ar_sobre_2sls.min(), 1), round(s.largura_ar_sobre_2sls.max(), 1)) == (1.3, 7.5)
assert (round(n.f_primeiro_estagio.min()), round(n.f_primeiro_estagio.max())) == (114, 251)
ng = n[n.variavel == "ipca_geral"].set_index("h").forma_reduzida
assert (round(ng[1], 3), round(ng[3], 3)) == (0.082, 0.133)
pe = pd.read_excel(RES / "onda4_4_paridade_iv.xlsx", sheet_name="primeiro_estagio")
pe = pe[(pe.h == 0) & (pe.regime == "PPI")].set_index("com_brent").F
assert (round(pe[False], 2), round(pe[True], 2)) == (13.65, 2.55)

TEXTO = [
    "A seção 3.4 trata o choque como endógeno porque nenhum instrumento externo testado se mostrou forte o "
    "bastante. Esta subseção resume a tentativa; as estatísticas completas estão nos Apêndices C, G e K.1. Um "
    "instrumento válido teria de mover o preço do petróleo, ou o preço doméstico do combustível, sem afetar a "
    "inflação brasileira por outra via, e sua força é medida pela estatística F do primeiro estágio, cujo limiar "
    "convencional é 10 (STAIGER; STOCK, 1997).",

    "O candidato mais próximo da literatura é a surpresa de oferta de Känzig (2021), a variação dos preços "
    "futuros do petróleo em torno de anúncios da OPEP. Na amostra, ela é diferente de zero em apenas 90 dos 270 "
    "meses, e o primeiro estágio tem F entre 2,9 e 6,8. A forma reduzida do IPCA Geral sobre a surpresa não é "
    "significativa em horizonte algum (p entre 0,302 e 0,979). O conjunto de Anderson-Rubin, robusto a "
    "instrumento fraco (ANDREWS; STOCK; SUN, 2019), é sempre limitado, tem de 1,3 a 7,5 vezes a largura do "
    "intervalo de dois estágios e nunca exclui o zero. A outra série do arquivo de Känzig, o choque estrutural "
    "que o autor recupera em modelo próprio, tem F entre 114 e 251 e produz respostas significativas do IPCA "
    "Geral, de 0,082 em um mês e 0,133 em três. Ela herda, porém, as hipóteses do modelo que a gerou, que inclui "
    "o próprio preço do petróleo, e por isso não serve como instrumento externo.",

    "O segundo grupo de candidatos instrumenta o preço doméstico do combustível com a margem de refino "
    "internacional, a cotação da gasolina e do diesel na Costa do Golfo dos Estados Unidos menos o Brent, "
    "ortogonalizada em relação ao choque. A lógica é que interrupções de refino, em geral climáticas, movem a "
    "margem sem mover o óleo cru. Com os regimes separados pelo anúncio de outubro de 2016, a margem da gasolina "
    "tem F de 5,45 depois e de 1,48 antes, contraste compatível com a hipótese do trabalho, e sua forma reduzida "
    "move o IPCA Transportes no regime posterior (0,357 no impacto e 0,620 em um mês) e, em um mês, o IPCA "
    "Geral (0,111, significativo a 10%). A margem do diesel não tem primeiro estágio em regime algum (F de 0,25 "
    "e 0,91). Nenhuma das duas alcança o limiar de 10.",

    "O único candidato acima do limiar é a cotação cheia da gasolina no Golfo, sem descontar o Brent, com F de "
    "13,65 para a gasolina ao consumidor no período da paridade, de março de 2017 a maio de 2023. Ela carrega, "
    "porém, o próprio preço do petróleo, que alcança a inflação brasileira por outros canais além do "
    "combustível, e por isso não satisfaz a restrição de exclusão; com o Brent incluído como controle, a "
    "estatística cai para 2,55.",

    "A dificuldade não é peculiar a este trabalho. Cardoso e Severnini (2023), com um painel de 555 municípios "
    "entre 2005 e 2016, obtêm estatística F próxima de 18 para o preço da gasolina na amostra completa e abaixo "
    "de 10 nos subperíodos de 2005 a 2009 e de 2010 a 2016. Com instrumentos fracos, o estimador de dois "
    "estágios fica viesado na direção do de mínimos quadrados e sua inferência convencional deixa de ser "
    "confiável. Por isso o trabalho mantém a estimação por mínimos quadrados, trata o choque como endógeno e lê "
    "o contraste entre regimes sob a hipótese de vieses comparáveis da seção 3.4. O primeiro estágio da margem "
    "da gasolina, mais forte depois da paridade, é coerente com essa leitura, mas não tem potência para "
    "sustentá-la sozinho.",
]
TITULO = "4.7.6 Identificação por variável instrumental"
assert not any(c in "".join(TEXTO) for c in "—–")

# ------------------------------------------------------------------ documento
doc = docx.Document(ENTRADA)
R = Revisor(doc)
corpo = doc.element.body
P = doc.paragraphs
usados = {e.get(W14_PARAID) for e in corpo.iter() if e.get(W14_PARAID)}
rnd = random.Random(20261015)


def paraid():
    while True:
        v = "%08X" % rnd.randrange(1, 0x7FFFFFFF)
        if v not in usados:
            usados.add(v)
            return v


def novo_paragrafo(modelo, texto):
    p = corpo.makeelement(qn("w:p"), {})
    p.set(W14_PARAID, paraid())
    ppr = copy.deepcopy(modelo.find(qn("w:pPr")))
    for velho in ppr.findall(qn("w:rPr")):
        ppr.remove(velho)
    rpr = ppr.makeelement(qn("w:rPr"), {}); rpr.append(R._marca("w:ins")); ppr.append(rpr)
    p.append(ppr)
    ins = R._marca("w:ins"); ins.append(R._run_novo(modelo.find(".//" + qn("w:r")), texto)); p.append(ins)
    return p


i475 = [i for i, p in enumerate(P) if p.style.name == "Heading 3" and p.text.startswith("4.7.5 ")]
i48 = [i for i, p in enumerate(P) if p.style.name == "Heading 2" and p.text.startswith("4.8 ")]
assert len(i475) == 1 and len(i48) == 1
tit_modelo, corpo_modelo = P[i475[0]]._p, P[i475[0] + 1]._p
ancora = P[i48[0] - 1]._p
assert P[i48[0] - 1].text.startswith("Nos subgrupos sem combustíveis")
for el in [novo_paragrafo(tit_modelo, TITULO)] + [novo_paragrafo(corpo_modelo, t) for t in TEXTO]:
    ancora.addnext(el); ancora = el

# sumário (texto digitado): linha nova depois da 4.7.5
toc = [p for p in P if p.style.name.lower().startswith("toc") and p.text.startswith("4.7.5 ")]
assert len(toc) == 1
linha = novo_paragrafo(toc[0]._p, TITULO)
if PAGINA:
    ins = linha.find(qn("w:ins"))
    r_tab = R._run_novo(toc[0]._p.find(".//" + qn("w:r")), ""); r_tab.remove(r_tab.find(qn("w:t")))
    r_tab.append(r_tab.makeelement(qn("w:tab"), {}))
    ins.append(r_tab); ins.append(R._run_novo(toc[0]._p.find(".//" + qn("w:r")), PAGINA))
toc[0]._p.addnext(linha)

# referências cruzadas
p = R.paragrafo("As verificações abaixo tratam de placebo")
R.substituir(p, "corte institucional e contribuição anual.",
             "corte institucional, contribuição anual e identificação por variável instrumental.")
p = R.paragrafo("Uma limitação da medida adotada é que o modelo não separa choques")
R.substituir(p, "nos Apêndices G e K.1,", "na seção 4.7.6 e nos Apêndices G e K.1,")
p = R.paragrafo("Convém explicitar o que esta especificação identifica")
R.substituir(p, "como reportam os Apêndices G e K.1.", "como mostram a seção 4.7.6 e os Apêndices G e K.1.")

# correções numéricas nos apêndices
p = R.paragrafo("O Apêndice K.1 reporta a projeção sobre o choque de notícia de oferta")
R.substituir(p, "o conjunto de Anderson-Rubin é aproximadamente 80% mais amplo nos casos reportados e nunca exclui "
                "o zero.",
             "o conjunto de Anderson-Rubin é sempre limitado, tem de 1,3 a 7,5 vezes a largura do intervalo de dois "
             "estágios (mediana de 1,6) e nunca exclui o zero.")
p = R.paragrafo("A outra série do arquivo, o choque estrutural")
R.substituir(p, "entre 114 e 276", "entre 114 e 251")
R.substituir(p, "os conjuntos de Anderson-Rubin são 80% mais amplos e não excluem o zero.",
             "os conjuntos de Anderson-Rubin têm de 1,3 a 7,5 vezes a largura do intervalo de dois estágios e não "
             "excluem o zero.")

doc.save(SAIDA)
print("salvo", SAIDA)
