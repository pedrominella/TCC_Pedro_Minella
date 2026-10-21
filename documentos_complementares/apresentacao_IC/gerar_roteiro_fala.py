# -*- coding: utf-8 -*-
"""
Roteiro de fala da apresentação IC (18 slides, cerca de 19 min 45 s) e preparação para perguntas.
Gera Roteiro_de_fala_IC.docx e roteiro_notas.json (texto das anotações do apresentador no Canva).
Todos os números são os dos slides e do TCC (TCC_PEDRO_MINELLA.docx, versão de 08/10/2026).
20/10/2026: "primeiro efeito" e "segundo efeito" no lugar de "primeira volta" e "segunda volta" (slide 11); slide 15 (robustez) alinhado à seção 4.8 e à endogeneidade assumida.
"""
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor

AQUI = Path(__file__).resolve().parent

SLIDES = [
    ("Capa", 30,
     "Boa tarde. Meu nome é Pedro Minella, sou aluno de Ciências Econômicas do Ibmec Brasília, e esta é a minha "
     "pesquisa de iniciação científica, orientada pelo professor Silvio Michael de Azevedo Costa. A pergunta do "
     "trabalho é simples: quanto de uma alta no preço internacional do petróleo chega à inflação brasileira, e em "
     "quanto tempo. A resposta curta é que depende da regra de preços dos combustíveis, e é isso que vou mostrar."),
    ("01 Introdução", 75,
     "Por que essa pergunta importa? Primeiro, porque os conflitos no Oriente Médio recolocaram o petróleo no centro "
     "do debate, e o Banco Central precisa saber de antemão quanto de um choque vai virar inflação. Segundo, porque o "
     "choque é comum: em 51 dos 272 meses da amostra, a variação do Brent igualou ou superou um desvio-padrão, cerca "
     "de duas vezes por ano. Terceiro, porque no Brasil a regra mudou: a Petrobras vendeu abaixo da referência de "
     "2011 a 2014 e adotou a paridade de importação em 2016, por decisão administrativa. E a literatura deixa duas "
     "lacunas: trata a regulação de preços como fixa em cada país e raramente mede quanto do efeito passa, de fato, "
     "pelos combustíveis. São essas duas lacunas que o trabalho ocupa."),
    ("02 A pergunta", 45,
     "A pergunta tem duas dimensões: magnitude e velocidade. O gráfico mostra a gasolina na refinaria, em vermelho, "
     "e ao consumidor, em laranja, contra a gasolina do Golfo dos Estados Unidos e o Brent, todos em reais e com base "
     "100 em 2003. Até 2016 a linha vermelha anda em degraus, descolada da referência. Depois, passa a acompanhá-la "
     "de perto."),
    ("03 Por que o Brasil é um caso especial", 75,
     "Esta é a cronologia. De 2011 a 2014, contenção de preços, com perda de receita estimada em cerca de R$ 98 "
     "bilhões para a Petrobras. Em 14 de outubro de 2016, o anúncio da paridade de importação. Os dados, como vou "
     "mostrar, datam a mudança em março de 2017, cinco meses depois. Entre 2018 e 2022 houve intervenções pontuais, e "
     "em maio de 2023 a paridade terminou. O gráfico de baixo é a variação mensal do preço na refinaria: antes, o "
     "preço ficou parado em 16,6% dos meses, com variação mediana de 0,17%; depois, nenhum mês parado, e a variação "
     "mediana sobe para 4,27%. É o retrato de um preço administrado que vira um preço indexado."),
    ("04 O que a literatura já sabe", 75,
     "O que a literatura já sabe. Primeiro, quanto chega: em 72 economias, Choi e coautores estimam que uma alta de "
     "10% no petróleo eleva a inflação em cerca de 0,4 ponto no ano, e que esse repasse caiu com a melhora da política "
     "monetária. Segundo, nem todo choque é igual: Kilian mostra que choques de oferta, de demanda global e de demanda "
     "precaucional têm efeitos diferentes. Terceiro, a regra importa: em até 125 economias, Akcura mostra que países "
     "com preço regulado repassam muito menos. Mas esses trabalhos tratam a regulação como fixa em cada país. O Brasil "
     "permite ver a mesma economia antes e depois de uma mudança de regra."),
    ("05 Dados e método", 75,
     "Os dados são mensais, de janeiro de 2003 a setembro de 2025, 273 meses. O choque é a variação do Brent em "
     "dólares, padronizada: um choque típico é uma alta de 10,7%. Estimo seis séries, da gasolina na refinaria ao "
     "IPCA cheio. O método tem quatro passos: medir o choque; localizar a mudança com um teste de quebra, cuja data é "
     "estimada pelos próprios dados; comparar os regimes com projeções locais; e rastrear o efeito da refinaria até o "
     "IPCA, separando o canal dos combustíveis."),
    ("06 O modelo", 90,
     "Esta é a equação. À esquerda, quanto o nível de preços subiu h meses depois do choque. O coeficiente em verde é "
     "o repasse antes da mudança; o em vermelho, o repasse depois. O delta capta a mudança de nível entre os regimes, "
     "e os controles são três defasagens, câmbio, atividade, Selic, expectativa de inflação, sazonalidade e pandemia. "
     "Estimo uma regressão para cada horizonte, de 0 a 12 meses, nas seis séries. A data de corte é março de 2017, "
     "estimada pelos dados. E a pergunta central é o teste de Wald: o repasse antes é igual ao repasse depois? Se não "
     "for, a regra mudou o repasse."),
    ("07 A data não foi escolhida, foi estimada", 75,
     "Como chegamos a março de 2017? Calculei a estatística de quebra para cada data candidata. O máximo é em março "
     "de 2017, com 58,1, contra um valor crítico de 42,9 a 5%. A faixa cinza mostra as datas compatíveis com os "
     "dados, de abril de 2016 a fevereiro de 2021. Ou seja, a data não é precisa ao mês, mas a mudança é clara. E se "
     "eu impuser a data do anúncio, outubro de 2016, o resultado se repete."),
    ("08 O resultado central", 90,
     "Este é o resultado central. No mês do choque, o repasse ao IPCA era de 0,005 ponto percentual antes da mudança. "
     "Depois, fica entre 0,088 e 0,176, conforme o tratamento da pandemia. Mais importante que o número é o padrão: "
     "antes da paridade, nenhuma das 18 estimativas, que são seis séries em três horizontes, se distingue de zero. "
     "Depois, as 18 são significativas. E há um ponto metodológico: se eu não separar os regimes, o teste conjunto "
     "não rejeita a ausência de efeito no IPCA, com p de 0,194. A média da amostra inteira esconde a mudança."),
    ("09 O impacto no IPCA e no IPCA Transportes", 60,
     "Olhando o IPCA cheio e o grupo Transportes lado a lado. No IPCA cheio, o impacto passa de 0,005 para 0,176, e "
     "o teste de Wald rejeita a igualdade com p de 0,001. No IPCA Transportes, que contém os combustíveis, o efeito "
     "é maior: passa de 0,065 para 0,675, com p menor que 0,001. Nas duas séries, a curva verde fica colada no zero e "
     "a vermelha salta já no primeiro mês."),
    ("10 Quando e por onde o efeito passa", 60,
     "Quando e por onde o efeito passa. Quando: rápido, quase toda a resposta aparece nos dois primeiros meses. Por "
     "onde: pelos combustíveis. Em três meses, o canal dos combustíveis carrega 0,003 ponto antes e 0,249 depois. O "
     "gráfico mostra a gasolina na refinaria e ao consumidor: depois da mudança, a refinaria responde forte já no mês "
     "do choque, e a bomba acompanha. No próximo slide, separo esse efeito em dois caminhos."),
    ("11 Primeiro e segundo efeito", 60,
     "Dentro do canal dos combustíveis, há dois caminhos. O primeiro efeito é aritmético: gasolina, diesel e etanol "
     "estão na cesta do IPCA, então, se sobem, o índice sobe sem que nenhum outro preço mude. O segundo efeito é "
     "econômico: o combustível mais caro encarece frete e insumos, e preços que não são combustível também sobem. "
     "Calculo o primeiro efeito pelos pesos da cesta e estimo o segundo depois de tirar essa parcela do índice. No mês "
     "do choque, o primeiro efeito responde por 63% do total; em três meses, os dois se igualam; em seis meses, o "
     "segundo efeito já é 87%, e em doze, 80%. A ressalva: o segundo efeito não é significativo em três meses, com p "
     "de 0,124, e a significância depende do cenário de pesos. O que é firme é a trajetória: o efeito da cesta é "
     "imediato e se esgota; o de custos é mais lento e persiste."),
    ("12 De que tamanho é isso", 90,
     "De que tamanho é isso? Fiz um exercício contrafactual: apliquei os choques de cada ano às duas regras e "
     "acumulei a contribuição ano a ano, reiniciando em janeiro, como o IPCA anual. Em 2021, a regra de paridade "
     "acrescentou 1,72 ponto, cerca de 18% do IPCA do ano, que foi de 10,06%. Em 2022, 1,15 ponto, cerca de 21% do "
     "IPCA de 5,79%. Em 2020, ano em que o petróleo despencou na pandemia, o sinal se inverte: pela conta, a "
     "paridade tirou 2,31 pontos do IPCA em relação à regra anterior. Mas esse é o número mais incerto: o modelo foi "
     "estimado sem os meses da pandemia e trata altas e quedas como iguais; com o índice de Oxford, seriam 0,63 "
     "ponto. Os intervalos são largos e o exercício é de equilíbrio parcial: mede a contribuição do petróleo, mantido "
     "o resto, e não diz qual teria sido a inflação."),
    ("13 O Brasil no mapa", 60,
     "O Brasil no mapa. Com a base do Banco Mundial que acompanha o trabalho de Akcura, comparei 65 países na alta de "
     "abril de 2020 a junho de 2022, do vale ao pico do petróleo, a mesma janela daquele trabalho. Países com preço "
     "desregulado repassaram, na mediana, 0,798 da variação internacional; os regulados, 0,288, um repasse quase três "
     "vezes menor. O Brasil aparece com 0,868, entre os 25% maiores e em sexto na América Latina. É o mesmo "
     "mecanismo, agora entre países."),
    ("14 O resultado à luz da literatura", 75,
     "Comparando com a literatura, em pontos por alta de 10% no petróleo. Antes da mudança, o Brasil era um caso "
     "extremo: Choi e coautores, com dados de 2000 a 2015, colocam o Brasil em último entre 71 economias. Depois, o "
     "Brasil se aproxima dos Estados Unidos: o IPCA sobe 0,33 ponto no pico, contra 0,4% no índice americano segundo "
     "Känzig. E no canal dos combustíveis, a estimativa de 0,23 em três meses cai dentro da calibração do Banco "
     "Central para 2021, de 0,20 a 0,27. Os métodos diferem, então é uma comparação de ordem de grandeza."),
    ("15 Robustez", 75,
     "Robustez. Primeiro, a endogeneidade: o Brent responde à demanda mundial, que também afeta a inflação "
     "brasileira, então trato o choque como endógeno, e os coeficientes como associações condicionais. Se o viés for "
     "igual nos dois regimes, ele se cancela na diferença; é uma hipótese. O Brasil é tomador de preço, e a proporção "
     "de meses de alta até cai depois, de 59,8% para 54,4%. Segundo, a variável instrumental. Para o choque, a "
     "surpresa de oferta da OPEP, de Känzig, é instrumento fraco, com F entre 2,2 e 6,6. Para os combustíveis, a "
     "margem de refino do diesel é forte depois de 2017, com F entre 19 e 31 do primeiro mês em diante, embora fique "
     "em 6,9 no próprio mês do choque. Com ela, o efeito dos combustíveis sobre o IPCA aparece em um e dois meses, "
     "sem efeito sobre os demais preços em até seis meses. Por fim, outras especificações: com a data do anúncio, "
     "0,155; até o fim da paridade, 0,214; com a pandemia por índice de rigor, 0,088; e controlando a inflação "
     "global, 0,170. Em todas, o contraste se mantém."),
    ("16 Conclusão", 60,
     "Para concluir: o repasse não é um parâmetro constante nas economias; no Brasil, ele é consequência da regra de "
     "formação de preços. Quanto: "
     "no mês do choque, o repasse ao IPCA passou de 0,005 para algo entre 0,088 e 0,176 ponto. Quando: quase todo nos "
     "dois primeiros meses. Por onde: pelos combustíveis. A implicação é que a calibragem do repasse precisa ser "
     "revista sempre que a regra muda, e que médias de amostras longas podem esconder regimes diferentes."),
    ("Obrigado", 15,
     "Obrigado. Fico à disposição para as perguntas."),
]

PERGUNTAS = [
    ("Por que usar o Brent e não o choque de Kilian?",
     "Porque o objeto é a transmissão do preço observado, que é a referência que a regra de precificação usa. A "
     "política da Petrobras toma a cotação internacional como parâmetro, e o Brent é a referência usual no Brasil. "
     "Kilian mede o custo do petróleo importado pelas refinarias dos EUA, adequado ao mercado americano."),
    ("E se o resultado vier de mudança no tipo de choque (oferta ou demanda), e não da regra?",
     "O choque é tratado como endógeno, e a leitura do contraste supõe que a contaminação seja parecida antes e "
     "depois; nesse caso, ela se cancela. Além disso, a parcela de meses sem reajuste na refinaria cai de 16,6% para zero, mudança "
     "de comportamento do preço doméstico que a composição do choque internacional não produz. A surpresa de oferta "
     "de Känzig é instrumento fraco para o choque, com F de 2,2 a 6,6; para o diesel, a margem de refino é forte "
     "depois de 2017 e confirma o efeito dos combustíveis sobre o IPCA (seção 4.8)."),
    ("Por que estimar a data em vez de usar o anúncio de outubro de 2016?",
     "Porque o anúncio marca a decisão, não a implementação, que foi progressiva. Mas impondo outubro de 2016 o "
     "resultado se repete: os 18 coeficientes do regime posterior seguem significativos e nenhum do anterior, e o "
     "IPCA no impacto fica em 0,155, contra 0,176 com a data estimada."),
    ("Fizeram teste de placebo?",
     "Sim. O choque de um mês não pode prever a inflação dos meses anteriores. Estimei a mesma equação com a inflação "
     "de um a três meses antes do choque: 17 dos 18 coeficientes não são significativos a 10%. A exceção, a refinaria "
     "um mês antes, com 0,644, cai para 0,350 e perde significância quando se inclui o choque daquele mês, porque o "
     "choque tem autocorrelação de 0,29."),
    ("E o câmbio?",
     "O câmbio entra como controle, defasado. Não entra no mês do choque porque responde ao próprio choque de "
     "petróleo dentro do mês, e incluí-lo absorveria parte do efeito que se quer medir."),
    ("Como a pandemia foi tratada? Por que dummies e não o índice de Oxford?",
     "Com uma dummy para cada mês de março a dezembro de 2020. No mês do choque, isso equivale a tirar esses meses "
     "da conta, porque neles o petróleo despencou e voltou por causa da pandemia. Testei o índice de rigor de Oxford "
     "no lugar das dummies: o impacto no IPCA cai para 0,088, o mesmo valor de não controlar nada, porque o índice "
     "quase não anda junto com o choque. E, com esses meses dentro, o teste de data passa a apontar fevereiro de "
     "2021, ou seja, encontra a pandemia e não a mudança de regra. Por isso mantenho as dummies e reporto a faixa de "
     "0,088 a 0,176; nas duas versões a diferença entre antes e depois é significativa, com p de 0,001 e 0,016."),
    ("Por que erro-padrão de Newey-West?",
     "Porque a resposta acumulada sobrepõe janelas de meses, o que gera autocorrelação nos resíduos. O erro de "
     "Newey-West corrige isso. Comparado ao erro-padrão robusto simples, as conclusões são as mesmas: 35 e 37 "
     "rejeições em 78 testes."),
    ("Por que três defasagens?",
     "Os critérios de informação indicam uma (bayesiano) ou duas (Akaike). Uso três seguindo o aumento de "
     "defasagem de Montiel Olea e Plagborg-Møller, que garante inferência válida mesmo com regressores persistentes, "
     "como a Selic."),
    ("Por que a amostra começa em 2003?",
     "Para excluir a crise cambial de 2002 e ficar inteiramente no regime de metas já consolidado, de modo que a "
     "mudança documentada não se confunda com transição de regime monetário."),
    ("Por que a comparação entre países usa abril de 2020 a junho de 2022?",
     "É a janela de alta do próprio Akcura, o segundo subperíodo daquele trabalho: começa em abril de 2020, quando o "
     "Brent caiu ao menor nível desde 1999, e termina em junho de 2022, no pico. Na região, os dois maiores repasses "
     "são de Panamá e Belize, que são regulados; por isso a leitura é a da mediana, e não país a país."),
    ("O resultado se mantém com outro método, como um VAR?",
     "Sim, está no Apêndice K.7. Um VAR por regime com as mesmas variáveis, três defasagens e o petróleo ordenado "
     "primeiro chega ao mesmo padrão: o IPCA no impacto vai de 0,000 para 0,094 e, em três meses, de 0,028 para "
     "0,335, com a banda de 90% do regime posterior acima de zero. Por unidade de choque, 0,295 contra 0,036 em três "
     "meses. Uso o VAR para confirmar o padrão, e não o número, porque ele não tem indicadoras de pandemia nem de "
     "mês."),
    ("E as intervenções tributárias de 2017 a 2023?",
     "Com indicadoras para a alta de PIS/Cofins de 2017, a subvenção ao diesel de 2018 e as desonerações de 2022 e "
     "a reoneração de 2023, o impacto no IPCA sobe de 0,176 para 0,185. As intervenções atenuam o repasse, então a estimativa "
     "principal é conservadora."),
    ("Pode haver uma segunda quebra, por exemplo em 2011?",
     "Procurei uma segunda quebra dentro de cada regime e não há: p de 0,107 no período anterior e de 0,615 no "
     "posterior."),
    ("O repasse é igual para altas e quedas do petróleo?",
     "No IPCA, sim: 0,141 para altas e 0,207 para quedas no mês do choque, com p de 0,686 para a simetria. No diesel, "
     "não: sobe 4,118 com uma alta e 0,623 com uma queda, com p de 0,002, mas esse resultado é sensível ao "
     "tratamento da pandemia."),
    ("E se o choque for medido em reais?",
     "O contraste se mantém no impacto e em um mês, com p de 0,006 e 0,004, mas não em três meses, com p de 0,391. "
     "Por isso a conclusão se apoia no impacto e no mês seguinte."),
    ("E se mudar o número de defasagens?",
     "De uma a seis defasagens, o impacto no IPCA fica entre 0,161 e 0,184, e a igualdade entre regimes é rejeitada "
     "a menos de 1% em todas. Com a correção de Teulings e Zubanov, as rejeições sobem de 35 para 47."),
    ("O exercício contrafactual diz que a inflação teria sido menor?",
     "Não. É equilíbrio parcial: mede quanto a contribuição dos choques de petróleo teria sido menor, mantido o "
     "resto. Com repasse menor, a política monetária e as expectativas teriam sido outras."),
]


def main():
    total = sum(t for _, t, _ in SLIDES)
    json.dump([{"slide": s, "segundos": t, "texto": x} for s, t, x in SLIDES],
              open(AQUI / "roteiro_notas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.5)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(12)
    doc.add_heading("Roteiro de fala: apresentação de iniciação científica", level=1)
    doc.add_paragraph(f"18 slides, tempo total previsto de {total // 60} min {total % 60:02d} s, para uma "
                      "apresentação de 20 minutos. O tempo de cada slide está entre parênteses; os números são os "
                      "dos slides e do TCC.")
    acum = 0
    for s, t, x in SLIDES:
        acum += t
        h = doc.add_heading(f"{s}  ({t} s; acumulado {acum // 60}:{acum % 60:02d})", level=2)
        doc.add_paragraph(x)
    doc.add_page_break()
    doc.add_heading("Perguntas prováveis da banca", level=1)
    for q, a in PERGUNTAS:
        p = doc.add_paragraph()
        r = p.add_run(q)
        r.bold = True
        doc.add_paragraph(a)
    arq = AQUI / "Roteiro_de_fala_IC.docx"
    doc.save(arq)
    print("tempo total", total, "s;", arq)


if __name__ == "__main__":
    main()
