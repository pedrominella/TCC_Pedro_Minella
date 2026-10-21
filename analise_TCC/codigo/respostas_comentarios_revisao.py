# -*- coding: utf-8 -*-
"""
Respostas do Claude aos comentários abertos do arquivo de revisão (14/10/2026).

Cada par é (chave, resposta). A chave é um trecho do comentário que abre o fio; com "=" na
frente, exige igualdade com o texto inteiro do comentário. O estado descrito é o do arquivo
salvo pelo autor depois de aceitar todas as alterações (14/10/2026, 19:53).
"""

RESPOSTAS = [
    # ---------------------------------------------------------------- pré-textuais e resumo
    ("Preencher o nome do examinador",
     "Pendente, decisão sua: o segundo examinador ainda está como [NOME DO EXAMINADOR]. Pelo plano de ensino, "
     "a composição da banca é definida pelo coordenador de TCC; preencha quando ela for divulgada."),
    ("Resumo reduzido para até meia página",
     "Ok. Em 14/10 entrou no resumo e no abstract uma frase sobre a endogeneidade do choque e a fraqueza dos "
     "instrumentos. Vale conferir se o resumo continua em meia página."),
    ("O resumo reúne muitos resultados numéricos",
     "Em parte. O resumo já foi encurtado; os números que restam (0,005 contra 0,088 a 0,176; 0,003 contra "
     "0,249) são o resultado do trabalho, e eu os manteria."),
    ("Revisão integral da redação",
     "Comentário geral, sem ação própria. O resumo atual não mistura o teste de nulidade com o de igualdade "
     "entre regimes; a distinção está na seção 4.2."),
    ("Conferir a equivalência entre resumo e abstract",
     "Resolvido. O abstract diz 'Responses concentrate in the first two months' e não menciona mais as "
     "dezoito estimativas."),
    ("A lista de tabelas omite as Tabelas J.1",
     "Procede, pendente. Conferido em 14/10: a lista não inclui as Tabelas J.1, J.2 e J.3. A lista é texto "
     "digitado, não campo do Word, então precisa ser acrescentada à mão (posso fazer)."),
    ("PPI aparece aqui como Preço de Paridade de Importação",
     "A sigla usada pela Petrobras e pela ANP é PPI, Preço de Paridade de Importação; 'paridade de preços "
     "internacionais' é a descrição genérica da ideia. Sugestão: 'Preço de Paridade de Importação (PPI)' para "
     "o preço de referência e 'política de paridade de importação' para a regra da Petrobras."),
    ("O sumário está desatualizado",
     "Procede, pendente. Duas coisas: (1) o sumário é texto digitado, não campo do Word, então Ctrl+A e F9 não "
     "o atualizam; as páginas precisam ser corrigidas à mão, ou o sumário trocado pelo automático (Referências "
     "> Sumário); (2) a seção 4.10 Síntese não existe mais no corpo, saiu no enxugamento. Decida se ela volta."),
    # ---------------------------------------------------------------- introdução
    ("Enxugamento para atender ao limite de 40 páginas",
     "Em parte. O corte passou do necessário: em relação ao mestre, a Introdução perdeu 70%, os Resultados 59% "
     "e a Conclusão 62%, e o corpo ficou em cerca de 32 páginas. O plano de ensino pede de 30 a 40, então há "
     "espaço para recuperar texto, a começar pela Introdução."),
    ("Acho que vc fez a introdução para caber",
     "Concordo. A introdução caiu de cerca de 6.000 para 1.800 caracteres e hoje apresenta o resultado antes da "
     "pergunta. Com o corpo em cerca de 32 de 40 páginas, dá para recuperar motivação, pergunta, contribuição e "
     "roteiro a partir do texto do mestre; posso fazer com controle de alterações."),
    ("A abertura menciona conflitos",
     "Resolvido. A abertura atual não menciona mais conflitos nem tráfego marítimo."),
    ("Um erro-padrão pequeno não prova",
     "Exagerado para um TCC. O texto não afirma ausência de efeito; diz que os coeficientes anteriores não "
     "diferem de zero. Margem de equivalência não é necessária."),
    ("Deixar mais claro o paragrafo",
     "Sugestão de redação: 'As estimativas indicam aumento do repasse ao IPCA Geral por volta de 2017. Antes, "
     "um choque de um desvio-padrão no Brent, alta de 10,7%, praticamente não move o índice no mês do choque "
     "(0,005 ponto, sem significância); depois, a resposta é de 0,176 ponto com as indicadoras de pandemia e de "
     "0,088 com o índice de rigor de Oxford. Quando se permite dinâmica própria a cada regime, os combustíveis "
     "mostram alguma resposta antes de 2017, mas o IPCA Geral continua sem resposta nesse período.'"),
    ("A dicotomia entre regra de mercado estável",
     "Seu Wald e os coeficientes antes e depois de 2017 provam a mudança no Brasil, e isso sustenta a narrativa. "
     "O ponto da outra IA é a generalização para outras economias ('onde a regra é de mercado, o repasse é "
     "permanente'), que o trabalho não testa. Sugestão para voltar ao seu parágrafo de forma suave: trocar "
     "'é característica razoavelmente permanente' por 'tende a ser característica mais estável'."),
    # ---------------------------------------------------------------- referencial
    ("A identificação por surpresas em anúncios da OPEP",
     "Em parte. O K.1 já diz qual série de Känzig entra em cada exercício. A ressalva cabe em meia frase na "
     "seção 2.2: a janela estreita 'reduz, sem eliminar', a contaminação por condições macroeconômicas."),
    ("A seção K.4 descreve interações do choque com o sinal",
     "Em parte. O K.4 separa o choque em partes positiva e negativa, que é a decomposição de Shin, Yu e "
     "Greenwood-Nimmo (2014), mas sem as somas parciais em nível do NARDL. Sugestão: 'segue a decomposição do "
     "choque em altas e quedas de Shin, Yu e Greenwood-Nimmo (2014)'."),
    ("Definir com precisão a grandeza entre 39,8% e 54,4%",
     "Resolvido em 14/10: o texto diz agora que 39,8% a 54,4% é o grau de repasse do petróleo em reais ao preço "
     "da gasolina ao consumidor, supondo que a refinaria acompanha o petróleo, conforme o boxe do BCB de março "
     "de 2022."),
    ("Foi utilizado explicitamente outubro de 2016",
     "Resolvido. O texto atual diz só que a correlação móvel dispensa o procedimento econométrico de datação, "
     "sem afirmar que nenhuma data foi escolhida."),
    ("Talvez colocar esse paragrafo depois da figura um",
     "Concordo: o parágrafo de Correa e Minella fica melhor depois da Figura 1 e da nota dela. Posso mover com "
     "controle de alterações."),
    ("Projeções locais continuam dependentes da especificação",
     "Seu ponto está certo: o texto diz que a estimação direta 'reduz a dependência de uma única especificação "
     "dinâmica', o que é correto e não afirma imunidade. Não precisa mudar."),
    ("=N entendi",
     "A frase quer dizer: num VAR, para deixar o repasse mudar entre períodos, é preciso estimar um sistema "
     "inteiro para cada período ou um modelo com regimes, o que multiplica os parâmetros; na projeção local "
     "basta um termo de interação na mesma regressão. Sugestão: 'Num modelo de vetores autorregressivos, a "
     "mesma flexibilidade exige estimar um sistema por regime, com muito mais parâmetros.'"),
    ("O resultado de Montiel Olea e Plagborg-Møller (2021) vale",
     "Em parte. Conferido no artigo: a validade uniforme vale para projeções locais com defasagens adicionais "
     "e horizontes curtos em relação à amostra, e os autores mostram que erro-padrão robusto à "
     "heterocedasticidade basta. O HAC usado aqui continua válido, só mais conservador. Sugestão: dizer isso "
     "em uma frase, no lugar de 'A persistência da Selic exige conferir as condições...'."),
    ("Somar estatísticas Wald univariadas",
     "Seu argumento (várias quebras segmentam a amostra e custam observações) é válido, mas é outro: Bai, "
     "Lumsdaine e Stock mostram que estimar uma data comum a várias séries estreita o intervalo da data. E a "
     "estatística do trabalho é a soma dos Wald por série, não o Wald multivariado deles. Sugestão: citar o "
     "artigo para a ideia de quebra comum e acrescentar a sua razão de eficiência."),
    ("Davidson e Flachaire (2008) analisam o wild bootstrap",
     "Procede. Davidson e Flachaire tratam do caso heterocedástico; aplicar os pesos de Rademacher por blocos "
     "de 12 meses é adaptação deste trabalho para a dependência temporal. Sugestão: dizer isso na seção 3.5."),
    # ---------------------------------------------------------------- metodologia
    ("Conferir a classificação de combustíveis como monitorados",
     "Procede, pendente de conferência. Em Figueiredo e Ferreira (BCB, 2002), o álcool aparece entre os preços "
     "monitorados, embora 'virtualmente livre'. Não conferi se a classificação mudou nas revisões posteriores "
     "da cesta; se o etanol tiver saído dos monitorados, ele entra nos comercializáveis e a 'exclusão por "
     "construção' precisa de ressalva."),
    ("Antes de 2017, apenas 16,6% dos meses",
     "Em parte. Antes de 2017 a refinaria fica parada em 16,6% dos meses (variação absoluta abaixo de 0,05%, "
     "critério do código). Isso mostra degraus, mas não 'longos trechos'. Sugestão: 'avança por degraus, com "
     "16,6% dos meses sem variação'."),
    ("A equação mistura 100 vezes a diferença logarítmica",
     "Resolvido. σ = 10,19 está em pontos logarítmicos e um choque unitário equivale a alta de 10,7%. Em 14/10 "
     "as elasticidades da Tabela 8 passaram à mesma convenção da seção 4.5.2 (divisão por 10,19)."),
    ("Esclarecer se os 51 meses",
     "Resolvido em 14/10: o texto diz agora que são 51 meses em valor absoluto, 25 de alta e 26 de queda."),
    ("A igualdade do viés entre regimes é uma hipótese",
     "Resolvido em 14/10: as seções 3.2 e 3.4 dizem agora que o choque é tratado como endógeno e que a "
     "igualdade dos vieses é hipótese, e não consequência da medida de choque."),
    ("A variável 100[ln P(t+h)-ln P(t-1)] está em pontos",
     "Procede, pendente. Os resultados estão em pontos logarítmicos e o texto fala em 'ponto percentual'. Nas "
     "respostas do IPCA a diferença é desprezível; nos combustíveis, não. Sugestão: uma frase na 3.3 dizendo "
     "que ponto logarítmico se aproxima de ponto percentual para variações pequenas."),
    ("=Ajustar essa equação",
     "Feito em 14/10: a equação está agora no formato nativo do Word (Cambria Math), sem notação de LaTeX."),
    ("=Melhorar a equação",
     "Feito em 14/10: a equação está agora no formato nativo do Word, com índices pré e pós e a indicadora "
     "1{t < τ} montados como equação."),
    ("=ajustar",
     "Feito em 14/10: convertido para o formato de equação do Word."),
    ("=Ajustar os betas e colocar em formato equação",
     "Feito em 14/10: os betas do texto e das equações estão agora em formato de equação do Word."),
    ("=Ajustar equações",
     "Feito em 14/10: as equações (10) a (12) estão agora no formato nativo do Word."),
    ("A transformação 100 ln(1+r/100) das taxas",
     "Em parte. As duas taxas entram em nível transformado, como diz o texto, e não precisam ser equivalentes "
     "em ordem de integração. Falta informar o horizonte da expectativa Focus, que a base não registra; vale "
     "conferir na planilha de origem e pôr no Quadro 1."),
    ("Demonstrar que a implementação satisfaz as condições",
     "Ver a resposta ao comentário sobre Montiel Olea e Plagborg-Møller na seção 2.4: o HAC continua válido e "
     "basta dizer que a validade uniforme vale para horizontes curtos em relação à amostra."),
    ("Como a dependente é uma variação acumulada do nível",
     "Procede, ajuste de redação. Sugestão: 'entra também isoladamente, para que uma mudança na média da "
     "variação acumulada entre os períodos não seja atribuída ao choque'."),
    ("Se eu n me engano eu pedi para o claude",
     "Na decisão de 14/10, as indicadoras de janela ficaram no modelo principal, com o índice de Oxford e a "
     "versão sem controle como robustez. Nenhum mês de 2020 sai da amostra: as dez indicadoras neutralizam "
     "março a dezembro de 2020, o que no impacto equivale numericamente a retirá-los. A seção 4.7.2 e a "
     "Figura 12 já trazem a alternativa (0,088 com Oxford). Para outra configuração, preciso reestimar."),
    ("Explicitar a fórmula das indicadoras de janela",
     "Procede, pendente. A definição no código é 1{t ≤ s ≤ t+h}, para cada mês s de março a dezembro de 2020. "
     "Sugestão: escrever isso em uma linha na seção 3.4."),
    ("O trabalho reporta magnitudes de cada regime",
     "Resolvido em 14/10: saiu a frase de que o nível não é objeto de interesse; a 3.4 diz agora que os "
     "coeficientes são associações condicionais e que a leitura causal fica restrita ao contraste, sob "
     "hipótese."),
    ("Definir os termos de viés como limites probabilísticos",
     "Resolvido em 14/10: o asterisco denota agora o limite em probabilidade, a analogia com diferenças em "
     "diferenças saiu e o texto explica o que a igualdade dos vieses exige."),
    ("A condição de tomador de preços reduz",
     "Procede. Com a endogeneidade assumida na 3.4, a condição de tomador de preço fica só como redução da "
     "causalidade reversa. O controle global do K.5 é evidência parcial."),
    ("Esclarecer por que a largura de banda é h+1",
     "Em parte. h+1 é a escolha usual quando a dependente sobrepõe h+1 variações. Não testei outras larguras; "
     "se quiser, rodo uma sensibilidade para registrar em nota."),
    ("O uso de bootstrap não implica inexistência",
     "Em parte. O texto não diz que não há distribuição assintótica; diz onde se usa reamostragem. Cabe uma "
     "nota curta sobre réplicas e covariância entre horizontes."),
    ("A soma dos Wald por série é uma estatística agregada",
     "As quatro séries são gasolina ao consumidor, diesel, IPCA Transportes e IPCA Geral; o conjunto de cinco "
     "acrescenta a refinaria (conferido no código). O sorteio do bootstrap é indexado por data, o mesmo para "
     "todas as séries, o que preserva a dependência entre elas. Vale dizer as duas coisas na 3.6."),
    ("O conjunto de datas com rejeição do teste de inexistência",
     "Procede. O texto já diz que a faixa não é intervalo de confiança da data, mas o cabeçalho da Tabela 6 "
     "ainda diz 'Faixa compatível'. Sugestão: 'Datas que rejeitam a 5%'."),
    ("A diferença entre coeficientes antes e depois de condicionar",
     "Procede; a seção 3.7 já trata a leitura como contábil. Nada a mudar."),
    ("Verificar a identidade algébrica",
     "Conferido: a identidade é verificada no código com os mesmos controles e a mesma amostra, com desvio "
     "zero. O erro-padrão é condicional aos regressores, inclusive aos preços dos combustíveis."),
    ("A redação original contradiz a seção 4.5.1",
     "Resolvido. A redação atual diz que os exercícios compartilham choque, controles e amostra."),
    ("O placebo é uma verificação de ausência",
     "Em parte. O texto já apresenta o placebo como verificação de associação antecipada; com a endogeneidade "
     "assumida, ele não precisa servir de prova de exogeneidade. Pode ficar."),
    ("Para permitir replicação, informar as versões",
     "Em parte. As sementes estão nos scripts; faltam as versões das bibliotecas e o número de réplicas por "
     "exercício. Um apêndice curto ou um repositório resolve."),
    # ---------------------------------------------------------------- resultados
    ("A Tabela 1 apresenta coeficientes do IPCA Geral significativos",
     "Procede. No IPCA Geral, o teste conjunto de nulidade de 0 a 6 meses tem estatística 12,84, p de 0,194 "
     "por bootstrap e 0,076 pelo qui-quadrado, enquanto os coeficientes em 0, 1 e 3 meses são significativos "
     "isoladamente. Sugestão: informar os dois p-valores e dizer que o bootstrap é mais conservador."),
    ("A potência depende da hipótese",
     "Resolvido na redação atual da seção 4.2."),
    ("Os Wald desta tabela são pontuais",
     "Procede. Sugestão para a nota da Tabela 2: 'Wald: p-valor pontual da igualdade entre regimes, "
     "condicional ao corte de março de 2017; o p-valor ajustado pela seleção da data está na seção 4.6.'"),
    ("A frase dezesseis dos dezoito coeficientes",
     "Resolvido. O texto diz 'nos horizontes de impacto, um e três meses', e a Tabela 2 usa '<0,001'."),
    ("Na Tabela 2, o IPCA Geral posterior permanece significativo",
     "Resolvido. O texto diz que 23 das 30 respostas posteriores são significativas, sem perda geral de "
     "significância."),
    ("A frase original inverte o objeto da rejeição",
     "Resolvido na redação. Se quiser citar as estatísticas dos testes conjuntos de igualdade (0 a 6 meses): "
     "IPCA Geral 27,88, Transportes 41,52, gasolina 55,34, diesel 65,82, refinaria 95,12 e etanol 19,10, com "
     "p bootstrap de no máximo 0,031."),
    ("A afirmação de que decomposição da variância não se aplica",
     "Em parte. O texto já diz que o ganho de ajuste não equivale à decomposição de variância. Falta dizer como "
     "foi calculado e em qual amostra; não localizei a rotina e não confirmei os valores 4,7, 9,4, 10,6 e 16,5."),
    ("O contraste deve ser associado à especificação com dinâmica comum",
     "Resolvido. A frase diz agora 'na especificação de referência'."),
    ("A Tabela 3 contém etanol em h=0",
     "Resolvido. 'Suas doze contribuições são significativas' vale a 10%; a do etanol no mês 0 tem p de 0,072."),
    ("=Verificar esse valor",
     "Conferido: 0,101 é a resposta acumulada do IPCA Geral em três meses no regime anterior, a mesma da "
     "Tabela 2 (erro-padrão 0,084, p = 0,232)."),
    ("Conferir  esses resultados",
     "Conferido: 0,314 em três meses e 0,269 em doze no regime posterior, os mesmos da Tabela 2. Em 14/10 "
     "corrigi a frase seguinte, que sugeria reversão no próprio IPCA Geral."),
    ("Conferir o acumulado de 0,269",
     "Resolvido em 14/10: 0,269 está certo e significativo a 5% (p = 0,019). A frase diz agora que o índice "
     "cheio não tem contribuição mensal negativa significativa e que a reversão aparece nas outras cinco séries."),
    ("A precisão do coeficiente condicionado não demonstra",
     "Em parte. O erro-padrão de 0,053 é HAC e condicional aos regressores, inclusive aos preços dos "
     "combustíveis, como é usual nesse tipo de decomposição; não incorpora a incerteza da resposta dos "
     "combustíveis ao choque. Sugestão: dizer isso em meia frase na seção 3.7."),
    ("Eu queria no grafico a direita o intervalo do efeito total",
     "Pendente. Na Figura 7 atual, o painel da direita mostra o efeito via combustíveis em cada regime, com as "
     "faixas dele. Posso refazer o painel da direita com o efeito total e só a faixa dele, ou pôr a faixa do "
     "efeito total no painel da esquerda; diga qual prefere."),
    ("O texto informa erro-padrão de 0,067",
     "Resolvido: a frase com 0,067 saiu. Para registro, o erro-padrão do total anterior em h = 3 é 0,064 na "
     "decomposição e 0,084 na Tabela 2; a decomposição calcula a variância com os resíduos da equação que "
     "inclui os combustíveis."),
    ("A decomposição não permite excluir canais de câmbio",
     "Resolvido. A redação atual diz que a leitura do mediador é contábil, e não causal."),
    ("A trajetória monotônica de parcelas pontuais",
     "Procede, sem mudança necessária: o texto já diz que a segunda volta não é significativa em três meses e "
     "depende dos pesos."),
    ("Há três grandezas diferentes no texto",
     "Resolvido em 14/10: 0,164 vinha de uma rodada anterior e a Tabela H.2 foi atualizada para 0,163. O 0,161 "
     "é outra grandeza: a segunda volta da construção aritmética (Tabela 5)."),
    ("A seção 2.3 refere-se a coeficiente de refinaria",
     "Resolvido em 14/10, junto com o comentário da seção 2.3: a calibração é do preço ao consumidor, e a "
     "comparação com 0,390 está na mesma etapa da cadeia."),
    ("A faixa compatível deriva de superar o crítico",
     "Procede. Ver o comentário na seção 3.6: sugiro trocar o cabeçalho por 'Datas que rejeitam a 5%'."),
    ("299 réplicas implicam resolução mínima",
     "Em parte. O p-valor é (1 + réplicas que superam a estatística)/(299 + 1), e '<0,004' indica que nenhuma "
     "superou. Mais réplicas melhorariam a precisão perto dos limiares, como no IPCA Geral (0,104 e 0,084)."),
    ("A faixa da Tabela 6 é definida por superar o crítico",
     "Procede. O texto da 4.6 já diz que a faixa não é intervalo de confiança; falta só trocar o cabeçalho da "
     "Tabela 6."),
    ("O IPCA Geral, desfecho central, não aparece",
     "Resolvido em 14/10: entrou o p ajustado do IPCA Geral, 0,104 com quatro séries e 0,084 com cinco, sem "
     "significância a 5%."),
    ("Comparabilidade da unidade de medida não equivale",
     "Resolvido em 14/10 com o novo texto da 3.4 sobre endogeneidade e igualdade dos vieses."),
    ("K.1 é explicitamente inconclusivo e K.6 altera",
     "Resolvido em parte. A redação atual não diz mais que todas as verificações confirmam; falta incluir o VAR "
     "(K.7) na lista do que o Apêndice K examina."),
    ("Indicar se a correção do placebo por choque",
     "Procede. Se a versão com o choque do mês da dependente foi feita depois de ver o placebo, convém dizer "
     "isso e apresentá-la como sensibilidade."),
    ("O grupo posterior inclui 74 meses",
     "O número está certo: 74 meses é a amostra que termina em abril de 2023 e dá 0,214 (reestimado em 14/10). "
     "A ressalva de que o regime posterior inclui 29 meses depois da paridade procede, e o texto já a faz."),
    ("O texto afirma que a equivalência com exclusão",
     "Resolvido. O texto diz que a equivalência vale no impacto e não se estende automaticamente aos horizontes "
     "com janelas sobrepostas."),
    ("A substituição por Stringency altera",
     "Em parte. A contagem confere: 43 parâmetros com as dez indicadoras e 34 com o índice (43 − 10 + 1). A "
     "seção 4.7.2 já trata os dois controles como escolhas diferentes, com a faixa de 0,088 a 0,176."),
    ("A escolha do controle muda a quebra",
     "Procede e é ponto provável da banca. A escolha das indicadoras tem justificativa anterior ao teste "
     "(Salkever, 1976; Lenza e Primiceri, 2020), e o texto mostra que com Oxford a data vai para fevereiro de "
     "2021 e, com cinco séries, para novembro de 2016. A conclusão já diz que magnitude e datação dependem do "
     "controle da pandemia."),
    ("O coeficiente pré de 0,431 significativo",
     "Procede. 0,431 vem da estimação só no período anterior, com dinâmica própria; 0,187 é da Tabela 2, com "
     "dinâmica comum. Sugestão: meia frase na 4.7.3, remetendo ao K.6, onde o mesmo 0,431 aparece com p de "
     "0,050."),
    ("A data institucional é externa ao procedimento de busca",
     "Procede. Com a endogeneidade assumida na 3.4, a frase da 4.7.4 ('reduz esse problema específico de "
     "seleção') está na medida certa."),
    ("A expressão coeficientes que de fato vigoraram",
     "Resolvido. O texto não usa mais essa expressão e a Tabela H.1 fala em 'diferença entre as "
     "parametrizações'. O 2,87 é a soma das diferenças, 1,72 e 1,15, e não de 1,86 e 0,94."),
    ("O reinício anual exclui contribuições",
     "Resolvido na redação atual, que explica o reinício anual e usa variação logarítmica."),
    ("A ausência de assimetria detectável do IPCA Geral",
     "Procede. O contrafactual usa os coeficientes do IPCA Geral, cuja simetria não é rejeitada. Sugestão: "
     "tirar 'além da assimetria indicativa do diesel' da seção 4.7.5."),
    ("Não rejeitar igualdade nos subgrupos",
     "Procede. Não rejeitar não prova ausência de diferença. Posso calcular os intervalos de 90% da diferença "
     "entre regimes e acrescentá-los à Tabela H.2."),
    ("A razão entre variações de preços de dois extremos",
     "Resolvido. A seção 4.8 diz que a comparação situa a magnitude, mas não identifica data nem causa."),
    ("Correção de coerência numérica: percentil 77",
     "Resolvido. 'Decil superior' saiu do texto; 16º de 65 é o quarto superior e o percentil 77 está certo."),
    ("Reescalar por 10/10,7 mistura",
     "Procede só em tese. Os dois fatores dão 0,9346 (10/10,7) e 0,9353 (100·ln 1,1/10,19), diferença de 0,1%, "
     "sem efeito nos números da Tabela 8. Basta dizer na nota qual convenção se usa."),
    ("Esta comparação mistura inflação anual",
     "Ver a resposta na seção 4.9: a diferença entre as convenções é de 0,1% e não altera a tabela."),
    ("As comparações de efeitos no ano do choque",
     "Os números conferem: 0,356 × 10/10,7 = 0,33. Procede pedir que a Tabela 8 diga a unidade de cada "
     "comparação; a nota já avisa que é ordem de grandeza."),
    ("A Tabela 5 apresentada termina em h=6",
     "Procede. Os valores existem: segunda volta de 0,219 em doze meses (p = 0,014), 80% do efeito total. "
     "Posso acrescentar a linha de h = 12 à Tabela 5."),
    ("A conclusão deve distinguir resultados condicionais",
     "Em parte. A conclusão já diz que magnitude e datação dependem do controle da pandemia e que o repasse "
     "anterior dos combustíveis muda com a dinâmica. Falta o p ajustado por seleção do IPCA Geral (0,104); "
     "sugiro uma frase."),
    ("Na Tabela 3, a parcela dos dois primeiros meses varia",
     "Resolvido. A conclusão diz que a transmissão 'concentra-se nos dois primeiros meses', sem 'integralmente'."),
    ("As limitações efetivamente alteram magnitude",
     "Resolvido em 14/10: a conclusão inclui a endogeneidade e a hipótese de vieses comparáveis. Falta o p "
     "ajustado do IPCA Geral (ver comentário anterior)."),
    ("O preço internacional de derivados pode afetar",
     "Resolvido. A agenda pede instrumento que satisfaça relevância e exclusão."),
    ("A extensão a qualquer economia",
     "Resolvido. A frase final não generaliza para outros países."),
    # ---------------------------------------------------------------- referências
    ("Conferir o título, as páginas e o idioma",
     "Conferido no PDF: 'Exchange rate pass-through: estimates from local projections', Monetary Policy Report "
     "de março de 2026, boxe a partir da p. 66, em inglês. Falta só endereço eletrônico e data de acesso, se a "
     "norma da instituição exigir."),
    ("A autoria Cardoso e Severnini foi conferida",
     "Ok. Falta acrescentar o endereço de acesso do working paper."),
    ("O ECB Working Paper 2968 foi publicado",
     "Resolvido em 14/10: o PDF usado é a versão revisada de março de 2026; as citações voltaram para 2026 e a "
     "referência registra a primeira versão, de 2024."),
    ("As referências às bases de dados são gerais",
     "Procede, pendente (ver o comentário no Quadro 1)."),
    # ---------------------------------------------------------------- apêndices
    ("Os conjuntos Anderson-Rubin podem ser ilimitados",
     "Procede, pendente de conferência no arquivo de resultados: falta dizer se os conjuntos são intervalos "
     "finitos e como os 80% foram calculados."),
    ("A ausência de variação em 16,6% dos meses",
     "Em parte. O critério existe no código: mês parado é variação absoluta abaixo de 0,05%; vale citá-lo. "
     "Procede a ressalva de que variação da média da ANP não prova reajuste da Petrobras em todos os meses."),
    ("O texto informa uma escada de uma a doze defasagens",
     "Resolvido em 14/10: o texto diz agora 3, 6 e 12 defasagens, como a figura."),
    ("A data institucional não é selecionada pelo sup-Wald",
     "Em parte. Com a endogeneidade assumida, a terminologia ficou coerente; o Apêndice F diz corretamente que "
     "a data institucional dispensa a correção por seleção."),
    ("A ortogonalização da margem em relação ao Brent",
     "Procede, mas perdeu importância: com a decisão de 14/10 de não usar instrumentos, o Apêndice G só "
     "documenta a tentativa."),
    ("A faixa 2,9 a 6,8 é repetida nas duas colunas",
     "Procede. A faixa refere-se ao conjunto de horizontes e variáveis, não a cada regime; sugiro célula "
     "mesclada ou nota. Registre também que a Tabela G.1 usa o corte de outubro de 2016 (108 e 161 "
     "observações), e não o de março de 2017."),
    ("A Tabela G.2 informa IPCA Geral em h=1",
     "Resolvido. O texto menciona o IPCA Geral em um horizonte."),
    ("A Tabela G.1 tem F anterior de 1,48",
     "Resolvido na redação, que fala em associação mais forte no regime posterior, e não em ausência de "
     "resposta antes."),
    ("Na Tabela H.1, diferença atribuível",
     "Resolvido. O rótulo é 'Diferença entre as parametrizações'. Os intervalos estão no texto da 4.7.5; pô-los "
     "na tabela é opcional."),
    ("A contribuição de 2020 aplica coeficientes",
     "Em parte. A nota da Tabela H.1 remete à ressalva de 2020 na 4.7.5, que traz o intervalo [−5,06; 0,30] e "
     "o resultado com Oxford."),
    ("A não rejeição pode decorrer de efeitos pequenos",
     "Procede. Ver o comentário na seção 4.7.5: posso acrescentar os intervalos da diferença à Tabela H.2."),
    ("Expressar varejo e referência em dólares",
     "Procede, ajuste de redação: 'de sorte que a variação cambial não entra separadamente no cálculo'."),
    ("O texto situa o Brasil entre a mediana",
     "Correto como está; nada a mudar."),
    ("A posição 16/65 sugere aproximadamente",
     "Conferido: 16º de 65 dá percentil 77 pela convenção (65 − 16)/(65 − 1). Sugestão: dizer a convenção na "
     "nota."),
    ("A Tabela J.1 contém duas taxas",
     "Procede, ajuste menor na nota da Tabela J.1."),
    ("Os pesos são aproximados e fixos",
     "Procede. Os pesos são de cenário (central: gasolina 4,6%, etanol 0,7%, diesel 0,15%), fixos na amostra; "
     "falta citar a fonte e o ano da ponderação do IBGE."),
    ("Valores arredondados como 0,000",
     "Procede, ajuste de formatação."),
    ("ADF e KPSS reportados referem-se",
     "Procede, ajuste no título da Tabela J.3."),
    ("A ordem de integração não é o único diagnóstico",
     "Ok, nada a mudar."),
    ("K.1 é inconclusivo, K.2 altera a significância",
     "Resolvido. A introdução do Apêndice K não diz mais que todas as verificações confirmam."),
    ("A não significância da forma reduzida",
     "Resolvido. O K.1 diz que é inconclusivo e que baixa potência é uma explicação possível."),
    ("Um choque estrutural recuperado por SVAR",
     "Procede. Sugestão: trocar 'circularidade' por 'a série herda as hipóteses do modelo que a gerou, que "
     "inclui o próprio preço do petróleo'."),
    ("A conclusão de que a via instrumental exige mudar",
     "Procede. Sugestão: 'a via instrumental pode exigir mudança da unidade de observação'."),
    ("O próprio parágrafo informa magnitudes diferentes",
     "Procede. Sugestão: dizer que a frequência de sinais é parecida, mas a magnitude média dos choques "
     "difere, o que afeta a precisão sem provar estabilidade do processo."),
    ("Não rejeitar simetria não demonstra",
     "Procede. Basta dizer 'não se detecta assimetria no IPCA Geral', sem afirmar que a assimetria não explica "
     "o contraste."),
    ("Esclarecer que coeficiente positivo associado",
     "Procede, ajuste de redação para deixar o sinal claro: coeficiente positivo sobre choque negativo "
     "significa queda do preço."),
    ("O mecanismo de margem de refino",
     "Procede. Apresentar o mecanismo de margem como hipótese."),
    ("O contrafactual de 4.7.5 usa coeficientes do IPCA Geral",
     "Procede. Sugestão para o K.4: 'aplicado a preços de combustíveis, um contrafactual com coeficientes "
     "simétricos poderia distorcer as contribuições de altas e quedas'; e tirar a menção ao diesel da 4.7.5."),
    ("Controles globais defasados não absorvem",
     "Procede. A expressão 'conceitualmente limpo' já não está no texto; o resto da ressalva vale."),
    ("O parágrafo registra que o coeficiente de diesel recua",
     "Procede. A introdução do K.5 ainda fala em estimativa conservadora; como o diesel recua com as "
     "indicadoras tributárias, convém limitar a afirmação às séries em que ela vale."),
    ("p=0,107 no segmento anterior",
     "Procede, ajuste de redação: 'não se detecta segunda quebra (p = 0,107)'."),
    ("Correlações de −0,368 e −0,314",
     "Procede. As correlações descrevem o câmbio; não mostram que o viés ficou igual."),
    ("A dinâmica comum é rejeitada em 56/78",
     "Procede. O corpo cita o K.6, mas não o número. Sugestão para a 4.3: 'a restrição de dinâmica comum é "
     "rejeitada em 56 das 78 equações; liberada, o contraste do IPCA Geral persiste no impacto e em um mês'."),
    ("Conferir a banda do impacto de IPCA Geral",
     "Conferido: banda de 0,061 a 0,110 para 0,094, por percentis 5 e 95 de 400 réplicas, igual ao arquivo de "
     "resultados. Sugestão: dizer 'banda percentil' na nota."),
    ("A reescala pelo desvio-padrão residual",
     "Procede; o texto já trata o VAR como verificação qualitativa de método."),
    ("Contribuições mensais somadas em desenho comum",
     "Resolvido em 14/10: a nota da Tabela 3 explica que o mês 0 do IPCA Geral é 0,127 nela e 0,176 na Figura "
     "5, por desenhos diferentes."),
    ("A atribuição da significância do produto",
     "Em parte. A inferência é condicional aos regressores, incluindo os preços dos combustíveis; nessa "
     "leitura a nota está certa. O 0,208 difere do 0,238 da Tabela 2 porque, na decomposição, a regressão da "
     "gasolina usa os controles da equação do IPCA, e não a dinâmica própria da gasolina."),
    ("Os efeitos totais da Tabela 5 diferem da Tabela 2",
     "Procede. Explicação concreta: na Tabela 5, as três equações usam como controle as defasagens do IPCA sem "
     "combustíveis, para que a identidade feche; na Tabela 2, as defasagens são do próprio IPCA. Daí 0,180 "
     "contra 0,176 e 0,317 contra 0,314."),
    ("Breusch-Godfrey robusto à heterocedasticidade",
     "Procede, ajuste menor: dizer na nota como a versão robusta foi calculada."),
    ("O IPCA Transportes é um grupo do próprio IPCA",
     "Procede. Sugestão para a nota da Figura 4: 'O grupo Transportes contém os combustíveis, e nele a "
     "separação entre regimes é mais nítida.'"),
    ("O Quadro 1 promete códigos",
     "Procede, pendente. O Quadro 1 só traz o código da Selic (SGS 432); faltam Brent (Ipeadata), câmbio, "
     "IBC-Br, IPCA Transportes, códigos dos subgrupos, horizonte do Focus e tabelas do SIDRA."),
    ("Documentar se o preço ANP original já exclui",
     "Conferido no histórico da série: a média de produtores e importadores da ANP inclui Cide e PIS/Cofins, "
     "que são subtraídos pelo calendário de atos legais, sem dupla retirada. A nota já informa a agregação "
     "pela média dos dias do mês civil."),
    # ---------------------------------------------------------------- comentários de formatação do autor
    ("N estou conseguindo subir o texto",
     "O título do capítulo tem 'quebra de página antes', por isso não sobe. É a regra da ABNT: seção primária "
     "começa em página nova. Não é defeito."),
    ("Igual do refwerencial teorico",
     "Mesmo caso: os títulos de capítulo têm 'quebra de página antes', pela regra da ABNT de que seção primária "
     "começa em página nova."),
]
