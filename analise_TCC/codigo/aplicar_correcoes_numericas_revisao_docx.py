# -*- coding: utf-8 -*-
"""
Correções numéricas no arquivo de revisão, como alterações controladas (14/10/2026).

Cada número abaixo foi conferido nos resultados atuais:
  4.6        p ajustado por seleção do IPCA Geral no impacto: 0,104 (quatro séries) e 0,084 (cinco)
             resultados/onda3_3_data_comum.xlsx, aba p_ajustado_selecao
  Tabela H.2 coeficientes de resultados/onda2_7_segunda_volta.xlsx (modo regime); p de Wald da
             igualdade recalculados com a mesma rotina (O2.ajustar sobre choque_pos - choque_pre)
  Apêndice E a escada da Figura E.1 tem 3, 6 e 12 defasagens (onda2_5_lags_residuos.py, ESCADA)
  Tabela 8   elasticidade = resposta em pontos log / 10,19: gasolina em h=1, 3,9765/10,19 = 0,390;
             segunda volta em h=12, 0,21915/10,19 = 0,0215
  3.2        |choque| >= 1 em 51 meses: 25 altas e 26 quedas
  Tabela 3   mês 0 do IPCA Geral: 0,127 no desenho de h=3 contra 0,176 na Figura 5 (onda2_6)
  4.4        no IPCA Geral nenhuma contribuição mensal negativa é significativa a 10% (onda2_6b);
             o mês 11 é negativo e significativo nas outras cinco séries
  3.1        correlação de 0,995 entre os choques do Brent do Ipeadata e da série diária da EIA
             (DCOILBRENTEU, base_de_dados/brent_diario_fred.csv), média mensal, 272 meses
  BCB (2022) Relatório de Inflação de março de 2022, boxe: 39,8% a 54,4% é o grau de repasse do
             petróleo em reais à gasolina ao consumidor, supondo a refinaria acompanhando o petróleo;
             0,20 a 0,27 p.p. vale para 2021 (pesos de dez/2020) e 0,31 a 0,43 para 2022
  López      o PDF consultado é a versão revisada de março de 2026 do ECB WP 2968

Uso: python aplicar_correcoes_numericas_revisao_docx.py [entrada saida]
Backup: historico/3_versoes_e_backups/backups_do_TCC/_backup_revisao_pre_correcoes_numericas.docx
"""
import sys
from pathlib import Path
import docx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 2 else RAIZ / "TCC_PEDRO_MINELLA_revisao_com_alteracoes.docx"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else ENTRADA

doc = docx.Document(ENTRADA)
R = Revisor(doc)

# ------------------------------------------------------------------ 3.1 frase do Brent incompleta
p = R.paragrafo("As séries foram validadas contra as fontes primárias.", aceita_revisao=True)
R.inserir_depois(p, "ECONÔMICA APLICADA, 2025)",
                 " coincide com a série de sua fonte primária (UNITED STATES ENERGY INFORMATION ADMINISTRATION, "
                 "2025): a correlação entre os choques padronizados construídos a partir das duas fontes é de 0,995")

# ------------------------------------------------------------------ 3.2 os 51 meses
p = R.paragrafo("Um choque unitário corresponde, portanto")
R.substituir(p, "igualou ou superou um desvio-padrão,",
             "igualou ou superou um desvio-padrão em valor absoluto, 25 de alta e 26 de queda,")

# ------------------------------------------------------------------ 2.3 calibração do BCB
p = R.paragrafo("A autoridade monetária produziu a estimativa mais próxima")
R.substituir(p, "com os pesos de 2021,", "para 2021,")
R.substituir(p, "A calibração supõe repasse integral do petróleo em reais à refinaria, com coeficiente entre "
                "39,8% e 54,4%.",
             "A calibração supõe que o preço da gasolina na refinaria acompanhe integralmente o petróleo em reais "
             "e obtém, para o preço ao consumidor, grau de repasse entre 39,8% e 54,4%.")

p = R.paragrafo("A parcela associada aos combustíveis, de 0,23 ponto")
R.substituir(p, "com pesos de 2021", "em 2021")

# ------------------------------------------------------------------ 4.4 o décimo primeiro mês
p = R.paragrafo("A resposta acumulada do IPCA Geral recua de 0,314")
R.substituir(p, "recua de 0,314 em três meses para 0,269 em doze. A contribuição do décimo primeiro mês é negativa "
                "e significativa a 10% em cinco séries, sinalizando reversão parcial",
             "passa de 0,314 em três meses para 0,269 em doze, sem contribuição mensal negativa significativa no "
             "índice cheio. Nas outras cinco séries, a contribuição do décimo primeiro mês é negativa e "
             "significativa a 10%, sinalizando reversão parcial nos combustíveis e em Transportes")

# ------------------------------------------------------------------ 4.6 p ajustado do IPCA Geral
p = R.paragrafo("Após refazer a seleção da data em cada réplica")
R.inserir_depois(p, "e 0,013 para Transportes.",
                 " No IPCA Geral, desfecho central, o p-valor ajustado é 0,104 com o conjunto de quatro séries e "
                 "0,084 com o de cinco: corrigida a seleção da data, a diferença no impacto não é significativa "
                 "a 5%.")

# ------------------------------------------------------------------ López et al.
for p in R.paragrafos_com("López et al. (2024)"):
    R.substituir(p, "López et al. (2024)", "López et al. (2026)")
p = R.paragrafo("LÓPEZ, L.; ODENDAHL, F.")
R.substituir(p, "European Central Bank, 2024. (ECB Working Paper, n. 2968).",
             "European Central Bank, 2026. (ECB Working Paper, n. 2968). Versão revisada em março de 2026; "
             "primeira versão de 2024.")

# ------------------------------------------------------------------ Tabela 8
t8 = R.tabela("Estudo", "Känzig (2021)")
linha = {t8.rows[i].cells[0].text.strip(): i for i in range(len(t8.rows))}
R.substituir(R.celula(t8, linha["Känzig (2021)"], 3), "elasticidade de 0,37", "elasticidade de 0,39")
R.substituir(R.celula(t8, linha["Baba e Lee (2022)"], 3), "0,020 em doze", "0,022 em doze")
R.substituir(R.celula(t8, linha["López et al. (2024)"], 0), "López et al. (2024)", "López et al. (2026)")
R.substituir(R.celula(t8, linha["Banco Central do Brasil (2022)"], 2),
             "no IPCA com pesos de 2021; 0,31 a 0,43 com os de 2022", "no IPCA em 2021; 0,31 a 0,43 em 2022")
p = R.paragrafo("Nota: os valores deste trabalho são os coeficientes")
R.substituir(p, "elasticidade é a razão entre a variação do preço e a do petróleo.",
             "elasticidade é a razão entre a variação logarítmica do preço e a do petróleo, de 10,19 pontos por "
             "choque unitário.")

# ------------------------------------------------------------------ Apêndice E
p = R.paragrafo("O Apêndice K.2 reporta que o resultado de regime no horizonte de impacto")
R.substituir(p, "A figura apresenta a escada completa, de uma a doze defasagens, para as variáveis principais.",
             "A figura compara as estimativas com três, seis e doze defasagens, para as variáveis principais.")

# ------------------------------------------------------------------ Tabela H.2 (rodada atual)
th = R.tabela("Subgrupo", "Não comercializáveis")
assert th.rows[5].cells[0].text.strip() == "Não comercializáveis" and th.rows[1].cells[0].text.strip() == "Comercializáveis"
H2 = {(1, 4): ("0,482", "0,480"),
      (2, 2): ("0,128", "0,127"), (2, 3): ("0,215", "0,214"), (2, 4): ("0,471", "0,469"),
      (3, 2): ("0,253", "0,252"), (3, 3): ("0,183", "0,182"),
      (4, 2): ("0,168", "0,167"), (4, 3): ("0,234", "0,233"), (4, 4): ("0,789", "0,788"),
      (5, 4): ("0,826", "0,830"),
      (6, 3): ("0,060", "0,059"), (6, 4): ("0,589", "0,590"),
      (7, 3): ("0,164", "0,163"), (7, 4): ("0,665", "0,666"),
      (8, 3): ("0,228", "0,227")}
for (i, j), (velho, novo) in H2.items():
    R.substituir(R.celula(th, i, j), velho, novo)

# ------------------------------------------------------------------ Tabela 3, nota
p = R.paragrafo("Nota: regime posterior a março de 2017. Conjunto de regressores e amostra fixos")
R.inserir_depois(p, "Os acumulados coincidem com os da Tabela 2.",
                 " Por usar o desenho do horizonte de três meses, inclusive as indicadoras de pandemia daquela "
                 "janela, as contribuições mensais diferem das respostas da Figura 5, estimadas com o desenho "
                 "próprio de cada horizonte: no IPCA Geral, a do mês do choque é 0,127 aqui e 0,176 na Figura 5.")

doc.save(SAIDA)
print("salvo", SAIDA)
