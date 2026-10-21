# -*- coding: utf-8 -*-
"""
Códigos de série no Quadro 1 (Apêndice M) e na referência do SGS, como alteração controlada (15/10/2026).

Cada código foi conferido contra a base (base_de_dados/IPCA.xlsx), baixando a série oficial:
  IPCA Geral            SGS 433, igual em 273 meses
  comercializáveis      SGS 4447 e 4448, diferença máxima de 0,02 ponto (arredondamento)
  câmbio                SGS 3698 (livre, venda, média do mês), diferença máxima de R$ 0,0025; a 3695, citada
                        antes na referência, é de fim de período e difere em até R$ 0,31
  atividade             SGS 24364 (IBC-Br dessazonalizado), correlação 0,9998 (revisões de safra)
  Brent                 Ipeadata EIA366_PBRENT366, média mensal da cotação diária, igual
  expectativa           Focus, IPCA esperado para os 12 meses seguintes (diferença média de 0,005 ponto)
  IPCA por grupos e     SIDRA 655 (ago/1999 a jun/2006), 2938 (jul/2006 a dez/2011), 1419 (2012 a 2019) e
  subitens              7060 (desde 2020), períodos conferidos na API do IBGE

Uso: python aplicar_quadro1_docx.py entrada saida
"""
import sys
from pathlib import Path
import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor

ENT, SAI = sys.argv[1], sys.argv[2]
doc = docx.Document(ENT); R = Revisor(doc)
q = R.tabela("Variável", "Banco Central do Brasil, SGS 432")
linhas = {r.cells[0].text.strip(): i for i, r in enumerate(q.rows)}
TROCAS = {
    "IPCA Geral e Transportes": ("IBGE, SIDRA",
        "IBGE, SIDRA, tabelas 655 (até jun/2006), 2938 (jul/2006 a dez/2011), 1419 (2012 a 2019) e 7060 (desde "
        "2020); IPCA Geral também na SGS 433"),
    "Comercializáveis e não comercializáveis": ("IBGE, séries especiais",
        "Banco Central do Brasil, SGS 4447 e 4448, calculadas a partir do IPCA"),
    "Gasolina, diesel e etanol ao consumidor": ("IBGE, SIDRA (tabela 7060)",
        "IBGE, SIDRA, mesmas tabelas do IPCA por grupos (subitens 5104001, 5104002 e 5104003 na tabela 7060)"),
    "Brent": ("Ipeadata, série de origem EIA",
        "Ipeadata, série EIA366_PBRENT366, média mensal da cotação diária, de origem EIA"),
    "Câmbio": ("Banco Central do Brasil, SGS",
        "Banco Central do Brasil, SGS 3698 (taxa livre, venda, média do mês)"),
    "Atividade econômica": ("Banco Central do Brasil, SGS (IBC-Br)",
        "Banco Central do Brasil, SGS 24364 (IBC-Br com ajuste sazonal)"),
    "Expectativa de inflação": ("Banco Central do Brasil, Focus",
        "Banco Central do Brasil, Focus, IPCA esperado para os 12 meses seguintes"),
}
for var, (velho, novo) in TROCAS.items():
    cel = R.celula(q, linhas[var], 2)
    R.substituir(cel, velho, novo)

ref = [p._p for p in doc.paragraphs if p.text.startswith("BANCO CENTRAL DO BRASIL. Sistema Gerenciador de Séries Temporais")]
assert len(ref) == 1
R.substituir(ref[0], "Séries 432, 3695 e 4448", "Séries 432, 433, 3698, 4447, 4448 e 24364")
doc.save(SAI); print("salvo", SAI)
