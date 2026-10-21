# -*- coding: utf-8 -*-
"""
VERSAO DE ARTIGO: DATA DE QUEBRA IMPOSTA EM SETEMBRO DE 2016

Motivo. No TCC a data de corte e ESTIMADA (mar/2017, sup-Wald de Andrews com
valores criticos por reamostragem). Isso e defensavel, mas cria duas exposicoes
que um parecerista ataca primeiro:

  1. o pico do sup-Wald se desloca para fev/2021 quando os dez meses de 2020
     voltam a amostra, e o conjunto que os dados nao descartam vai de abr/2016 a
     fev/2021, quase cinco anos;
  2. a data foi escolhida maximizando uma estatistica, o que exige inferencia
     pos-selecao.

A alternativa e IMPOR a data do anuncio publico da politica de paridade de
importacao, setembro de 2016. Deixa de haver estimacao da data, deixa de haver
selecao, e o exercicio de sup-Wald passa a ser CORROBORACAO e nao identificacao.

Este script NAO altera nada do TCC. Ele monkey-patcha a constante da data e
redireciona a saida dos modulos ja verificados para pastas proprias:

    resultados_artigo/   figuras_artigo/

Uso:
    python artigo_set2016.py rodar     reestima tudo com set/2016
    python artigo_set2016.py figuras   gera as figuras de comparacao
    python artigo_set2016.py           faz os dois
"""
import sys, importlib, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
PASTA = AQUI.parent
SAIDA_A = PASTA / "resultados_artigo"; SAIDA_A.mkdir(exist_ok=True)
FIGS_A = PASTA / "figuras_artigo"; FIGS_A.mkdir(exist_ok=True)

DATA_ARTIGO = "2016-09-01"          # anuncio publico da PPI
DATA_TCC = "2017-03-01"             # estimada, usada no TCC

MODULOS = ["onda2_8_teulings_zubanov",      # resposta acumulada por regime
           "onda2_6_resposta_nao_acumulada",  # resposta nao acumulada
           "onda2_4_canal_combustiveis",      # decomposicao do canal
           "onda2_7_segunda_volta"]           # primeira e segunda volta


def rodar():
    import lp_base_nova as L
    print("=" * 78)
    print("REESTIMANDO COM CORTE IMPOSTO EM %s (anuncio da PPI)" % DATA_ARTIGO)
    print("  saida: %s" % SAIDA_A.name)
    print("=" * 78)
    L.DATA_CORTE_FINAL = DATA_ARTIGO
    L.SAIDA = SAIDA_A
    for nome in MODULOS:
        m = importlib.import_module(nome)
        m.SAIDA = SAIDA_A                       # redireciona planilhas
        if hasattr(m, "FIGS"):
            m.FIGS = FIGS_A
        if hasattr(m, "L"):
            m.L.DATA_CORTE_FINAL = DATA_ARTIGO  # o modulo compartilha o mesmo L
        print("\n>>> %s" % nome)
        m.rodar()
    print("\nconcluido. planilhas em %s" % SAIDA_A)


if __name__ == "__main__":
    acao = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    if acao in ("rodar", "tudo"):
        rodar()
    if acao in ("figuras", "tudo"):
        import figuras_artigo
        figuras_artigo.main()
