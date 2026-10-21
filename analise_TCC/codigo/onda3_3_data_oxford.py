# -*- coding: utf-8 -*-
"""
A DATA DA QUEBRA COM O ÍNDICE DE RIGOR DE OXFORD NO LUGAR DAS INDICADORAS (criado em 14/10/2026)

Roda exatamente o procedimento de onda3_3_data_comum.escolher() (soma das estatísticas F
das séries, argmax na grade aparada em 15%, valor crítico por bootstrap selvagem em blocos
de 12 meses com 299 réplicas e a mesma semente), trocando apenas o controle de pandemia:

  dummies   as dez indicadoras de janela de mar-dez/2020 (o que está no TCC); serve de
            conferência: tem de reproduzir mar/2017, 58,1 e o crítico de 42,9
  oxford    o índice de rigor do OxCGRT de mar-dez/2020 (si_covid), um regressor contínuo,
            como na seção 4.7.2; os dez meses voltam à estimação

As saídas vão para resultados/data_quebra_oxford/<modo>/onda3_3_data_comum.xlsx e não
tocam no arquivo do TCC (resultados/onda3_3_data_comum.xlsx).

Uso: python onda3_3_data_oxford.py dummies | oxford
"""
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lp_base_nova as L
import teste_oxford_covid as OX
import onda3_3_data_comum as D

modo = sys.argv[1] if len(sys.argv) > 1 else "oxford"
assert modo in ("dummies", "oxford"), modo

if modo == "oxford":
    _b, _dum, _pand, _ = OX.prepara()          # base com si_covid, zerado fora de mar-dez/2020

    def _carregar(juros="corrigida", **kw):
        assert juros == "corrigida"
        return _b, _dum, (lambda d, h: ["si_covid"])

    L.carregar_pandemia = _carregar           # onda3_3 chama L.carregar_pandemia("corrigida")

saida = AQUI.parent / "resultados" / "data_quebra_oxford" / modo
saida.mkdir(parents=True, exist_ok=True)
D.SAIDA = saida
t, aj = D.escolher()

if modo == "dummies":
    q = t.set_index("conjunto").loc["4 principais"]
    assert f"{q.data_escolhida:%Y-%m}" == "2017-03", q.data_escolhida
    assert round(q.estatistica, 1) == 58.1 and round(q.critico_95, 1) == 42.9, (q.estatistica, q.critico_95)
    print("conferência OK: reproduz mar/2017, 58,1 e 42,9")
