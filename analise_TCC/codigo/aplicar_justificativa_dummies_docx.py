# -*- coding: utf-8 -*-
"""
Reescreve o parágrafo da seção 4.7.2 sobre a data e o tratamento da pandemia (14/10/2026).

O parágrafo anterior citava 67,5 e 27,8, números de um sup-Wald rápido (teste_oxford_covid.py)
que não é a estatística da Tabela 6 (58,1 contra crítico de 42,9). Agora os números vêm do
mesmo procedimento da Tabela 6, rodado com o índice de rigor no lugar das indicadoras
(onda3_3_data_oxford.py), e o parágrafo explica por que as indicadoras ficam no modelo principal.

Fontes conferidas com assert:
  resultados/data_quebra_oxford/dummies/onda3_3_data_comum.xlsx   (reproduz a Tabela 6)
  resultados/data_quebra_oxford/oxford/onda3_3_data_comum.xlsx
  resultados/comparacao_2020_pandemia.xlsx                         (0,088 com e sem o índice)
Backup feito antes: historico/3_versoes_e_backups/backups_do_TCC/_backup_TCC_pre_justificativa_dummies.docx
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
import docx

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import lp_base_nova as L
import teste_oxford_covid as OX
import onda2_6_resposta_nao_acumulada as N

RAIZ = AQUI.parents[1]
ARQ = RAIZ / "TCC_PEDRO_MINELLA.docx"
RES = RAIZ / "analise_TCC" / "resultados"


def v(x, casas=1):
    return (f"%.{casas}f" % x).replace(".", ",").replace("-", "−")


# ---------------------------------------------------------------- números da data
def escolha(modo):
    t = pd.read_excel(RES / "data_quebra_oxford" / modo / "onda3_3_data_comum.xlsx", sheet_name="escolha")
    p = pd.read_excel(RES / "data_quebra_oxford" / modo / "onda3_3_data_comum.xlsx", sheet_name="perfil")
    p["data"] = pd.to_datetime(p["data"])
    return t.set_index("conjunto"), p


td, _ = escolha("dummies")
assert f"{pd.Timestamp(td.loc['4 principais', 'data_escolhida']):%Y-%m}" == "2017-03"
assert v(td.loc["4 principais", "estatistica"]) == "58,1" and v(td.loc["4 principais", "critico_95"]) == "42,9"

to, po = escolha("oxford")
q4, q5 = to.loc["4 principais"], to.loc["5 com refinaria"]
p4 = po[po.conjunto == "4 principais"]
mar17 = float(p4.loc[p4.data == "2017-03-01", "soma_F"].iloc[0])
out16 = float(p4.loc[p4.data == "2016-10-01", "soma_F"].iloc[0])
acima = p4[p4.soma_F >= p4.critico_95].data.sort_values()
assert f"{pd.Timestamp(q4.data_escolhida):%Y-%m}" == "2021-02"
assert [v(q4.estatistica), v(q4.critico_95), v(mar17), v(out16)] == ["59,4", "54,3", "22,2", "22,4"]
assert [f"{d:%Y-%m}" for d in acima] == ["2020-10", "2020-11", "2020-12", "2021-01", "2021-02"]
assert f"{pd.Timestamp(q5.data_escolhida):%Y-%m}" == "2016-11"
assert [v(q5.estatistica), v(q5.critico_95)] == ["77,2", "55,8"]

# ---------------------------------------------------------------- índice de rigor contra sem controle
c = pd.read_excel(RES / "comparacao_2020_pandemia.xlsx", sheet_name="coeficientes").set_index("tratamento")
ox_imp = c.loc["Índice de rigor de Oxford", "impacto_depois"]
sc_imp = c.loc["Sem controle de pandemia", "impacto_depois"]
assert v(ox_imp, 3) == "0,088" and v(sc_imp, 3) == "0,088"

b, dum, pand, _ = OX.prepara()
d, din = N._base("ipca_geral", b)
d["__y"] = L.y_acum(d, "ipca_geral", 0)
base = ["choque_pre", "choque_pos", "regime_pos"] + din + dum
am = d[["Data", "__y"] + base + ["si_covid"]].replace([np.inf, -np.inf], np.nan).dropna()
Z = sm.add_constant(am[[x for x in base if x != "choque_pos"]], has_constant="add")
rho = np.corrcoef(sm.OLS(am["choque_pos"], Z).fit().resid, sm.OLS(am["si_covid"], Z).fit().resid)[0, 1]
assert v(rho, 3) == "−0,003", rho

NOVO = (
    "Cabe declarar a contrapartida, que é também a razão de as indicadoras constituírem a construção adotada. "
    "Repetida a busca da seção 4.6 com o índice de rigor no lugar das indicadoras, pelo mesmo procedimento de "
    "bootstrap, o máximo do conjunto das quatro séries principais desloca-se para fevereiro de 2021, com "
    f"estatística de {v(q4.estatistica)} contra valor crítico de {v(q4.critico_95)}, e apenas as datas de outubro "
    f"de 2020 a fevereiro de 2021 superam o crítico, ao passo que março de 2017 cai para {v(mar17)} e outubro de "
    f"2016 para {v(out16)}. Com os dez meses de 2020 na amostra, o procedimento passa a datar a própria pandemia, "
    "e não a mudança da regra de preços. Acrescida a gasolina na refinaria, a data volta a novembro de 2016, com "
    f"{v(q5.estatistica)} contra crítico de {v(q5.critico_95)}, de modo que o deslocamento se concentra nas séries "
    "de preços ao consumidor. O índice, por sua vez, quase nada controla do choque: com ele o coeficiente de "
    f"impacto, {v(ox_imp, 3)}, coincide com o obtido sem controle algum de pandemia, porque nos meses de 2020 o "
    f"índice é praticamente ortogonal ao choque, com correlação parcial de {v(rho, 3)}. As indicadoras impedem "
    "que um choque sanitário, sem relação com a regra, determine a datação, e por isso permanecem na "
    "especificação principal, com o índice de rigor reportado como alternativa."
)

# ---------------------------------------------------------------- documento
doc = docx.Document(ARQ)
P = doc.paragraphs
alvo = P[397]
assert alvo.text.startswith("Cabe declarar a contrapartida. A data de março de 2017 não é invariante"), alvo.text[:80]
assert "67,5" in alvo.text and "27,8" in alvo.text
alvo.runs[0].text = NOVO
for r in alvo.runs[1:]:
    r.text = ""
doc.save(ARQ)
print(NOVO)
print("salvo", ARQ)
