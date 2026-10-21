# -*- coding: utf-8 -*-
"""
Ajustes oferecidos nas respostas aos comentários, como alterações controladas (14/10/2026).

  1  Tabela 6: cabeçalho "Faixa compatível" -> "Datas que rejeitam a 5%"
  2  4.7.5: sai "além da assimetria indicativa do diesel" (o contrafactual usa coeficientes do IPCA,
     cuja simetria não é rejeitada no K.4)
  3  Tabela 5: linha de h = 12 (onda2_7b_voltas_por_regime.xlsx, cenário central, posterior)
  4  Tabela H.2: intervalos de 90% da diferença entre regimes na nota (mesma rotina do Wald)
  5  4.3: frase sobre o K.6 (onda6_regime_interagido.xlsx)
  6  Conclusão: p ajustado por seleção do IPCA Geral, 0,104 (onda3_3_data_comum.xlsx)
  7  Figura 7: nova versão (fig_canal_combustiveis.py), faixa só do efeito total no painel da
     direita; troca de imagem e de nota marcadas
  8  Parágrafo de Correa e Minella movido para depois da Figura 1 (exclusão e inserção marcadas;
     as âncoras dos comentários vão junto)

Uso: python aplicar_ajustes_comentarios_docx.py [entrada saida]
Backup: historico/3_versoes_e_backups/backups_do_TCC/_backup_revisao_pre_ajustes_comentarios.docx
"""
import copy, io, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import docx
from docx.oxml.ns import qn
from PIL import Image

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from revisao_track import Revisor
import lp_base_nova as L
import onda2_bandas_placebo as O2
from onda2_7_segunda_volta import preparar, _dinamica

RAIZ = AQUI.parents[1]
RES = AQUI.parent / "resultados"
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 2 else RAIZ / "TCC_PEDRO_MINELLA_revisao_com_alteracoes.docx"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else ENTRADA
FIG7 = AQUI.parent / "figuras" / "canal_combustiveis_paineis_v2.png"


def br(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


doc = docx.Document(ENTRADA)
R = Revisor(doc)
corpo = doc.element.body


def marca_paragrafo(p, tipo):
    """Marca a marca de parágrafo como inserida ou excluída (w:pPr/w:rPr/w:ins ou w:del)."""
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = p.makeelement(qn("w:pPr"), {}); p.insert(0, ppr)
    rpr = ppr.find(qn("w:rPr"))
    if rpr is None:
        rpr = ppr.makeelement(qn("w:rPr"), {})
        fim = ppr.find(qn("w:sectPr")) if ppr.find(qn("w:sectPr")) is not None else ppr.find(qn("w:pPrChange"))
        (fim.addprevious(rpr) if fim is not None else ppr.append(rpr))
    rpr.insert(0, R._marca(tipo))


W14_PARAID = "{http://schemas.microsoft.com/office/word/2010/wordml}paraId"
_ids_usados = {e.get(W14_PARAID) for e in corpo.iter() if e.get(W14_PARAID)}


def ids_novos(el):
    """Cópias de linha ou parágrafo recebem w14:paraId novos, únicos no documento."""
    import random
    rnd = random.Random(20261014 + len(_ids_usados))
    for e in el.iter():
        if e.get(W14_PARAID):
            while True:
                novo = "%08X" % rnd.randrange(1, 0x7FFFFFFF)
                if novo not in _ids_usados:
                    break
            _ids_usados.add(novo)
            e.set(W14_PARAID, novo)


def unico(lista, rotulo):
    assert len(lista) == 1, (rotulo, len(lista))
    return lista[0]


# ------------------------------------------------------------------ 1 Tabela 6
t6 = R.tabela("Conjunto de séries", "Quatro principais")
R.substituir(R.celula(t6, 0, 5), "Faixa compatível", "Datas que rejeitam a 5%")

# ------------------------------------------------------------------ 2 4.7.5 sem o diesel
p = R.paragrafo("Em 2020, a contribuição posterior termina em")
R.substituir(p, ", além da assimetria indicativa do diesel,", "")

# ------------------------------------------------------------------ 3 Tabela 5, h = 12
v = pd.read_excel(RES / "onda2_7b_voltas_por_regime.xlsx")
v = v[(v.cenario == "central") & (v.regime == "posterior") & (v.h == 12)].iloc[0]
estrela = "**" if v.p_segunda_volta < 0.05 else ("*" if v.p_segunda_volta < 0.10 else "")
nova = ["12", br(v.efeito_total), br(v.primeira_volta), br(v.segunda_volta) + estrela,
        br(v.p_segunda_volta), f"{round(100 * v.parcela_primeira):d}%"]
assert nova == ["12", "0,273", "0,054", "0,219**", "0,014", "20%"], nova
t5 = R.tabela("h", "% primeira volta")
ultima = t5.rows[-1]._tr
assert [c.text for c in t5.rows[-1].cells][0] == "6"
tr = copy.deepcopy(ultima)
ids_novos(tr)
trpr = tr.find(qn("w:trPr"))
if trpr is None:
    trpr = tr.makeelement(qn("w:trPr"), {}); tr.insert(0 if tr.find(qn("w:tblPrEx")) is None else 1, trpr)
trpr.append(R._marca("w:ins"))
for tc, texto in zip(tr.findall(qn("w:tc")), nova):
    ps = tc.findall(qn("w:p")); assert len(ps) == 1
    p = ps[0]
    modelo = p.find(".//" + qn("w:r"))
    for filho in list(p):
        if filho.tag != qn("w:pPr"):
            p.remove(filho)
    ins = R._marca("w:ins"); ins.append(R._run_novo(modelo, texto)); p.append(ins)
    marca_paragrafo(p, "w:ins")
# a borda inferior que fecha a tabela passa para a linha de 12 meses. Para que rejeitar devolva a
# tabela exata, a linha de 6 meses atual fica marcada como excluída e entra uma cópia dela sem a
# borda inferior, marcada como inserida (mudança de formatação de célula controlada desmonta a
# tabela ao ser rejeitada no Word; testado em 14/10/2026)
seis = copy.deepcopy(ultima)
ids_novos(seis)
seis.find(qn("w:trPr")).append(R._marca("w:ins"))
for tc in seis.findall(qn("w:tc")):
    baixo = tc.find(qn("w:tcPr") + "/" + qn("w:tcBorders") + "/" + qn("w:bottom"))
    assert baixo is not None and baixo.get(qn("w:val")) == "single"
    for k in list(baixo.attrib):
        del baixo.attrib[k]
    baixo.set(qn("w:val"), "nil")
    for p in tc.findall(qn("w:p")):
        for r in p.findall(qn("w:r")):
            i = R._marca("w:ins"); r.addprevious(i); i.append(r)
        marca_paragrafo(p, "w:ins")
ultima.find(qn("w:trPr")).append(R._marca("w:del"))
for tc in ultima.findall(qn("w:tc")):
    for p in tc.findall(qn("w:p")):
        for r in p.findall(qn("w:r")):
            dl = R._marca("w:del"); r.addprevious(dl); dl.append(r)
            for t in r.findall(qn("w:t")):
                t.tag = qn("w:delText")
        marca_paragrafo(p, "w:del")
ultima.addnext(seis)
seis.addnext(tr)

# ------------------------------------------------------------------ 4 Tabela H.2, intervalos
b, dum, pand = L.carregar_pandemia("corrigida"); b, _ = preparar(b)
b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"]); b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
ic = {}
for nome in ["ipca_comercializaveis", "ipca_naocomercializaveis"]:
    d = b.copy(); din = _dinamica(d, nome)
    for h in (0, 1, 3, 6):
        d["__y"] = L.y_acum(d, nome, h)
        regs = ["choque_pre", "choque_pos", "regime_pos"] + din + dum + pand(d, h)
        am = d[["Data", "__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
        g = O2.ajustar(am, regs, h, ("choque_pre", "choque_pos"))
        ic[(nome, h)] = (g["valor"] - L.Z_90 * g["ep"], g["valor"] + L.Z_90 * g["ep"])
assert all(a < 0 < s for a, s in ic.values())
lista = lambda nome: ", ".join(f"[{br(a)}; {br(s)}]" for (n, h), (a, s) in ic.items() if n == nome)
p = R.paragrafo("Nota: mesma especificação da Tabela 2, com os subgrupos como dependentes.")
R.inserir_depois(p, "Asteriscos indicam significância a 10% (*) e 5% (**).",
                 " Intervalos de 90% da diferença entre regimes, posterior menos anterior, em h = 0, 1, 3 e 6: "
                 f"comercializáveis {lista('ipca_comercializaveis')}; não comercializáveis "
                 f"{lista('ipca_naocomercializaveis')}. Todos contêm o zero e são amplos em relação às respostas "
                 "estimadas, de modo que a não rejeição não exclui diferenças da ordem das próprias respostas.")

# ------------------------------------------------------------------ 5 4.3, K.6
k6 = pd.read_excel(RES / "onda6_regime_interagido.xlsx")
assert int((k6.wald_p_dinamica_comum < 0.05).sum()) == 56 and len(k6) == 78
ip = k6[k6.variavel == "ipca_geral"].set_index("h").wald_p_igualdade
assert ip[0] < 0.001 and round(ip[1], 3) == 0.001 and round(ip[3], 3) == 0.349
p = R.paragrafo("Nos horizontes de impacto, um e três meses, os dezoito coeficientes anteriores")
R.inserir_depois(p, "(Apêndice K.6).",
                 " A restrição de dinâmica comum é rejeitada a 5% em 56 das 78 equações; liberada, a igualdade "
                 "entre regimes no IPCA Geral continua rejeitada no impacto e em um mês (p < 0,001 e p = 0,001), "
                 "mas não em três meses (p = 0,349).")

# ------------------------------------------------------------------ 6 Conclusão, p ajustado do IPCA Geral
aj = pd.read_excel(RES / "onda3_3_data_comum.xlsx", sheet_name="p_ajustado_selecao")
pa = aj[(aj.conjunto == "4 principais") & (aj.variavel == "ipca_geral") & (aj.h == 0)].p_ajustado_selecao.iloc[0]
assert round(pa, 3) == 0.104
p = R.paragrafo("A interpretação enfrenta limites:")
R.inserir_depois(p, "sem localizar precisamente a quebra;",
                 " corrigida a seleção da data, a diferença do IPCA Geral no impacto não é significativa a 5% "
                 "(p = 0,104);")

# ------------------------------------------------------------------ 7 Figura 7
paras = list(corpo.iter(qn("w:p")))
leg = unico([i for i, x in enumerate(paras)
             if "".join(t.text or "" for t in x.iter(qn("w:t"))).startswith("Figura 7 – Decomposição")
             and paras[i + 1].find(".//" + qn("w:drawing")) is not None], "legenda da Figura 7")
pimg = paras[leg + 1]
run_old = unico([r for r in pimg.iter(qn("w:r")) if r.find(".//" + qn("w:drawing")) is not None], "run da imagem")
rid_novo, _ = doc.part.get_or_add_image(str(FIG7))
run_new = copy.deepcopy(run_old)
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
for blip in run_new.iter(A + "blip"):
    blip.set(RNS + "embed", rid_novo)
largura, altura = Image.open(FIG7).size
for ext in list(run_new.iter(WP + "extent")) + list(run_new.iter(A + "ext")):
    ext.set("cy", str(int(int(ext.get("cx")) * altura / largura)))
maior = max(int(x.get("id")) for x in corpo.iter(WP + "docPr"))
for pr in run_new.iter(WP + "docPr"):
    pr.set("id", str(maior + 1)); pr.set("name", "Figura 7 v2")
    pr.set("descr", "Efeito via combustíveis por regime e decomposição da resposta do IPCA Geral no regime posterior")
dl = R.apagar_elemento(run_old)
ins = R._marca("w:ins"); ins.append(run_new); dl.addnext(ins)
nota = paras[leg + 2]
assert "".join(t.text or "" for t in nota.iter(qn("w:t"))).startswith("Nota: no painel A, a decomposição")
R.substituir(nota, "no painel A, a decomposição da resposta acumulada do IPCA Geral no regime posterior em efeito "
                   "total, parcela que transita pelos combustíveis, com faixa de 90%, e parcela não atribuída; no "
                   "painel B, o efeito via combustíveis em cada regime, com faixas de 90%.",
             "no painel A, o efeito via combustíveis em cada regime, com faixas de 90%; no painel B, a "
             "decomposição da resposta acumulada do IPCA Geral no regime posterior em efeito total, com faixa de "
             "90%, parcela associada aos combustíveis e parcela não atribuída.")

# ------------------------------------------------------------------ 8 Correa e Minella depois da Figura 1
origem = R.paragrafo("Existe, por fim, precedente brasileiro para a ideia central deste trabalho")
fonte = R.paragrafo("Fonte: Elaboração própria a partir de dados da EIA, da ANP e do Banco Central do Brasil.")
paras = list(corpo.iter(qn("w:p")))
assert paras.index(fonte) - paras.index(origem) == 4, "a Figura 1 não está logo depois do parágrafo"
copia = copy.deepcopy(origem)
ids_novos(copia)
for el in list(copia):                       # a cópia leva só pPr e texto; as âncoras dos comentários vêm do original
    if el.tag in (qn("w:commentRangeStart"), qn("w:commentRangeEnd"), qn("w:bookmarkStart"), qn("w:bookmarkEnd")):
        copia.remove(el)
    elif el.tag == qn("w:r") and el.find(qn("w:commentReference")) is not None:
        copia.remove(el)
for el in copia.findall(qn("w:r")):
    i = R._marca("w:ins"); el.addprevious(i); i.append(el)
marca_paragrafo(copia, "w:ins")
inicios = origem.findall(qn("w:commentRangeStart"))
fins = origem.findall(qn("w:commentRangeEnd")) + [r for r in origem.findall(qn("w:r")) if r.find(qn("w:commentReference")) is not None]
texto = "".join(t.text or "" for t in origem.iter(qn("w:t")))
R.substituir(origem, texto, "")
marca_paragrafo(origem, "w:del")
ppr = copia.find(qn("w:pPr"))
for el in reversed(inicios):
    ppr.addnext(el)
for el in fins:
    copia.append(el)
fonte.addnext(copia)

doc.save(SAIDA)
print("salvo", SAIDA)
