# -*- coding: utf-8 -*-
"""
Troca a imagem da Figura 7 pela versão no padrão das demais (fig_canal_combustiveis.py) e ajusta a
nota aos painéis A e B (14/10/2026). Os números não mudam.
Backup feito antes: historico/3_versoes_e_backups/backups_do_TCC/_backup_TCC_pre_figura7.docx
"""
import io
from pathlib import Path
import docx
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
ARQ = RAIZ / "TCC_PEDRO_MINELLA.docx"
FIG = RAIZ / "analise_TCC" / "figuras" / "canal_combustiveis_paineis.png"

doc = docx.Document(ARQ)
P = doc.paragraphs
[i] = [i for i, p in enumerate(P) if p.text.startswith("Figura 7 – Decomposição da resposta do IPCA Geral")
       and P[i + 1]._p.xpath(".//a:blip")]
pimg, nota = P[i + 1], P[i + 2]
rid = pimg._p.xpath(".//a:blip/@r:embed")[0]
novo = FIG.read_bytes()
doc.part.related_parts[rid]._blob = novo
w, h = Image.open(io.BytesIO(novo)).size
for ext in pimg._p.xpath(".//wp:extent") + pimg._p.xpath(".//a:ext"):
    ext.set("cy", str(int(int(ext.get("cx")) * h / w)))
assert nota.text.startswith("Nota: à esquerda, a decomposição da resposta acumulada")
nota.runs[0].text = ("Nota: no painel A, a decomposição da resposta acumulada do IPCA Geral no regime posterior "
                     "em efeito total, parcela que transita pelos combustíveis, com faixa de 90%, e parcela não "
                     "atribuída; no painel B, o efeito via combustíveis em cada regime, com faixas de 90%.")
for r in nota.runs[1:]:
    r.text = ""
doc.save(ARQ)
print("salvo", ARQ)
