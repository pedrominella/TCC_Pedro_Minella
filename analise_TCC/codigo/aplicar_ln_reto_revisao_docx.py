# -*- coding: utf-8 -*-
"""
Seção 3.3: refaz a equação em linha 100·ln(1+r/100) com "ln" como nome de função, sem itálico
(14/10/2026). A equação antiga, já do Word, tinha "ln" como duas variáveis em itálico. Alteração
controlada, autor "Claude": a equação antiga fica marcada como excluída e a nova como inserida.

Uso: python aplicar_ln_reto_revisao_docx.py [entrada saida]
Backup: historico/3_versoes_e_backups/backups_do_TCC/_backup_revisao_pre_ln.docx
"""
import sys
from pathlib import Path
import docx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor
from omml import Eq

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 2 else RAIZ / "TCC_PEDRO_MINELLA_revisao_com_alteracoes.docx"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else ENTRADA
MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

doc = docx.Document(ENTRADA)
R = Revisor(doc)
E = Eq(doc.element.body)

alvo = [om for p in doc.paragraphs if p.text.startswith("O vetor de controles X reúne")
        for om in p._p.iter(MATH + "oMath")
        if "".join(t.text or "" for t in om.iter(MATH + "t")) == "100·ln1+r/100"]
assert len(alvo) == 1, len(alvo)
d = R.apagar_elemento(alvo[0])
ins = R._marca("w:ins")
ins.append(E.omath("100·", E.func("ln", E.d("1+r/100"))))
d.addnext(ins)

doc.save(SAIDA)
print("salvo", SAIDA)
