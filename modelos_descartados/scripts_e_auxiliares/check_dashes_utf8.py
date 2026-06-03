# -*- coding: utf-8 -*-
import docx

doc = docx.Document('TCC_Pedro_v7_resultados_finais.docx')
with open('tcc_dashes.txt', 'w', encoding='utf-8') as f:
    count = 0
    for idx, p in enumerate(doc.paragraphs):
        if '—' in p.text or '--' in p.text or ' - ' in p.text:
            f.write(f"Para {idx}: '{p.text[:120]}'\n")
            count += 1
    f.write(f"Total paragraphs with dashes: {count}\n")
