# -*- coding: utf-8 -*-
import docx

doc = docx.Document('TCC_Pedro_v7_resultados_finais.docx')
count = 0
for idx, p in enumerate(doc.paragraphs):
    if '—' in p.text or '--' in p.text or ' - ' in p.text:
        print(f"Para {idx}: '{p.text[:120]}'")
        count += 1
print(f"Total paragraphs with dashes: {count}")
