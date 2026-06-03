# -*- coding: utf-8 -*-
import docx

doc = docx.Document('TCC_Pedro_v7_resultados_finais.docx')
for i, p in enumerate(doc.paragraphs):
    if p.style.name.startswith('Heading') or p.text.strip().isupper() or len(p.text.strip()) < 50:
        text = p.text.strip()
        if text:
            print(f"Para {i} ({p.style.name}): {text[:100]}")
