# -*- coding: utf-8 -*-
import docx

doc = docx.Document('TCC_Pedro_v7_resultados_finais.docx')
with open('tcc_elements.txt', 'w', encoding='utf-8') as f:
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if text:
            # check if it is a heading or has short text
            if p.style.name.startswith('Heading') or len(text) < 150:
                f.write(f"Para {i} ({p.style.name}): {text}\n")
            else:
                f.write(f"Para {i} ({p.style.name}): [LONG TEXT, {len(text)} chars]: {text[:80]}...\n")
