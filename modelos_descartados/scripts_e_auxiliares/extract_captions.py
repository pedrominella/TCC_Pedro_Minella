# -*- coding: utf-8 -*-
import docx

doc = docx.Document('TCC_Pedro_v7_resultados_finais.docx')
figures = []
tables = []

for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if text.startswith('Figura '):
        figures.append((i, text))
    elif text.startswith('Tabela '):
        tables.append((i, text))

with open('captions.txt', 'w', encoding='utf-8') as f:
    f.write(f"Found {len(figures)} figures:\n")
    for idx, fig in figures:
        f.write(f"  Para {idx}: '{fig}'\n")
    
    f.write(f"\nFound {len(tables)} tables:\n")
    for idx, tbl in tables:
        f.write(f"  Para {idx}: '{tbl}'\n")
