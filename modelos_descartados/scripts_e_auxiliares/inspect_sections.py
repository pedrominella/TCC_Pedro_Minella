# -*- coding: utf-8 -*-
import docx

doc = docx.Document('TCC_Pedro_v7_resultados_finais.docx')
print(f"Total sections: {len(doc.sections)}")
for idx, sec in enumerate(doc.sections):
    print(f"Section {idx}: start_type={sec.start_type}")
    # Let's see if we can find which paragraphs are in this section
    # Note: in python-docx, we can inspect paragraph properties or XML to see section breaks
