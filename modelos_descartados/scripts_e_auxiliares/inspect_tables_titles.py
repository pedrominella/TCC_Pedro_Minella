# -*- coding: utf-8 -*-
import docx

doc = docx.Document('TCC_Pedro_v7_resultados_finais.docx')
body = doc.element.body

# Let's map elements to see what is immediately preceding each table
element_list = list(body)
with open('tables_preceding.txt', 'w', encoding='utf-8') as f:
    for idx, table in enumerate(doc.tables):
        # find the table in the element list
        tbl_el = table._tbl
        tbl_idx = element_list.index(tbl_el)
        
        # let's look at the elements before the table
        preceding_paras = []
        for offset in range(1, 4):
            prev_idx = tbl_idx - offset
            if prev_idx >= 0:
                prev_el = element_list[prev_idx]
                if prev_el.tag.endswith('p'):
                    # it is a paragraph
                    p = docx.text.paragraph.Paragraph(prev_el, doc)
                    preceding_paras.append(p.text.strip())
        
        f.write(f"Table {idx}:\n")
        f.write(f"  Header row: {[c.text.strip() for c in table.rows[0].cells[:4]]}\n")
        f.write(f"  Preceding paragraphs (reverse order): {preceding_paras}\n\n")
