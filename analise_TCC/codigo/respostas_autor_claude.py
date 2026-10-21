import zipfile, json, sys, shutil
from lxml import etree
src, dst = sys.argv[1], sys.argv[2]
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
norm=lambda s:' '.join(s.split())
respostas={norm(r['resposta']) for r in json.load(open('respostas.json',encoding='utf-8'))}
zin=zipfile.ZipFile(src)
c=etree.fromstring(zin.read('word/comments.xml'))
n=0; autores=set()
for cm in c.findall('{%s}comment'%W):
    t=norm(' '.join(''.join(p.itertext()) for p in cm.findall('{%s}p'%W)))
    if t in respostas:
        autores.add(cm.get('{%s}author'%W))
        cm.set('{%s}author'%W,'Claude'); cm.set('{%s}initials'%W,'C'); n+=1
novo=etree.tostring(c,xml_declaration=True,encoding='UTF-8',standalone=True)
zout=zipfile.ZipFile(dst,'w')
for item in zin.infolist():
    dados = novo if item.filename=='word/comments.xml' else zin.read(item.filename)
    zout.writestr(item, dados)
zout.close()
print('respostas com autor trocado:',n,'autores originais:',autores,'total comentarios:',len(c))
