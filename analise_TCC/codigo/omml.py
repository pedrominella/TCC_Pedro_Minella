# -*- coding: utf-8 -*-
"""
Construtor de equações nativas do Word (OMML) e inserção como alteração controlada (14/10/2026).

As equações do trabalho estavam como texto linear (β_pós(h), Σ(i=1..3), 1{t ≥ τ}); aqui elas
são montadas como objetos de equação do Word, com sub e sobrescritos, somatórios, frações e
delimitadores de verdade, na fonte Cambria Math.

Marcação de revisão: cada m:r vai envolvido em w:ins e cada m:ctrlPr leva w:ins, que é a forma
pela qual o Word registra uma equação digitada com controle de alterações; rejeitar a marcação
remove a equação inteira.
"""
from docx.oxml.ns import qn

MNS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def m(tag):
    return "{%s}%s" % (MNS, tag)


def w(tag):
    return "{%s}%s" % (WNS, tag)


class Eq:
    """Monta OMML; `rev` é uma função que devolve um elemento w:ins novo (ou None, sem revisão)."""

    def __init__(self, fabrica, rev=None, tamanho=None):
        self.f, self.rev, self.sz = fabrica, rev, tamanho   # fabrica: um elemento qualquer do documento

    def _el(self, tag, attrs=None):
        return self.f.makeelement(tag, attrs or {})

    def _wrpr(self):
        rpr = self._el(w("rPr"))
        fonts = self._el(w("rFonts"), {w("ascii"): "Cambria Math", w("hAnsi"): "Cambria Math"})
        rpr.append(fonts)
        if self.sz:
            rpr.append(self._el(w("sz"), {w("val"): str(self.sz)}))
            rpr.append(self._el(w("szCs"), {w("val"): str(self.sz)}))
        return rpr

    def _ctrl(self):
        c = self._el(m("ctrlPr"))
        if self.rev:
            i = self.rev()
            i.append(self._wrpr())
            c.append(i)
        else:
            c.append(self._wrpr())
        return c

    # ------------------------------------------------------------ elementos
    def r(self, texto, reto=False, palavra=False):
        """Run de equação. reto=True: sem itálico (nomes de função, rótulos de índice).
        palavra=True: texto comum dentro da equação (palavras, espaços), sem itálico e com espaços."""
        run = self._el(m("r"))
        if reto or palavra:
            mrpr = self._el(m("rPr"))
            if palavra:
                mrpr.append(self._el(m("nor")))
            else:
                mrpr.append(self._el(m("sty"), {m("val"): "p"}))
            run.append(mrpr)
        run.append(self._wrpr())
        t = self._el(m("t"))
        t.text = texto
        if texto != texto.strip():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        run.append(t)
        if self.rev:
            i = self.rev()
            i.append(run)
            return [i]
        return [run]

    def _arg(self, tag, conteudo):
        e = self._el(m(tag))
        for x in self._lista(conteudo):
            e.append(x)
        return e

    def _lista(self, conteudo):
        if isinstance(conteudo, str):
            return self.r(conteudo)
        saida = []
        for x in conteudo:
            if isinstance(x, str):
                saida.extend(self.r(x))
            elif hasattr(x, "tag"):          # elemento já montado
                saida.append(x)
            else:                            # lista de elementos
                saida.extend(self._lista(x))
        return saida

    def sub(self, base, indice):
        s = self._el(m("sSub"))
        pr = self._el(m("sSubPr")); pr.append(self._ctrl()); s.append(pr)
        s.append(self._arg("e", base)); s.append(self._arg("sub", indice))
        return [s]

    def sup(self, base, exp):
        s = self._el(m("sSup"))
        pr = self._el(m("sSupPr")); pr.append(self._ctrl()); s.append(pr)
        s.append(self._arg("e", base)); s.append(self._arg("sup", exp))
        return [s]

    def subsup(self, base, indice, exp):
        s = self._el(m("sSubSup"))
        pr = self._el(m("sSubSupPr")); pr.append(self._ctrl()); s.append(pr)
        s.append(self._arg("e", base)); s.append(self._arg("sub", indice)); s.append(self._arg("sup", exp))
        return [s]

    def d(self, conteudo, abre="(", fecha=")"):
        el = self._el(m("d"))
        pr = self._el(m("dPr"))
        if abre != "(":
            pr.append(self._el(m("begChr"), {m("val"): abre}))
        if fecha != ")":
            pr.append(self._el(m("endChr"), {m("val"): fecha}))
        pr.append(self._ctrl()); el.append(pr)
        el.append(self._arg("e", conteudo))
        return [el]

    def frac(self, num, den, tipo=None):
        """tipo="lin": fração em linha, a/b, no tamanho do texto."""
        el = self._el(m("f"))
        pr = self._el(m("fPr"))
        if tipo:
            pr.append(self._el(m("type"), {m("val"): tipo}))
        pr.append(self._ctrl()); el.append(pr)
        el.append(self._arg("num", num)); el.append(self._arg("den", den))
        return [el]

    def soma(self, inferior, superior, corpo, simbolo="∑"):
        el = self._el(m("nary"))
        pr = self._el(m("naryPr"))
        pr.append(self._el(m("chr"), {m("val"): simbolo}))
        pr.append(self._el(m("limLoc"), {m("val"): "undOvr"}))
        if superior is None:
            pr.append(self._el(m("supHide"), {m("val"): "1"}))
        pr.append(self._ctrl()); el.append(pr)
        el.append(self._arg("sub", inferior or []))
        el.append(self._arg("sup", superior or []))
        el.append(self._arg("e", corpo))
        return [el]

    def func(self, nome, arg):
        el = self._el(m("func"))
        pr = self._el(m("funcPr")); pr.append(self._ctrl()); el.append(pr)
        el.append(self._arg("fName", self.r(nome, reto=True)))
        el.append(self._arg("e", arg))
        return [el]

    def func_lim(self, nome, limite, arg):
        """Função com limite embaixo do nome, como arg max com τ ∈ C."""
        el = self._el(m("func"))
        pr = self._el(m("funcPr")); pr.append(self._ctrl()); el.append(pr)
        el.append(self._arg("fName", self.limabaixo(self.r(nome, reto=True), limite)))
        el.append(self._arg("e", arg))
        return [el]

    def limabaixo(self, base, limite):
        """Ex.: arg max com τ ∈ C embaixo."""
        el = self._el(m("limLow"))
        pr = self._el(m("limLowPr")); pr.append(self._ctrl()); el.append(pr)
        el.append(self._arg("e", base)); el.append(self._arg("lim", limite))
        return [el]

    def acento(self, base, caractere="̃"):
        el = self._el(m("acc"))
        pr = self._el(m("accPr"))
        pr.append(self._el(m("chr"), {m("val"): caractere}))
        pr.append(self._ctrl()); el.append(pr)
        el.append(self._arg("e", base))
        return [el]

    def omath(self, *partes):
        om = self._el(m("oMath"))
        for x in self._lista(list(partes)):
            om.append(x)
        return om
