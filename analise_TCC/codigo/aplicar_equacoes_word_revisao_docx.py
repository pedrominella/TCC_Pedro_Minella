# -*- coding: utf-8 -*-
"""
Equações no formato nativo do Word, como alterações controladas (14/10/2026).

Pedido do autor: as equações (1) a (12) estavam escritas como texto linear, à moda de LaTeX
(β_pós(h), Σ(i=1..3), 1{t ≥ τ}, IPCA^ex), e a (13)-(14) era uma equação do Word montada de forma
improvisada (Σ seguido de "h=0..H", número da equação dentro da fórmula). Aqui todas passam a
objetos de equação do Word (OMML), com somatórios, sub e sobrescritos, frações e delimitadores, e
as notações em linha do texto (β_pré(h), v_pós(h), bⱼ(h), T₀ ...) também.

O conteúdo matemático é o mesmo; só muda a forma. O texto antigo fica marcado como excluído e a
equação nova como inserida, autor "Claude"; rejeitar devolve o original.

Uso: python aplicar_equacoes_word_revisao_docx.py [entrada saida]
Backup: historico/3_versoes_e_backups/backups_do_TCC/_backup_revisao_pre_equacoes.docx
"""
import copy, sys
from pathlib import Path
import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revisao_track import Revisor
from omml import Eq

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 2 else RAIZ / "TCC_PEDRO_MINELLA_revisao_com_alteracoes.docx"
SAIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else ENTRADA
MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

doc = docx.Document(ENTRADA)
R = Revisor(doc)
corpo = doc.element.body
E = Eq(corpo)                     # equações sem marca interna: a marca é o w:ins que as envolve


# ------------------------------------------------------------------ atalhos de notação
def P_(x): return E.d(x)
def C_(x): return E.d(x, "[", "]")
def CH_(x): return E.d(x, "{", "}")
def RT(s): return E.r(s, reto=True)
def DL(): return RT("Δ")
def TX(s): return E.r(s, palavra=True)
def ln(arg): return E.func("ln", arg)
def beta(rot): return E.sub("β", RT(rot))
def beta_h(rot): return [beta(rot), P_("h")]
def beta_est(rot): return [E.subsup("β", RT(rot), "*"), P_("h")]
def choque(arg): return [TX("choque"), P_(arg)]
def acum(): return ["100·", C_([ln(["P", P_("t+h")]), "−", ln(["P", P_("t−1")])])]
def ind(rel): return ["1", CH_("t" + rel + "τ")]
def soma_j(corpo_): return E.soma("j", None, corpo_)
def dif_est(): return C_([beta_est("pós"), "−", beta_est("pré")])


# ------------------------------------------------------------------ localização pelo texto visível
def visivel(p):
    out = []
    for e in p.iter(qn("w:t")):
        anc = {a.tag for a in e.iterancestors()}
        if qn("w:del") in anc or MATH + "oMath" in anc:
            continue
        out.append(e.text or "")
    return "".join(out)


def achar(inicio):
    achados = [p for p in corpo.iter(qn("w:p")) if visivel(p).startswith(inicio)]
    assert len(achados) == 1, (inicio[:60], len(achados))
    return achados[0]


def trocar(p, trecho, *partes):
    R.substituir_por(p, trecho, [E.omath(*partes)])


def trocar_em_insercao(p, trecho, *partes):
    """Para texto que eu mesmo inseri nesta revisão: a equação entra dentro da mesma inserção."""
    alvos = [r for ins in p.iter(qn("w:ins")) if ins.get(qn("w:author")) == R.autor
             for r in ins.findall(qn("w:r")) if r.find(qn("w:t")) is not None and trecho in (r.find(qn("w:t")).text or "")]
    assert len(alvos) == 1 and alvos[0].find(qn("w:t")).text.count(trecho) == 1, (trecho, len(alvos))
    r = alvos[0]; t = r.find(qn("w:t"))
    esq, dir_ = t.text.split(trecho)
    t.text = esq
    resto = copy.deepcopy(r); resto.find(qn("w:t")).text = dir_
    om = E.omath(*partes)
    r.addnext(om); om.addnext(resto)


def run_texto(modelo, texto=None, tab=False, quebra=False):
    rr = corpo.makeelement(qn("w:r"), {})
    rpr = modelo.find(qn("w:rPr")) if modelo is not None else None
    if rpr is not None:
        rr.append(copy.deepcopy(rpr))
    if tab:
        rr.append(corpo.makeelement(qn("w:tab"), {}))
    if quebra:
        rr.append(corpo.makeelement(qn("w:br"), {}))
    if texto is not None:
        t = corpo.makeelement(qn("w:t"), {}); t.text = texto
        rr.append(t)
    return rr


# ------------------------------------------------------------------ (1)
p = achar("choque(t) = 100 · [ln B(t)")
trocar(p, "choque(t) = 100 · [ln B(t) − ln B(t−1)] / σ,   σ = 10,19",
       choque("t"), "=", E.frac(["100·", C_([ln(["B", P_("t")]), "−", ln(["B", P_("t−1")])])], "σ", tipo="lin"),
       ",", TX("    "), "σ=10,19")

# ------------------------------------------------------------------ (2)
p = achar("100·[ln P(t+h) − ln P(t−1)] = α + β(h)·choque(t)+ Σ(i=1..3)")
trocar(p, "100·[ln P(t+h) − ln P(t−1)] = α + β(h)·choque(t)",
       *acum(), "=α+β", P_("h"), "·", *choque("t"))
trocar(p, "+ Σ(i=1..3) [ γᵢ·Δln P(t−i) + φᵢ·choque(t−i) + θᵢ′·X(t−i) ]",
       "+", E.soma("i=1", "3", C_([E.sub("γ", "i"), "·", DL(), ln(["P", P_("t−i")]), "+", E.sub("φ", "i"), "·",
                                  *choque("t−i"), "+", E.subsup("θ", "i", "′"), "·X", P_("t−i")])))
trocar(p, "+ sazonais mensais + indicadoras de pandemia + u(t,h)",
       "+", TX("sazonais mensais"), "+", TX("indicadoras de pandemia"), "+u", P_("t,h"))

# ------------------------------------------------------------------ (3)
p = achar("100·[ln P(t+h) − ln P(t−1)] = α + β_pré(h)")
trocar(p, "100·[ln P(t+h) − ln P(t−1)] = α + β_pré(h)·choque(t)·1{t < τ}",
       *acum(), "=α+", *beta_h("pré"), "·", *choque("t"), "·", *ind("<"))
trocar(p, "+ β_pós(h)·choque(t)·1{t ≥ τ} + δ(h)·1{t ≥ τ} + controles + u(t,h)",
       "+", *beta_h("pós"), "·", *choque("t"), "·", *ind("≥"), "+δ", P_("h"), "·", *ind("≥"),
       "+", TX("controles"), "+u", P_("t,h"))

# ------------------------------------------------------------------ (4)
p = achar("[β*_pós(h) − β*_pré(h)] = [β_pós(h)")
trocar(p, "[β*_pós(h) − β*_pré(h)] = [β_pós(h) − β_pré(h)] + [v_pós(h) − v_pré(h)]",
       dif_est(), "=", C_([*beta_h("pós"), "−", *beta_h("pré")]), "+",
       C_([E.sub("v", RT("pós")), P_("h"), "−", E.sub("v", RT("pré")), P_("h")]))

# ------------------------------------------------------------------ (5)
p = achar("W(h) = [β*_pós(h)")
trocar(p, "W(h) = [β*_pós(h) − β*_pré(h)]² / V*[β*_pós(h) − β*_pré(h)]  ~  χ²(1)",
       "W", P_("h"), "=", E.frac(E.sup(dif_est(), "2"), P_([E.sup("V", "*"), dif_est()]), tipo="lin"),
       "∼", E.sup("χ", "2"), P_("1"))
trocar(p, "H₀: β_pré(h) = β_pós(h)", E.sub("H", "0"), ":", *beta_h("pré"), "=", *beta_h("pós"))
trocar(p, "V*", E.sup("V", "*"))

# ------------------------------------------------------------------ (6)
p = achar("τ* = arg max")
trocar(p, "τ* = arg max (τ ∈ C) Σ(k) W_k(τ),   C = { τ : 0,15·T ≤ τ ≤ 0,85·T }",
       E.sup("τ", "*"), "=", E.func_lim("arg max", "τ∈C", E.soma("k", None, [E.sub("W", "k"), P_("τ")])),
       ",", TX("    "), "C=", CH_("τ:0,15·T≤τ≤0,85·T"))

# ------------------------------------------------------------------ (7), (8), (9)
p = achar("100·[ln P(t+h) − ln P(t−1)] = α + β(h)·choque(t) + controles")
trocar(p, "100·[ln P(t+h) − ln P(t−1)] = α + β(h)·choque(t) + controles + u(t,h)",
       *acum(), "=α+β", P_("h"), "·", *choque("t"), "+", TX("controles"), "+u", P_("t,h"))
trocar(p, "100·[ln P(t+h) − ln P(t−1)] = α + β_dir(h)·choque(t)",
       *acum(), "=α+", *beta_h("dir"), "·", *choque("t"))
linha8 = [x for x in [visivel(p)] if "+ Σ(j) δⱼ(h)·Fⱼ(t,h) + controles + " in x][0]
fim8 = linha8.split("+ Σ(j) δⱼ(h)·Fⱼ(t,h) + controles + ")[1].split("(8)")[0]     # "ũ(t,h)" como estiver gravado
trocar(p, "+ Σ(j) δⱼ(h)·Fⱼ(t,h) + controles + " + fim8,
       "+", soma_j([E.sub("δ", "j"), P_("h"), "·", E.sub("F", "j"), P_("t,h")]), "+", TX("controles"), "+",
       E.acento("u"), P_("t,h"))
trocar(p, "canal(h) = β(h) − β_dir(h) = Σ(j) δⱼ(h)·bⱼ(h)",
       TX("canal"), P_("h"), "=β", P_("h"), "−", *beta_h("dir"), "=",
       soma_j([E.sub("δ", "j"), P_("h"), "·", E.sub("b", "j"), P_("h")]))

# ------------------------------------------------------------------ (10), (11), (12)
p = achar(":  β₁ª(h)") if any(visivel(x).startswith(":  β₁ª(h)") for x in corpo.iter(qn("w:p"))) \
    else achar("Efeitos de Primeira Ordem:")
trocar(p, "β₁ª(h) = Σ(j) wⱼ · bⱼ(h)",
       beta("1ª"), P_("h"), "=", soma_j([E.sub("w", "j"), "·", E.sub("b", "j"), P_("h")]))
trocar(p, "β₂ª(h) = β(h) − Σ(j) wⱼ · bⱼ(h)",
       beta("2ª"), P_("h"), "=β", P_("h"), "−", soma_j([E.sub("w", "j"), "·", E.sub("b", "j"), P_("h")]))
trocar(p, "Δln IPCA^ex(t) = Δln IPCA(t) − Σ(j) wⱼ · Δln Pⱼ(t)",
       DL(), ln([E.sup(RT("IPCA"), RT("ex")), P_("t")]), "=", DL(), ln([RT("IPCA"), P_("t")]), "−",
       soma_j([E.sub("w", "j"), "·", DL(), ln([E.sub("P", "j"), P_("t")])]))

# ------------------------------------------------------------------ (13), (14): refeitas
p = [x for x in corpo.iter(qn("w:p")) if any(
    "h=0..H" in "".join(t.text or "" for t in om.iter(MATH + "t")) for om in x.findall(MATH + "oMath"))]
assert len(p) == 1
p = p[0]
antiga = p.find(MATH + "oMath")
modelo = p.find(qn("w:r"))
d = R.apagar_elemento(antiga)
ins = R._marca("w:ins")
ins.append(E.omath(E.sub("c", "r"), P_("s"), "=",
                   E.soma("h=0", "H", [E.sub("b", "r"), P_("h"), "·", *choque("s−h")]),
                   ",", TX("  com  "), "s−h≥", E.sub("T", "0")))
ins.append(run_texto(modelo, tab=True)); ins.append(run_texto(modelo, "(13)"))
ins.append(run_texto(modelo, quebra=True)); ins.append(run_texto(modelo, tab=True))
ins.append(E.omath(E.sub("C", "r"), "=", E.soma(["s=", E.sub("T", "0")], E.sub("T", "1"), [E.sub("c", "r"), P_("s")]),
                   ",", TX("    "), DL(), "=", E.sub("C", RT("pós")), "−", E.sub("C", RT("pré"))))
ins.append(run_texto(modelo, tab=True)); ins.append(run_texto(modelo, "(14)"))
d.addnext(ins)

# ------------------------------------------------------------------ notações em linha no texto
p = achar("A variável dependente é a resposta acumulada")
trocar(p, "100·[ln P(t+h) − ln P(t−1)]", *acum())

p = achar("A indicadora 1{t ≥ τ}")
trocar(p, "1{t ≥ τ}", *ind("≥"))
trocar(p, "β_pré(h)", *beta_h("pré"))
trocar(p, "β_pós(h)", *beta_h("pós"))

p = achar("Convém explicitar o que esta especificação identifica")
trocar_em_insercao(p, "β_pré(h)", *beta_h("pré"))
trocar_em_insercao(p, "β_pós(h)", *beta_h("pós"))

p = achar("O asterisco denota")
trocar_em_insercao(p, "v_pré(h)", E.sub("v", RT("pré")), P_("h"))
trocar_em_insercao(p, "v_pós(h)", E.sub("v", RT("pós")), P_("h"))

p = achar("Em (8), Fⱼ(t,h)")
trocar(p, "Fⱼ(t,h)", E.sub("F", "j"), P_("t,h"))
trocar(p, "bⱼ(h)", E.sub("b", "j"), P_("h"))

p = achar("O peso wⱼ")
trocar(p, "wⱼ", E.sub("w", "j"))
trocar(p, "bⱼ(h)", E.sub("b", "j"), P_("h"))

p = achar("r indica o regime; T₀ e T₁")
trocar(p, "s−h ≥ T₀", "s−h≥", E.sub("T", "0"))
trocar(p, "T₀", E.sub("T", "0"))
trocar(p, "T₁", E.sub("T", "1"))
trocar(p, "b_r(h)", E.sub("b", "r"), P_("h"))

p = achar("As séries que o modelo transforma em logaritmo")
trocar(p, "100·[ln P(t) − ln P(jan/2003)]",
       "100·", C_([ln(["P", P_("t")]), "−", ln(["P", P_(TX("jan/2003"))])]))

doc.save(SAIDA)
print("salvo", SAIDA)
