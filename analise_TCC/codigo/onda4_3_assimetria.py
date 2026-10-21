# -*- coding: utf-8 -*-
"""
ONDA 4.3 -- ASSIMETRIA: ALTA E BAIXA DO PETROLEO REPASSAM IGUAL?

Decompoe o choque em parte positiva e parte negativa (MORK, 1989):

    ch_pos(t) = max(choque(t), 0)      >= 0
    ch_neg(t) = min(choque(t), 0)      <= 0
    ch_pos + ch_neg = choque           (identidade, conferida por assertiva)

e estima, por regime,

    y_acum(t,h) = a + b_pos(h) ch_pos + b_neg(h) ch_neg + ... + u

A simetria e H0: b_pos = b_neg. O padrao "foguetes e penas" (BACON, 1991;
BORENSTEIN, CAMERON e GILBERT, 1997) e b_pos > b_neg: sobe rapido, desce devagar.

POR QUE ISSO IMPORTA AQUI, E NAO E SO UM EXTRA
  (1) E a hipotese alternativa ao resultado de 2023. Depois de jun/2023 nao houve
      choque grande, e se o repasse for nao linear no tamanho ou no sinal, o
      coeficiente menor do 3o regime pode nao ser efeito de politica. O teste de
      choque pequeno (onda4_2) fecha o lado do TAMANHO; este fecha o do SINAL.
  (2) E previsao da propria hipotese institucional. Preco administrado tem
      discricao, e discricao produz assimetria: segura a alta, demora na baixa.
      Regra mecanica de paridade nao tem discricao, logo o repasse sob PPI
      deveria ser MAIS SIMETRICO que no periodo anterior. Se os dados mostrarem
      isso, e evidencia independente de que a regra de fato passou a valer.

Estrutura em dois e em tres regimes, com a mesma inferencia do modelo principal:
HAC de Newey-West (Bartlett, banda h+1, correcao T/(T-K)) e teste conjunto em
h = 0..6 com critico por wild block bootstrap.

Uso: python onda4_3_assimetria.py
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
from onda2_bandas_placebo import (ajustar, replicas, hac_conjunta_horizontes,
                                  _amostra)
import onda4_1_tres_regimes as O41

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
H_CONJUNTO = 6
B, BLOCO, SEMENTE = 999, 12, 20260831
PRINCIPAIS = ["ipca_geral", "ipca_transportes", "gasolina_consumidor", "diesel",
              "gasolina_refinaria"]


def montar(tres_regimes):
    """Base com ch_pos e ch_neg interagidos com os regimes."""
    b, dum, pand = O41.montar(O41.CORTE_FIM)
    b["ch_pos"] = b["choque_petroleo"].clip(lower=0.0)
    b["ch_neg"] = b["choque_petroleo"].clip(upper=0.0)
    dv = float(np.abs(b.ch_pos + b.ch_neg - b.choque_petroleo).max())
    assert dv < 1e-12, f"decomposicao por sinal falhou: {dv}"

    d_pre = 1.0 - b["d_ppi"] - b["d_pos"]
    if tres_regimes:
        blocos = [("pre", d_pre), ("ppi", b["d_ppi"]), ("pos", b["d_pos"])]
        niveis = ["d_ppi", "d_pos"]
    else:
        blocos = [("pre", d_pre), ("pos", b["d_ppi"] + b["d_pos"])]
        b["d_reg"] = b["d_ppi"] + b["d_pos"]
        niveis = ["d_reg"]
    cab = []
    for rot, ind in blocos:
        for s in ["pos", "neg"]:
            c = f"ch_{s}_{rot}"
            b[c] = b[f"ch_{s}"] * ind
            cab.append(c)
    return b, dum, pand, cab, niveis, [r for r, _ in blocos]


def contagem(b, regimes, tres):
    d_pre = 1.0 - b["d_ppi"] - b["d_pos"]
    ind = ({"pre": d_pre, "ppi": b["d_ppi"], "pos": b["d_pos"]} if tres
           else {"pre": d_pre, "pos": b["d_ppi"] + b["d_pos"]})
    lin = []
    for r in regimes:
        m = ind[r] == 1
        c = b.loc[m, "choque_petroleo"].dropna()
        lin.append({"regime": r, "meses": int(m.sum()),
                    "altas": int((c > 0).sum()), "baixas": int((c < 0).sum()),
                    "media_alta": float(c[c > 0].mean()),
                    "media_baixa": float(c[c < 0].mean())})
    return pd.DataFrame(lin)


def rodar(tres_regimes):
    rng = np.random.default_rng(SEMENTE)
    b, dum, pand, cab, niveis, regimes = montar(tres_regimes)
    datas_ref = pd.DatetimeIndex(sorted(b["Data"].unique()))
    n = len(datas_ref); nb = int(np.ceil(n / BLOCO))
    etas = np.column_stack([np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:n]
                            for _ in range(B)])
    linhas, conj = [], []
    for nome in L.VARIAVEIS:
        d = b.copy()
        for lag in range(1, L.LAGS + 1):
            d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
        din = []
        for lag in range(1, L.LAGS + 1):
            din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                   [f"{c}_lag{lag}" for c in L.CONTROLES]
        for h in range(0, L.H_MAX + 1):
            d["__y"] = L.y_acum(d, nome, h)
            regs = cab + niveis + din + dum + pand(d, h)
            am = _amostra(d, regs)
            for r in regimes:
                gp = ajustar(am, regs, h, f"ch_pos_{r}")
                gn = ajustar(am, regs, h, f"ch_neg_{r}")
                gd = ajustar(am, regs, h, (f"ch_neg_{r}", f"ch_pos_{r}"))
                z = gd["valor"] / gd["ep"]
                linhas.append({
                    "variavel": nome, "regime": r, "h": h,
                    "b_alta": gp["valor"], "ep_alta": gp["ep"],
                    "b_baixa": gn["valor"], "ep_baixa": gn["ep"],
                    "diferenca": gd["valor"], "ep_diferenca": gd["ep"],
                    "p_simetria": float(2 * (1 - stats.norm.cdf(abs(z)))),
                    "observacoes": gp["nobs"]})
        # ---- teste conjunto de simetria, h = 0..6
        for r in regimes:
            mont = []
            for h in range(0, H_CONJUNTO + 1):
                d["__y"] = L.y_acum(d, nome, h)
                regs = cab + niveis + din + dum + pand(d, h)
                mont.append((h, regs, _amostra(d, regs)))
            com = set.intersection(*[set(am["Data"]) for _, _, am in mont])
            gs = [ajustar(am[am["Data"].isin(com)], regs, h,
                          (f"ch_neg_{r}", f"ch_pos_{r}")) for h, regs, am in mont]
            V = hac_conjunta_horizontes(gs, H_CONJUNTO + 1)
            Vi = np.linalg.pinv(V)
            bh = np.array([g["valor"] for g in gs])
            W = float(bh @ Vi @ bh)
            Bm = np.column_stack([replicas(g, etas, datas_ref)[0] for g in gs])
            Wb = np.einsum("ij,jk,ik->i", Bm - bh, Vi, Bm - bh)
            conj.append({"variavel": nome, "regime": r,
                         "teste": "simetria h=0..6", "estatistica": W,
                         "p_bootstrap": float((Wb >= W).mean()),
                         "critico_95": float(np.quantile(Wb, 0.95))})
        print(f"  {nome}")
    return (pd.DataFrame(linhas), pd.DataFrame(conj),
            contagem(b, regimes, tres_regimes))


def figura():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.read_excel(SAIDA / "onda4_3_assimetria.xlsx", sheet_name="tres_regimes")
    rot = {"pre": "pré-PPI", "ppi": "PPI", "pos": "pós-PPI"}
    cor = {"pre": "#5F6549", "ppi": "#9E3D25", "pos": "#0B5A56"}
    for nome in ["ipca_geral", "ipca_transportes", "gasolina_consumidor"]:
        s = t[t.variavel == nome]
        fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
        for j, (col, ep, tit) in enumerate(
                [("b_alta", "ep_alta", "resposta a ALTA do petróleo"),
                 ("b_baixa", "ep_baixa", "resposta a BAIXA do petróleo")]):
            for r in ["pre", "ppi", "pos"]:
                q = s[s.regime == r].sort_values("h")
                ax[j].plot(q.h, q[col], color=cor[r], lw=2, marker="o", ms=3,
                           label=rot[r])
                ax[j].fill_between(q.h, q[col] - L.Z_90 * q[ep],
                                   q[col] + L.Z_90 * q[ep],
                                   color=cor[r], alpha=0.12, lw=0)
            ax[j].axhline(0, color="0.35", lw=0.9)
            ax[j].set_title(tit, fontsize=9.5)
            ax[j].set_xlabel("meses após o choque", fontsize=8.5)
            ax[j].set_xticks(range(0, 13, 2)); ax[j].tick_params(labelsize=7.5)
        ax[0].set_ylabel("pontos percentuais", fontsize=8.5)
        ax[0].legend(fontsize=7.5)
        fig.suptitle(f"{nome}: assimetria do repasse por regime\n"
                     f"faixas = banda pontual de 90%", fontsize=10.5)
        fig.tight_layout()
        arq = FIGS / f"assimetria_{nome}.png"
        fig.savefig(arq, dpi=150); plt.close(fig)
        print(f"  {arq.name}")


def main():
    print(__doc__)
    print("=== dois regimes (modelo principal) ===")
    t2, c2, n2 = rodar(tres_regimes=False)
    print("\n=== tres regimes ===")
    t3, c3, n3 = rodar(tres_regimes=True)

    with pd.ExcelWriter(SAIDA / "onda4_3_assimetria.xlsx") as w:
        t2.to_excel(w, sheet_name="dois_regimes", index=False)
        c2.to_excel(w, sheet_name="conjunto_dois", index=False)
        t3.to_excel(w, sheet_name="tres_regimes", index=False)
        c3.to_excel(w, sheet_name="conjunto_tres", index=False)
        n3.to_excel(w, sheet_name="contagem", index=False)

    print("\n=== altas e baixas por regime ===")
    print(n3.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    print("\n=== assimetria em h=0 e h=1, tres regimes ===")
    v = t3[(t3.h.isin([0, 1])) & (t3.variavel.isin(PRINCIPAIS))]
    print(v[["variavel", "regime", "h", "b_alta", "b_baixa", "diferenca",
             "p_simetria"]].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n=== teste conjunto de simetria h=0..6, tres regimes ===")
    print(c3[c3.variavel.isin(PRINCIPAIS)].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n=== o mesmo, no modelo principal de dois regimes ===")
    print(c2[c2.variavel.isin(PRINCIPAIS)].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))
    figura()


if __name__ == "__main__":
    main()
