# -*- coding: utf-8 -*-
"""(Banca 2) Valor critico por reamostragem para a busca de SEGUNDA quebra.

A secao 4.7.10 reporta que a busca sequencial nao encontra segunda quebra, mas o
faz comparando a estatistica observada com a da quebra principal. Aqui o valor
critico e obtido pelo mesmo procedimento da secao 3.6 e do script onda3_3:
bootstrap selvagem em blocos com regressores fixos (HANSEN, 2000), pesos de
Rademacher e residuos corrigidos por alavancagem (DAVIDSON e FLACHAIRE, 2008).

HIPOTESE NULA. Nao ha segunda quebra dentro do segmento; vale o modelo de DOIS
regimes, com a quebra principal em mar/2017 ja imposta. O DGP das replicas e o
ajuste desse modelo.

CORRECAO PARA SELECAO. Cada replica refaz a busca inteira sobre todas as datas
candidatas do segmento e a estatistica e avaliada no argmax DELA, de modo que o
valor critico e o quantil 95 da distribuicao do MAXIMO, e nao de um ponto.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import lp_base_nova as L
import onda3_3_data_comum as O33

SAIDA = L.SAIDA / "banca2"
SAIDA.mkdir(parents=True, exist_ok=True)

B, BLOCO, SEMENTE = 299, 12, 20260907
MIN = 36                  # comprimento minimo de subsegmento, em meses
H = 0                     # horizonte de impacto, como em onda3_3
PRINCIPAIS = ["gasolina_consumidor", "diesel", "ipca_geral", "ipca_transportes"]


def perfil_segundo(am, W, cand, etas_df, tau, dentro_seg):
    """F(c) sob H0 de DOIS regimes, para cada candidato c dentro do segmento.

    X0  : z*(1-pos), z*pos, pos, W                    -> dois regimes (nula)
    X1  : acrescenta a particao do segmento em c      -> tres regimes (alternativa)
    """
    y = np.asarray(am["__y"], float)
    z = np.asarray(am["choque_petroleo"], float)
    datas = pd.DatetimeIndex(am["Data"])
    n = len(y)
    pos = np.asarray(datas >= tau, float)
    seg = np.asarray([d in dentro_seg for d in datas], float)

    X0 = np.column_stack([z * (1 - pos), z * pos, pos, W])
    P0 = np.linalg.pinv(X0)
    ajus = X0 @ (P0 @ y)
    lev = np.clip(np.einsum("ij,ji->i", X0, P0), 0.0, 0.999)
    res = (y - ajus) / np.sqrt(1.0 - lev)
    E = etas_df.reindex(datas).to_numpy()
    Y = ajus[:, None] + res[:, None] * E

    s0o = float(((y - ajus) ** 2).sum())
    R0 = Y - X0 @ (P0 @ Y)
    s0b = (R0 * R0).sum(axis=0)

    fo = np.empty(len(cand))
    fb = np.empty((E.shape[1], len(cand)))
    for j, c in enumerate(cand):
        dep = seg * np.asarray(datas >= c, float)      # 2a parte do segmento
        ant = seg * np.asarray(datas < c, float)       # 1a parte do segmento
        X1 = np.column_stack([z * ant, z * (seg - ant), z * (1 - seg), dep, pos, W])
        P1 = np.linalg.pinv(X1)
        k = X1.shape[1]
        r = y - X1 @ (P1 @ y)
        s1o = float(r @ r)
        fo[j] = ((s0o - s1o) / 1) / (s1o / (n - k))
        R1 = Y - X1 @ (P1 @ Y)
        s1b = (R1 * R1).sum(axis=0)
        fb[:, j] = ((s0b - s1b) / 1) / (s1b / (n - k))
    return fo, fb


def rodar():
    b, dum, pand = L.carregar_pandemia("corrigida")
    tau = pd.Timestamp(L.DATA_CORTE_FINAL)
    datas_ref = pd.DatetimeIndex(sorted(b["Data"].unique()))
    rng = np.random.default_rng(SEMENTE)
    nb = int(np.ceil(len(datas_ref) / BLOCO))
    etas_df = pd.DataFrame(
        np.column_stack([np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:len(datas_ref)]
                         for _ in range(B)]), index=datas_ref)

    ams = {nome: O33.montar(nome, H, b, dum, pand) for nome in PRINCIPAIS}
    comuns = sorted(set.intersection(*[set(am["Data"]) for (am, _, _) in ams.values()]))

    linhas, perfis = [], []
    for seg, sel in [("anterior", [d for d in comuns if d < tau]),
                     ("posterior", [d for d in comuns if d >= tau])]:
        dentro = set(sel)
        cand = sel[MIN:len(sel) - MIN]
        print(f"\n  segmento {seg}: {len(sel)} meses, {len(cand)} candidatas, {B} réplicas")

        soma = np.zeros(len(cand))
        somab = np.zeros((B, len(cand)))
        for nome in PRINCIPAIS:
            am, W, _ = ams[nome]
            fo, fb = perfil_segundo(am, W, cand, etas_df, tau, dentro)
            soma += fo
            somab += fb
            print(f"    perfil {nome}")

        j = int(np.argmax(soma))
        estat = float(soma[j])
        maxb = somab.max(axis=1)
        crit = float(np.quantile(maxb, 0.95))
        p = float((maxb >= estat).mean())
        linhas.append({"segmento": seg, "data_argmax": cand[j], "estatistica": estat,
                       "critico_95": crit, "p_bootstrap": p,
                       "n_candidatas": len(cand), "replicas": B})
        for d_, v in zip(cand, soma):
            perfis.append({"segmento": seg, "data": d_, "soma_F": v, "critico_95": crit})
        print(f"    argmax {cand[j]:%Y-%m}  F = {estat:.2f}  crítico 5% = {crit:.2f}  "
              f"p = {p:.3f}")

    t = pd.DataFrame(linhas)
    with pd.ExcelWriter(SAIDA / "bai_perron_bootstrap.xlsx") as w:
        t.to_excel(w, sheet_name="resultado", index=False)
        pd.DataFrame(perfis).to_excel(w, sheet_name="perfil", index=False)

    print("\n" + "=" * 70)
    print("SEGUNDA QUEBRA: valor crítico por reamostragem, ajustado por seleção")
    print("=" * 70)
    print(t.to_string(index=False))
    print("\nNota: p-valor e critico do MAXIMO sobre todas as candidatas do segmento,")
    print("com cada replica refazendo a busca inteira sob a hipotese nula de dois regimes.")
    return t


if __name__ == "__main__":
    rodar()
