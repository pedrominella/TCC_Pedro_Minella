# -*- coding: utf-8 -*-
"""
ONDA 4.1 -- O EFEITO CAUSAL DA PPI, COM INICIO E FIM

A politica de Preco de Paridade de Importacao foi instituida em 2016, teve o
repasse detectado a partir de mar/2017 (quebra estimada em onda3_3) e foi
ENCERRADA em 16 de maio de 2023, quando a Petrobras anunciou nova estrategia
comercial. A especificacao de dois regimes usada ate aqui junta o periodo PPI
com o periodo posterior ao fim da politica, e portanto SUBESTIMA o efeito.

Este script estima tres regimes:

    pre   : t <  mar/2017                    (preco administrado)
    ppi   : mar/2017 <= t < jun/2023         (paridade de importacao)
    pos   : t >= jun/2023                    (fim da paridade)

    100*[ln P(t+h) - ln P(t-1)]
        = a + b_pre(h)*choque*1{pre} + b_ppi(h)*choque*1{ppi}
            + b_pos(h)*choque*1{pos} + d1(h)*1{ppi} + d2(h)*1{pos}
        + soma_{i=1..3}[g_i dlnP(t-i) + f_i choque(t-i) + O_i X(t-i)]
        + 11 indicadoras mensais + indicadoras de pandemia da janela de h
        + u(t,h)

POR QUE ISSO IDENTIFICA MELHOR QUE UMA QUEBRA SO
  Uma quebra unica e compativel com "algo mudou em 2017". Duas quebras em
  sentidos OPOSTOS, cada uma casada com uma decisao de politica datada, e um
  desenho liga-desliga: o repasse deve subir em 2017 e cair em 2023. Uma causa
  concorrente teria de explicar as duas viradas, nas duas datas certas.

POR QUE A DATA DE 2023 E IMPOSTA, E NAO ESTIMADA
  O anuncio e publico e datado (16/05/2023), logo a data e exogena ao ajuste do
  modelo e NAO ha problema de inferencia pos-selecao: o p-valor convencional
  vale. Em 2017 a data foi estimada porque a implementacao foi gradual, e la o
  p-valor precisou de ajuste por selecao (onda3_3). Usa-se jun/2023, primeiro mes
  cheio sob a nova politica, com mai/2023 como robustez.

RESSALVA DE POTENCIA, REPORTADA E NAO ESCONDIDA
  O terceiro regime tem poucos meses, e o numero cai com h porque a dependente
  precisa de t+h. O script imprime o n efetivo de CADA regime em CADA horizonte.
  Coeficiente do regime pos com n pequeno e impreciso por construcao: leia o
  erro-padrao, nao so o ponto.

Inferencia identica a do modelo principal: HAC de Newey-West (Bartlett, banda
h+1, correcao T/(T-K)), banda uniforme sup-t plug-in, e wild block bootstrap
apenas no critico dos testes conjuntos.

Uso: python onda4_1_tres_regimes.py
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
from onda2_bandas_placebo import (ajustar, replicas, sup_t_plugin,
                                  hac_conjunta_horizontes, _amostra)

PASTA = Path(__file__).resolve().parent.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"

CORTE_PPI = L.DATA_CORTE_FINAL          # 2017-03-01, estimado em onda3_3
CORTE_FIM = "2023-06-01"                # 1o mes cheio apos o anuncio de 16/05/2023
CORTE_FIM_ALT = "2023-05-01"            # robustez: mes do anuncio
H_CONJUNTO = 6
B, BLOCO, SEMENTE = 999, 12, 20260831

REGIMES = ["c_pre", "c_ppi", "c_pos"]
ROTULO = {"c_pre": "pre-PPI", "c_ppi": "PPI", "c_pos": "pos-PPI"}
CORES = {"c_pre": "#5F6549", "c_ppi": "#9E3D25", "c_pos": "#0B5A56"}
PRINCIPAIS = ["ipca_geral", "ipca_transportes", "gasolina_consumidor", "diesel",
              "gasolina_refinaria"]


def montar(corte_fim):
    """Base com as tres interacoes e as duas indicadoras de nivel."""
    b, dum, pand = L.carregar_pandemia("corrigida")
    r_ppi = L.regime(b.Data, CORTE_PPI)          # 1{t >= mar/2017}
    r_fim = L.regime(b.Data, corte_fim)          # 1{t >= corte_fim}
    b["d_ppi"] = r_ppi - r_fim                   # 1{mar/2017 <= t < corte_fim}
    b["d_pos"] = r_fim
    d_pre = 1.0 - r_ppi
    b["c_pre"] = b["choque_petroleo"] * d_pre
    b["c_ppi"] = b["choque_petroleo"] * b["d_ppi"]
    b["c_pos"] = b["choque_petroleo"] * b["d_pos"]
    # as tres interacoes particionam o choque: soma tem de devolver o original
    desvio = float(np.abs(b["c_pre"] + b["c_ppi"] + b["c_pos"]
                          - b["choque_petroleo"]).max())
    assert desvio < 1e-12, f"particao do choque falhou: {desvio}"
    assert set(np.unique(b["d_ppi"])) <= {0.0, 1.0}, "d_ppi nao e indicadora"
    return b, dum, pand


def rodar(corte_fim=CORTE_FIM, salvar=True, rotulo_corte="jun/2023"):
    rng = np.random.default_rng(SEMENTE)
    b, dum, pand = montar(corte_fim)
    datas_ref = pd.DatetimeIndex(sorted(b["Data"].unique()))
    n = len(datas_ref); nb = int(np.ceil(n / BLOCO))
    etas = np.column_stack([np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:n]
                            for _ in range(B)])
    print(f"corte do fim da PPI: {rotulo_corte}   "
          f"meses: pre={int((b.d_ppi + b.d_pos == 0).sum())} "
          f"ppi={int(b.d_ppi.sum())} pos={int(b.d_pos.sum())}")

    linhas, waldh, conjuntos = [], [], []
    for nome in L.VARIAVEIS:
        d = b.copy()
        for lag in range(1, L.LAGS + 1):
            d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
        din = []
        for lag in range(1, L.LAGS + 1):
            din += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                   [f"{c}_lag{lag}" for c in L.CONTROLES]
        cab = REGIMES + ["d_ppi", "d_pos"]

        montado, guarda = [], {}
        for h in range(0, L.H_MAX + 1):
            d["__y"] = L.y_acum(d, nome, h)
            regs = cab + din + dum + pand(d, h)
            am = _amostra(d, regs)
            montado.append((h, regs, am))
            # n efetivo de cada regime NESTE horizonte
            nreg = {"pre": int((am["d_ppi"] + am["d_pos"] == 0).sum()),
                    "ppi": int(am["d_ppi"].sum()), "pos": int(am["d_pos"].sum())}
            g = {r: ajustar(am, regs, h, r) for r in REGIMES}
            guarda[h] = (g, am, regs, nreg)
            for r in REGIMES:
                linhas.append({"variavel": nome, "regime": ROTULO[r], "h": h,
                               "coeficiente": g[r]["valor"], "erro_padrao": g[r]["ep"],
                               "observacoes": g[r]["nobs"],
                               "n_pre": nreg["pre"], "n_ppi": nreg["ppi"],
                               "n_pos": nreg["pos"]})
            # ---- Wald por horizonte, entre pares de regimes
            for a1, a2, rot in [("c_pre", "c_ppi", "pre = ppi"),
                                ("c_ppi", "c_pos", "ppi = pos"),
                                ("c_pre", "c_pos", "pre = pos")]:
                gd = ajustar(am, regs, h, (a1, a2))
                z = gd["valor"] / gd["ep"]
                waldh.append({"variavel": nome, "h": h, "teste": rot,
                              "diferenca": gd["valor"], "erro_padrao": gd["ep"],
                              "p_valor": float(2 * (1 - stats.norm.cdf(abs(z)))),
                              "n_pos": nreg["pos"]})

        # ---- banda uniforme sup-t, uma para cada trajetoria de regime
        comuns = set.intersection(*[set(am["Data"]) for _, _, am in montado])
        cunif = {}
        for r in REGIMES:
            gs_c = [ajustar(am[am["Data"].isin(comuns)], regs, h, r)
                    for h, regs, am in montado]
            cunif[r] = sup_t_plugin(hac_conjunta_horizontes(gs_c, L.H_MAX + 1))
        for reg in linhas:
            if reg["variavel"] != nome:
                continue
            r = [k for k, v in ROTULO.items() if v == reg["regime"]][0]
            cf, ep = reg["coeficiente"], reg["erro_padrao"]
            t_obs = abs(cf / ep)
            reg["pontual_inferior"] = cf - L.Z_90 * ep
            reg["pontual_superior"] = cf + L.Z_90 * ep
            reg["p_valor"] = float(2 * (1 - stats.norm.cdf(t_obs)))
            reg["signif_pontual_10"] = bool(t_obs > L.Z_90)
            reg["critico_uniforme"] = cunif[r]
            reg["uniforme_inferior"] = cf - cunif[r] * ep
            reg["uniforme_superior"] = cf + cunif[r] * ep
            reg["signif_uniforme_10"] = bool(t_obs > cunif[r])

        # ---- testes conjuntos h = 0..6, critico por wild block bootstrap
        for a1, a2, rot in [("c_pre", "c_ppi", "ligou em 2017 (pre = ppi, h=0..6)"),
                            ("c_ppi", "c_pos", "desligou em 2023 (ppi = pos, h=0..6)")]:
            mont = []
            for h in range(0, H_CONJUNTO + 1):
                d["__y"] = L.y_acum(d, nome, h)
                regs = cab + din + dum + pand(d, h)
                mont.append((h, regs, _amostra(d, regs)))
            com = set.intersection(*[set(am["Data"]) for _, _, am in mont])
            gs = [ajustar(am[am["Data"].isin(com)], regs, h, (a1, a2))
                  for h, regs, am in mont]
            V = hac_conjunta_horizontes(gs, H_CONJUNTO + 1)
            Vi = np.linalg.pinv(V)
            bh = np.array([g["valor"] for g in gs])
            W = float(bh @ Vi @ bh)
            Bm = np.column_stack([replicas(g, etas, datas_ref)[0] for g in gs])
            Wb = np.einsum("ij,jk,ik->i", Bm - bh, Vi, Bm - bh)
            conjuntos.append({"variavel": nome, "teste": rot, "estatistica": W,
                              "p_qui2": float(stats.chi2.sf(W, H_CONJUNTO + 1)),
                              "p_bootstrap": float((Wb >= W).mean()),
                              "critico_95_bootstrap": float(np.quantile(Wb, 0.95)),
                              "observacoes": gs[0]["nobs"]})
        print(f"  {nome}: sup-t = " +
              ", ".join(f"{ROTULO[r]} {cunif[r]:.2f}" for r in REGIMES))

    t, tw, tc = pd.DataFrame(linhas), pd.DataFrame(waldh), pd.DataFrame(conjuntos)
    if salvar:
        with pd.ExcelWriter(SAIDA / "onda4_1_tres_regimes.xlsx") as w:
            t.to_excel(w, sheet_name="respostas", index=False)
            tw.to_excel(w, sheet_name="wald_por_horizonte", index=False)
            tc.to_excel(w, sheet_name="testes_conjuntos", index=False)
    return t, tw, tc


def consistencia():
    """Se o 3o regime for vazio, o modelo tem de reproduzir o de 2 regimes."""
    print("\n=== checagem: corte alem do fim da amostra reproduz 2 regimes ===")
    t, _, _ = rodar(corte_fim="2030-01-01", salvar=False, rotulo_corte="fora da amostra")
    ref = pd.read_excel(SAIDA / "onda3_3_regimes_data_escolhida.xlsx")
    ref = ref[ref.corte == "escolhida"]
    pior = 0.0
    for _, x in ref.iterrows():
        for reg, col in [("pre-PPI", "coef_pre"), ("PPI", "coef_pos")]:
            s = t[(t.variavel == x.variavel) & (t.h == x.h) & (t.regime == reg)]
            if len(s):
                pior = max(pior, abs(float(s.coeficiente.iloc[0]) - float(x[col])))
    print(f"desvio maximo contra onda3_3: {pior:.3e}  "
          f"({'OK' if pior < 1e-8 else 'ATENCAO'})")
    return pior


def figuras():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.read_excel(SAIDA / "onda4_1_tres_regimes.xlsx", sheet_name="respostas")
    for nome in PRINCIPAIS:
        s = t[t.variavel == nome]
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for r in REGIMES:
            q = s[s.regime == ROTULO[r]].sort_values("h")
            ax.plot(q.h, q.coeficiente, color=CORES[r], lw=2, marker="o", ms=3.2,
                    label=ROTULO[r])
            ax.fill_between(q.h, q.pontual_inferior, q.pontual_superior,
                            color=CORES[r], alpha=0.13, lw=0)
        ax.axhline(0, color="0.35", lw=0.9)
        ax.set_xlabel("meses após o choque", fontsize=8.5)
        ax.set_ylabel("pontos percentuais", fontsize=8.5)
        ax.set_xticks(range(0, 13, 2)); ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=7.5)
        ax.set_title(f"{nome}: resposta acumulada por regime de política de preços\n"
                     f"faixas = banda pontual de 90%", fontsize=10)
        fig.tight_layout()
        arq = FIGS / f"tres_regimes_{nome}.png"
        fig.savefig(arq, dpi=150); plt.close(fig)
        print(f"  {arq.name}")


def main():
    print(__doc__)
    t, tw, tc = rodar()

    print("\n=== respostas por regime, horizontes do artigo ===")
    v = t[t.h.isin([0, 1, 3, 6])]
    print(v.pivot_table(index=["variavel", "h"], columns="regime",
                        values="coeficiente").round(3).to_string())
    print("\n=== n efetivo do regime pos-PPI, por horizonte ===")
    print(t[t.variavel == "ipca_geral"][["h", "n_pre", "n_ppi", "n_pos"]]
          .drop_duplicates().to_string(index=False))

    print("\n=== Wald por horizonte ===")
    w = tw[tw.h.isin([0, 1, 3])]
    print(w.pivot_table(index=["variavel", "h"], columns="teste",
                        values="p_valor").round(3).to_string())

    print("\n=== testes conjuntos h=0..6 (critico por bootstrap) ===")
    print(tc[["variavel", "teste", "estatistica", "p_bootstrap",
              "critico_95_bootstrap"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))

    consistencia()

    print("\n=== robustez: corte em mai/2023 em vez de jun/2023 ===")
    ta, _, tca = rodar(corte_fim=CORTE_FIM_ALT, salvar=False, rotulo_corte="mai/2023")
    m = ta[(ta.h.isin([0, 1, 3])) & (ta.variavel.isin(PRINCIPAIS))]
    print(m.pivot_table(index=["variavel", "h"], columns="regime",
                        values="coeficiente").round(3).to_string())

    figuras()


if __name__ == "__main__":
    main()
