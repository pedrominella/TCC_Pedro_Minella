# -*- coding: utf-8 -*-
"""
FIGURA DO VAR COM IDENTIFICACAO RECURSIVA (CHOLESKY), POR REGIME

Mesmas variaveis, mesmas 3 defasagens, mesma amostra e mesmo corte do modelo
principal. Ordem recursiva: petroleo, cambio, atividade, preco, Selic,
expectativa, com o petroleo em primeiro. Resposta ACUMULADA a um choque de um
desvio-padrao.

BANDA: bootstrap selvagem de residuos com pesos de Rademacher, o mesmo esquema
usado no restante do trabalho. Em cada replica o sistema e reconstruido
recursivamente a partir das tres observacoes iniciais efetivas, o VAR e
reestimado e a IRF acumulada e recalculada. A banda e o intervalo percentil de
90%. Nao se usa errband_mc do statsmodels, que devolve banda por periodo e nao
da acumulada.

A linha tracejada e a projecao local do capitulo 4, sobreposta para comparacao.

Uso: python fig_var_cholesky.py
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
import onda5_alternativos as A

PASTA = Path(__file__).resolve().parent.parent
FIGS = PASTA / "figuras"
SAIDA = PASTA / "resultados" / "modelo_11_alternativos"
SAIDA.mkdir(parents=True, exist_ok=True)
OLIVA, TERRA, ESCURO = "#5F6549", "#9E3D25", "#272A24"
REPL = 400

ROTULO = {"gasolina_refinaria": "Gasolina na refinaria",
          "gasolina_consumidor": "Gasolina ao consumidor",
          "diesel": "Óleo diesel", "etanol": "Etanol",
          "ipca_geral": "IPCA Geral", "ipca_transportes": "IPCA Transportes"}
ORDEM = ["ipca_geral", "ipca_transportes", "gasolina_refinaria",
         "gasolina_consumidor", "diesel", "etanol"]


def cum_irf(res, j, H):
    """IRF ortogonalizada acumulada da variavel j a um choque na equacao 0."""
    return res.irf(H).orth_cum_effects[:, j, 0]


def banda(m, j, H, repl, gerador):
    """Bootstrap selvagem de residuos com pesos de Rademacher."""
    Y = m.values
    res = VAR(Y).fit(L.LAGS, trend="c")
    U = res.resid                       # (T - p, k)
    A_ = res.coefs                      # (p, k, k)
    c_ = res.intercept
    p, T, k = L.LAGS, len(Y), Y.shape[1]
    fora = np.empty((repl, H + 1)); fora[:] = np.nan
    for r in range(repl):
        s = gerador.choice([-1.0, 1.0], size=len(U))[:, None]
        Ub = U * s
        Yb = np.empty_like(Y)
        Yb[:p] = Y[:p]
        for t in range(p, T):
            v = c_.copy()
            for l in range(p):
                v = v + A_[l] @ Yb[t - 1 - l]
            Yb[t] = v + Ub[t - p]
        try:
            fora[r] = cum_irf(VAR(Yb).fit(p, trend="c"), j, H)
        except Exception:
            pass
    fora = fora[~np.isnan(fora).any(axis=1)]
    return (np.percentile(fora, 5, axis=0), np.percentile(fora, 95, axis=0),
            len(fora))


def prepara(b, nome, rg):
    d = b[b.regime_pos == rg]
    cols = ["choque_petroleo", "dlog_cambio", "dlog_atividade",
            f"dlog_{nome}", "selic", "expectativa"]
    m = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return m, cols.index(f"dlog_{nome}")


def main():
    b, _ = A.base()
    lp = pd.read_excel(PASTA / "resultados" / "onda3_3_regimes_data_escolhida.xlsx")
    lp = lp[lp.data_corte.astype(str).str.startswith("2017-03")]
    g = np.random.default_rng(7)

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.2))
    hh = np.arange(L.H_MAX + 1)
    guarda = []

    for k, nome in enumerate(ORDEM):
        a = axes[k // 3, k % 3]
        for rg, cor, rot in [(0, OLIVA, "regime anterior"), (1, TERRA, "regime posterior")]:
            m, j = prepara(b, nome, rg)
            cum = cum_irf(VAR(m.values).fit(L.LAGS, trend="c"), j, L.H_MAX)
            lo, hi, nb = banda(m, j, L.H_MAX, REPL, g)
            a.fill_between(hh, lo, hi, color=cor, alpha=0.16, lw=0, zorder=1)
            a.plot(hh, cum, "-", color=cor, lw=2.2, label="%s, VAR" % rot, zorder=4)
            for h in range(L.H_MAX + 1):
                guarda.append({"variavel": nome, "regime": "pos" if rg else "pre", "h": h,
                               "var_acumulada": cum[h], "banda_inf": lo[h],
                               "banda_sup": hi[h], "n": len(m), "replicas": nb})
            z = lp[lp.variavel == nome].sort_values("h")
            a.plot(z.h, z["coef_pos" if rg else "coef_pre"], "--", color=cor, lw=1.4,
                   alpha=0.9, dashes=(4, 2.4), label="%s, projeção local" % rot, zorder=3)
        a.axhline(0, color="0.4", lw=0.9)
        a.set_title(ROTULO[nome], fontsize=10.5, loc="left", color=ESCURO, pad=6)
        a.tick_params(labelsize=8); a.set_xticks([0, 2, 4, 6, 8, 10, 12])
        if k // 3 == 1:
            a.set_xlabel("meses após o choque", fontsize=9)
        if k % 3 == 0:
            a.set_ylabel("resposta acumulada, p.p.", fontsize=9)
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
        print("   %-22s bandas com %d réplicas válidas" % (nome, nb))

    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Resposta acumulada a um choque de um desvio-padrão no preço do petróleo",
                 fontsize=12, x=0.008, ha="left", y=0.995, color=ESCURO)
    fig.text(0.008, 0.958,
             "VAR recursivo com as mesmas variáveis, as mesmas três defasagens e o mesmo corte "
             "do modelo principal; banda de 90%% por bootstrap selvagem de resíduos, %d réplicas"
             % REPL, fontsize=8.6, color="0.35", ha="left")
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    arq = FIGS / "var_cholesky_regimes.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(guarda).to_excel(SAIDA / "var_cholesky_bandas.xlsx", index=False)
    print("  ", arq.name)


if __name__ == "__main__":
    main()
