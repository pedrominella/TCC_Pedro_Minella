# -*- coding: utf-8 -*-
"""
LJUNG-BOX NOS RESIDUOS DA ESPECIFICACAO PRINCIPAL, h = 0 (criado em 06/10/2026)

Mesma regressao da Tabela 1 e do onda2_5 (3 defasagens, 11 sazonais, 10 indicadoras de pandemia,
choque contemporaneo), uma para cada uma das seis series, em h = 0, onde a dependente e a variacao
de um mes e nao ha sobreposicao de janelas. Calcula, para 1, 6 e 12 defasagens de residuo:
  - Ljung-Box simples (graus de liberdade = numero de defasagens);
  - Ljung-Box com graus de liberdade reduzidos em 3 (as defasagens da propria variavel), a
    correcao usual quando o residuo vem de um modelo autorregressivo. Com regressores exogenos o
    ajuste correto nao e exato; por isso as duas versoes sao mostradas;
  - Breusch-Godfrey classico e robusto, ja calculados em onda2_5_lags_residuos.xlsx, so copiados
    aqui para ficar tudo numa tabela.
Saida: resultados/onda2_9_ljungbox_h0.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey
warnings.filterwarnings("ignore")
PASTA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_base_nova as L
from onda3_3_data_comum import sem_colunas_vazias

b, dum, pand = L.carregar_pandemia("corrigida")
linhas = []
for nome in L.VARIAVEIS:
    d = b.copy()
    for lag in range(1, L.LAGS + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
    regs = ["choque_petroleo"]
    for lag in range(1, L.LAGS + 1):
        regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in L.CONTROLES]
    d["__y"] = L.y_acum(d, nome, 0)
    regs += dum + pand(d, 0)
    am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
    X = sem_colunas_vazias(sm.add_constant(am[regs], has_constant="add"))
    r = sm.OLS(am["__y"], X).fit()
    for q in (1, 6, 12):
        lb = acorr_ljungbox(r.resid, lags=[q], model_df=0)
        lb3 = acorr_ljungbox(r.resid, lags=[q], model_df=min(3, q - 1)) if q > 3 else None
        bg = acorr_breusch_godfrey(r, nlags=q)
        linhas.append(dict(variavel=nome, defasagens_residuo=q, obs=int(r.nobs),
                           ljungbox_p=float(lb["lb_pvalue"].iloc[0]),
                           ljungbox_p_gl_ajustado=float(lb3["lb_pvalue"].iloc[0]) if lb3 is not None else np.nan,
                           breusch_godfrey_classico_p=float(bg[1])))
t = pd.DataFrame(linhas)
bgr = pd.read_excel(PASTA / "resultados" / "onda2_5_lags_residuos.xlsx", sheet_name="breusch_godfrey_h0")
bgr = bgr[bgr.lags_modelo == 3].rename(columns={"bg_lags": "defasagens_residuo", "bg_robusto_p": "breusch_godfrey_robusto_p"})
t = t.merge(bgr[["variavel", "defasagens_residuo", "breusch_godfrey_robusto_p"]], on=["variavel", "defasagens_residuo"], how="left")
t.to_csv(PASTA / "resultados" / "onda2_9_ljungbox_h0.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False))
