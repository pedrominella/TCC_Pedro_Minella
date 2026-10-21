# -*- coding: utf-8 -*-
"""
ONDA 2, ITEM 2.4 -- QUANTO DO REPASSE DO PETROLEO A INFLACAO PASSA PELOS COMBUSTIVEIS

Este e o objeto que da titulo ao trabalho. Nao o efeito total do petroleo sobre a
inflacao brasileira, mas a PARCELA desse efeito que atravessa os precos domesticos
dos combustiveis:

    choque do petroleo  ->  precos domesticos dos combustiveis  ->  IPCA

O caminho e decomposto na MESMA amostra, por Frisch-Waugh:

    efeito TOTAL   : beta do choque sem os combustiveis no modelo
    efeito DIRETO  : beta do choque COM os combustiveis acumulados no modelo
    CANAL          : total - direto = a parcela que atravessa os combustiveis

e vale a identidade exata

    canal = soma_j (delta_j * b_j)

com b_j o coeficiente do mediador j no modelo direto e delta_j o coeficiente do
choque na regressao auxiliar do mediador j sobre o desenho do modelo TOTAL. Os
dois sao reportados separadamente porque sao a leitura economica do canal: quanto
o choque move o combustivel, e quanto o combustivel move a inflacao. A identidade
tambem serve de verificacao numerica do codigo.

O mediador e acumulado na MESMA janela do desfecho -- 100*[ln P(t+h) - ln P(t-1)] --
porque a inflacao acumulada entre 0 e h responde a variacao de combustivel acumulada
entre 0 e h, nao apenas a do mes do choque. A versao so contemporanea entra como
verificacao.

Nenhum mediador contemporaneo de cambio, atividade ou expectativa entra: eles sao
pos-tratamento e fechariam justamente o canal que se quer medir, inclusive a parte
que opera via cambio (paridade de importacao) e via revisao de expectativas diante
do combustivel mais caro. As defasagens de todos os controles permanecem.

RESSALVA DE IDENTIFICACAO -- PRECISA ESTAR NO TEXTO. A perna b nao e causalmente
identificada: nao ha instrumento separado para o preco do combustivel, e choques
proprios de combustivel correlacionados com a inflacao contaminariam o coeficiente.
A leitura defensavel e CONTABIL: quanto da resposta estimada da inflacao ao choque
desaparece quando se condiciona no combustivel. Nao "quanto do efeito e causado
pelo combustivel".

ESPECIFICACAO
-------------
A de referencia da pasta, com todas as correcoes da auditoria de 27/08:
resposta acumulada em log-nivel, meta Selic SGS 432, dummies de pandemia por janela
do horizonte, HAC com T/(T-K), quebra em mar/2017 sem deslocamento.

Difere da implementacao antiga (`revisao_2026_08/.../parte_6_repasse_via_combustiveis.py`)
em tudo isso -- aquela somava as colunas `Var_X` (o vies do item 1.1), usava a serie
de juros errada, nao tinha dummies de pandemia e usava bootstrap de PARES em blocos
moveis, que reamostra o desenho. Aqui o desenho e fixo e o bootstrap e wild em blocos
com residuo corrigido por alavancagem, como no resto da pasta.

POR REGIME. A auditoria mostrou que a heterogeneidade de regime e o resultado robusto
do trabalho e que a media de amostra cheia nao e um parametro estrutural. A
decomposicao e portanto estimada tambem por regime, interagindo o choque E os
mediadores com a indicadora de mar/2017. E a pergunta que interessa: depois da PPI,
o repasse cresceu -- e ele passou MAIS pelos combustiveis?

Uso: python onda2_4_canal_combustiveis.py rodar | figuras
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")

_NOMES = {"ipca_geral": "IPCA Geral", "ipca_transportes": "IPCA Transportes",
          "gasolina_consumidor": "Gasolina ao consumidor",
          "gasolina_refinaria": "Gasolina na refinaria",
          "diesel": "Óleo diesel", "etanol": "Etanol",
          "ipca_comercializaveis": "Comercializáveis",
          "ipca_naocomercializaveis": "Não comercializáveis"}

AQUI = Path(__file__).resolve().parent
PASTA = AQUI.parent
SAIDA = PASTA / "resultados"; FIGS = PASTA / "figuras"
SAIDA.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)

import lp_base_nova as L

B = 999
BLOCO = 12
SEMENTE = 20260830
DESFECHOS = ["ipca_geral", "ipca_transportes"]
MEDIADORES = {
    "gasolina_consumidor": ["gasolina_consumidor"],
    "diesel": ["diesel"],
    "gasolina_e_diesel": ["gasolina_consumidor", "diesel"],
    "todos_os_combustiveis": ["gasolina_consumidor", "diesel", "etanol"],
}
ROTULOS = {"gasolina_consumidor": "Gasolina ao consumidor", "diesel": "Diesel",
           "gasolina_e_diesel": "Gasolina e diesel",
           "todos_os_combustiveis": "Gasolina, diesel e etanol"}
PRINCIPAL = "todos_os_combustiveis"


# ------------------------------------------------------------------
def _linhas_proj(X, js):
    """Linhas c tais que c'y = beta_j, para cada j em js. Uma pinv so."""
    A = np.linalg.pinv(X.T @ X)
    return {j: A[:, j] @ X.T for j in js}


def _ajuste(X, y):
    P = np.linalg.pinv(X)
    beta = P @ y
    resid = y - X @ beta
    lev = np.clip(np.einsum("ij,ji->i", X, P), 0.0, 0.999)
    return beta, resid, resid / np.sqrt(1.0 - lev)


def decompor(dados, y, choques, mediadores, comuns, h, T_ref=None):
    """Decomposicao de Frisch-Waugh para um conjunto de mediadores.

    choques    : lista de colunas do choque (1 na amostra cheia, 2 por regime)
    mediadores : lista de colunas mediadoras a acrescentar no modelo direto
    Devolve um dict por choque com total, direto, canal, erros-padrao HAC e as
    pernas delta_j e b_j que reconstroem o canal.
    """
    n = len(y)
    banda = max(1, h + 1)
    Xt = np.column_stack([np.ones(n), dados[choques + comuns].to_numpy(float)])
    Xd = np.column_stack([np.ones(n), dados[choques + mediadores + comuns].to_numpy(float)])
    bt, _, _ = _ajuste(Xt, y)
    bd, res_d, resb_d = _ajuste(Xd, y)          # o modelo DIRETO e o mais geral: DGP

    js = list(range(1, 1 + len(choques)))
    CT, CD = _linhas_proj(Xt, js), _linhas_proj(Xd, js)
    saida = {"Xt": Xt, "Xd": Xd, "resid_boot": resb_d, "ajuste_d": Xd @ bd, "n": n,
             "cols": {"ct": {ch: CT[1 + i] for i, ch in enumerate(choques)},
                      "cd": {ch: CD[1 + i] for i, ch in enumerate(choques)}}}
    for i, ch in enumerate(choques):
        ct, cd = CT[1 + i], CD[1 + i]
        total, direto = float(ct @ y), float(cd @ y)
        canal = total - direto
        # variancia sob o modelo mais geral; os tres sao lineares em y
        vt = L.hac_var_row(ct, res_d, banda, n, Xd.shape[1])
        vd = L.hac_var_row(cd, res_d, banda, n, Xd.shape[1])
        vc = L.hac_var_row(ct - cd, res_d, banda, n, Xd.shape[1])
        reg = {"choque": ch, "efeito_total": total, "ep_total": np.sqrt(vt),
               "efeito_nao_via_combustivel": direto, "ep_nao_via": np.sqrt(vd),
               "efeito_via_combustivel": canal, "ep_canal": np.sqrt(vc),
               "p_canal": float(2 * (1 - stats.norm.cdf(abs(canal / np.sqrt(vc))))) if vc > 0 else np.nan,
               "parcela_via_combustivel": canal / total if abs(total) > 1e-8 else np.nan,
               "observacoes": n}
        # pernas: delta_j (auxiliar do mediador j sobre o desenho TOTAL) x b_j
        soma = 0.0
        for k, m in enumerate(mediadores):
            b_j = float(bd[1 + len(choques) + k])
            delta_j = float(ct @ dados[m].to_numpy(float))
            reg[f"delta_choque_para_{m}"] = delta_j
            reg[f"b_{m}_para_desfecho"] = b_j
            reg[f"produto_{m}"] = delta_j * b_j
            soma += delta_j * b_j
        reg["soma_produtos"] = soma
        reg["desvio_identidade"] = abs(soma - canal)
        saida[ch] = reg
    return saida


def _pesos(rng, datas_ref, n_rep):
    n = len(datas_ref)
    nb = int(np.ceil(n / BLOCO))
    return pd.DataFrame(
        np.column_stack([np.repeat(rng.choice([-1.0, 1.0], size=nb), BLOCO)[:n]
                         for _ in range(n_rep)]), index=datas_ref)


def rodar():
    rng = np.random.default_rng(SEMENTE)
    b, dum, pand = L.carregar_pandemia("corrigida")
    b["regime_pos"] = L.regime(b.Data, L.DATA_CORTE_FINAL)
    b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
    b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
    datas_ref = pd.DatetimeIndex(sorted(b["Data"].unique()))
    etas = _pesos(rng, datas_ref, B)

    linhas, linhas_dif, pior_identidade = [], [], 0.0
    for modo in ["amostra cheia", "por regime"]:
        choques = ["choque_petroleo"] if modo == "amostra cheia" else ["choque_pre", "choque_pos"]
        for desfecho in DESFECHOS:
            d = b.copy()
            for lag in range(1, L.LAGS + 1):
                d[f"y_lag{lag}"] = d[f"dlog_{desfecho}"].shift(lag)
            comuns_base = []
            for lag in range(1, L.LAGS + 1):
                comuns_base += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + \
                               [f"{c}_lag{lag}" for c in L.CONTROLES]
            if modo == "por regime":
                comuns_base = ["regime_pos"] + comuns_base

            for h in range(0, L.H_MAX + 1):
                d["__y"] = L.y_acum(d, desfecho, h)
                # mediadores acumulados na MESMA janela do desfecho
                todos_comb = sorted({c for v in MEDIADORES.values() for c in v})
                med_cols = {}
                for c in todos_comb:
                    d[f"__m_{c}"] = L.y_acum(d, c, h)
                    if modo == "por regime":
                        d[f"__m_{c}_pre"] = d[f"__m_{c}"] * (1 - d["regime_pos"])
                        d[f"__m_{c}_pos"] = d[f"__m_{c}"] * d["regime_pos"]
                        med_cols[c] = [f"__m_{c}_pre", f"__m_{c}_pos"]
                    else:
                        med_cols[c] = [f"__m_{c}"]
                comuns = comuns_base + dum + pand(d, h)
                usadas = ["Data", "__y"] + choques + comuns + \
                         [x for c in todos_comb for x in med_cols[c]]
                dados = d[usadas].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
                y = dados["__y"].to_numpy(float)
                datas = pd.DatetimeIndex(dados["Data"])
                E = etas.reindex(datas).to_numpy()

                for nome_canal, comps in MEDIADORES.items():
                    meds = [x for c in comps for x in med_cols[c]]
                    r = decompor(dados, y, choques, meds, comuns, h)
                    pior_identidade = max(pior_identidade,
                                          max(r[ch]["desvio_identidade"] for ch in choques))
                    # bootstrap wild em blocos, desenho fixo, DGP = modelo direto
                    Y = r["ajuste_d"][:, None] + r["resid_boot"][:, None] * E
                    boot = {}
                    for ch in choques:
                        tb = r["cols"]["ct"][ch] @ Y
                        db = r["cols"]["cd"][ch] @ Y
                        cb = tb - db
                        pb = np.divide(cb, tb, out=np.full_like(cb, np.nan), where=np.abs(tb) > 1e-8)
                        boot[ch] = (cb, pb)
                    # (F1.1) TESTE DA DIFERENCA DE CANAL ENTRE OS DOIS REGIMES.
                    # A secao 4.5 afirma que a diferenca entre o canal do regime
                    # anterior e o do posterior "e testavel diretamente" e nao
                    # reporta o teste. Ela e, e por duas vias que aqui coincidem.
                    # (a) ANALITICA: o canal de cada regime e (ct - cd) @ y, linear
                    #     em y; logo a diferenca tambem e, com a linha
                    #     w = (ct_pos - cd_pos) - (ct_pre - cd_pre), e admite o
                    #     mesmo erro-padrao HAC usado no resto do trabalho.
                    # (b) BOOTSTRAP: as duas series de replicas vem das MESMAS
                    #     reamostragens, entao a diferenca replica a replica ja
                    #     incorpora a covariancia entre os dois canais.
                    if modo == "por regime" and len(choques) == 2:
                        cbp, cba = boot["choque_pos"][0], boot["choque_pre"][0]
                        dif_b = cbp - cba
                        dif_obs = (r["choque_pos"]["efeito_via_combustivel"]
                                   - r["choque_pre"]["efeito_via_combustivel"])
                        w = ((r["cols"]["ct"]["choque_pos"] - r["cols"]["cd"]["choque_pos"])
                             - (r["cols"]["ct"]["choque_pre"] - r["cols"]["cd"]["choque_pre"]))
                        res_d = y - r["ajuste_d"]
                        v_dif = L.hac_var_row(w, res_d, max(1, h + 1), r["n"], r["Xd"].shape[1])
                        ep_dif = float(np.sqrt(v_dif)) if v_dif > 0 else np.nan
                        p_hac = (float(2 * (1 - stats.norm.cdf(abs(dif_obs / ep_dif))))
                                 if ep_dif == ep_dif and ep_dif > 0 else np.nan)
                        # p por inversao do intervalo percentil (mesma convencao
                        # que canal_signif_boot_10 ja usa para o nivel)
                        p_perc = float(2 * min(np.nanmean(dif_b <= 0), np.nanmean(dif_b >= 0)))
                        # p com a distribuicao recentrada na hipotese nula
                        dif_c = dif_b - np.nanmean(dif_b)
                        p_nulo = float(np.nanmean(np.abs(dif_c) >= abs(dif_obs)))
                        linhas_dif.append({
                            "desfecho": desfecho, "mediador": nome_canal,
                            "rotulo": ROTULOS[nome_canal], "h": h,
                            "canal_pre": r["choque_pre"]["efeito_via_combustivel"],
                            "canal_pos": r["choque_pos"]["efeito_via_combustivel"],
                            "diferenca": dif_obs, "ep_hac_diferenca": ep_dif,
                            "p_hac": p_hac, "p_boot_percentil": p_perc,
                            "p_boot_nulo": p_nulo,
                            "dif_boot_ic5": float(np.nanpercentile(dif_b, 5)),
                            "dif_boot_ic95": float(np.nanpercentile(dif_b, 95)),
                            "signif_10_hac": bool(p_hac == p_hac and p_hac < 0.10),
                            "signif_10_boot": bool(
                                np.nanpercentile(dif_b, 5) * np.nanpercentile(dif_b, 95) > 0),
                            "observacoes": r["n"],
                        })
                    for ch in choques:
                        cb, pb = boot[ch]
                        reg = dict(r[ch])
                        reg.update({
                            "modo": modo, "desfecho": desfecho, "mediador": nome_canal,
                            "rotulo": ROTULOS[nome_canal], "h": h,
                            "canal_boot_ic5": float(np.nanpercentile(cb, 5)),
                            "canal_boot_ic95": float(np.nanpercentile(cb, 95)),
                            "canal_signif_boot_10": bool(
                                np.nanpercentile(cb, 5) * np.nanpercentile(cb, 95) > 0),
                            "parcela_boot_ic5": float(np.nanpercentile(pb, 5)),
                            "parcela_boot_ic95": float(np.nanpercentile(pb, 95)),
                        })
                        linhas.append(reg)
            print(f"  {modo:14s} {desfecho}")

    t = pd.DataFrame(linhas)
    cols = ["modo", "desfecho", "mediador", "rotulo", "choque", "h", "observacoes",
            "efeito_total", "ep_total", "efeito_nao_via_combustivel", "ep_nao_via",
            "efeito_via_combustivel", "ep_canal", "p_canal",
            "canal_boot_ic5", "canal_boot_ic95", "canal_signif_boot_10",
            "parcela_via_combustivel", "parcela_boot_ic5", "parcela_boot_ic95",
            "soma_produtos", "desvio_identidade"]
    resto = [c for c in t.columns if c not in cols]
    t = t[cols + resto]
    with pd.ExcelWriter(SAIDA / "onda2_4_canal_combustiveis.xlsx") as w:
        t[t.modo == "amostra cheia"].to_excel(w, sheet_name="amostra_cheia", index=False)
        t[t.modo == "por regime"].to_excel(w, sheet_name="por_regime", index=False)
        pd.DataFrame(linhas_dif).to_excel(w, sheet_name="diferenca_regimes", index=False)
        t[(t.mediador == PRINCIPAL) & (t.h.isin(L.HORIZONTES))].to_excel(
            w, sheet_name="tabela_artigo", index=False)

    print(f"\nidentidade canal = soma(delta_j * b_j): desvio maximo {pior_identidade:.2e}"
          f"  ({'OK' if pior_identidade < 1e-8 else 'FALHOU'})")

    print("\n=== amostra cheia: quanto do repasse passa pelos combustiveis ===")
    v = t[(t.modo == "amostra cheia") & (t.mediador == PRINCIPAL) & (t.h.isin(L.HORIZONTES))]
    print(v[["desfecho", "h", "efeito_total", "efeito_via_combustivel", "ep_canal",
             "p_canal", "efeito_nao_via_combustivel", "parcela_via_combustivel"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n=== por regime (mar/2017), mediador = todos os combustiveis ===")
    v = t[(t.modo == "por regime") & (t.mediador == PRINCIPAL) & (t.h.isin(L.HORIZONTES))]
    print(v[["desfecho", "choque", "h", "efeito_total", "efeito_via_combustivel",
             "p_canal", "parcela_via_combustivel"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n=== por conjunto de combustiveis, IPCA Geral, amostra cheia, h=3 ===")
    v = t[(t.modo == "amostra cheia") & (t.desfecho == "ipca_geral") & (t.h == 3)]
    print(v[["rotulo", "efeito_via_combustivel", "ep_canal", "p_canal",
             "parcela_via_combustivel", "parcela_boot_ic5", "parcela_boot_ic95"]].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))
    return t


def figuras():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # painel esquerdo tambem por regime (auditoria 06/09): usa o regime posterior
    t = pd.read_excel(SAIDA / "onda2_4_canal_combustiveis.xlsx", sheet_name="por_regime")
    t = t[t.choque == "choque_pos"]
    tr = pd.read_excel(SAIDA / "onda2_4_canal_combustiveis.xlsx", sheet_name="por_regime")
    C = {"total": "#3b74d4", "canal": "#0f9b8e", "direto": "#d1495b"}
    for desfecho in DESFECHOS:
        s = t[(t.desfecho == desfecho) & (t.mediador == PRINCIPAL)].sort_values("h")
        fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.7))
        ax[0].plot(s.h, s.efeito_total, color=C["total"], lw=2, label="efeito total")
        ax[0].plot(s.h, s.efeito_via_combustivel, color=C["canal"], lw=2,
                   label="via combustíveis")
        ax[0].fill_between(s.h, s.canal_boot_ic5, s.canal_boot_ic95,
                           color=C["canal"], alpha=0.16)
        ax[0].plot(s.h, s.efeito_nao_via_combustivel, color=C["direto"], lw=1.6,
                   ls="--", label="não atribuído")
        ax[0].axhline(0, color="0.4", lw=0.8)
        ax[0].set_xlabel("meses"); ax[0].set_ylabel("%")
        ax[0].set_title("decomposição da resposta acumulada\n(regime posterior)", fontsize=9)
        ax[0].legend(fontsize=7)

        for ch, cor, rot in [("choque_pre", "#8a8f7d", "pré mar/2017"),
                             ("choque_pos", "#9E3D25", "pós mar/2017")]:
            r = tr[(tr.desfecho == desfecho) & (tr.mediador == PRINCIPAL)
                   & (tr.choque == ch)].sort_values("h")
            ax[1].plot(r.h, r.efeito_via_combustivel, color=cor, lw=2, label=rot)
            ax[1].fill_between(r.h, r.canal_boot_ic5, r.canal_boot_ic95, color=cor, alpha=0.14)
        ax[1].axhline(0, color="0.4", lw=0.8)
        ax[1].set_xlabel("meses")
        ax[1].set_title("efeito via combustíveis, por regime", fontsize=9)
        ax[1].legend(fontsize=7)
        fig.suptitle(f"{_NOMES.get(desfecho, desfecho)}: repasse do petróleo via combustíveis", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGS / f"canal_combustiveis_{desfecho}.png", dpi=150); plt.close(fig)
        print(f"  figura {desfecho}")


if __name__ == "__main__":
    {"rodar": rodar, "figuras": figuras}[sys.argv[1] if len(sys.argv) > 1 else "rodar"]()
