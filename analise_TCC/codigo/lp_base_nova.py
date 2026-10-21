# -*- coding: utf-8 -*-
"""
LP com a base corrigida -- modulo comum da Onda 2/3.

Duas mudancas em relacao ao que esta no TCC_artigo (que fica intacto):
  - resposta acumulada em log-nivel, 100*[ln P(t+h) - ln P(t-1)]   (item 1.1)
  - juros pela meta Selic oficial (SGS 432)                        (adendo 2)

Roda LP-OLS, LP-IV e regimes nas duas versoes da serie de juros e compara, para
confirmar que a troca nao move resultado. Sem VAR.

CORRECOES DE 27/08 (ver LEIA-ME, secao "Auditoria")
---------------------------------------------------
(C1) O indicador de regime nao leva mais `.shift(1)`. A convencao antiga fazia o
     regime comecar UM MES DEPOIS da data nominal (set/2016 virava out/2016), e
     era incompativel com a busca de quebra endogena, que sempre usou
     1{t >= tau}. Agora as duas pontas usam a mesma definicao: `regime()`.

(C2) O choque e o instrumento sao padronizados por um desvio-padrao calculado
     numa JANELA DE REFERENCIA FIXA (2003-01 a 2025-09), nao na amostra truncada
     de cada rotina. Antes o LP-OLS (fim 2025-09) e o LP-IV (fim 2025-06) usavam
     escalas diferentes e os coeficientes nao eram estritamente comparaveis.

(C3) As indicadoras de pandemia passam a acompanhar a JANELA DO HORIZONTE. Como
     y(t,h) = lnP(t+h) - lnP(t-1) contem as variacoes de t a t+h, um mes
     pandemico s contamina a observacao t sempre que t <= s <= t+h. A dummy
     correta e 1{t <= s <= t+h}; a antiga, 1{t = s}, so cobria h=0 e deixava,
     em h=12, 12 das 22 observacoes contaminadas sem controle nenhum.
     Em h=0 a nova definicao recai exatamente na antiga.

(C4) Todos os erros-padrao HAC passam a usar correcao de amostra finita
     T/(T-K). Com 41 a 44 regressores em ~265 observacoes (K/T de 15 a 17%) e
     banda de ate 13, a versao sem correcao subestima a variancia.

(C5) O instrumento externo BRUTO de Kanzig (`Oil supply surprise series`, a
     surpresa de futuros em torno de anuncios da OPEP) passa a ser carregado ao
     lado do choque recuperado do proxy-SVAR (`Oil supply news shock`). Os dois
     nao sao a mesma coisa: o segundo e uma combinacao linear dos residuos de
     forma reduzida daquele VAR -- um dos quais e o do proprio preco do
     petroleo --, e e por isso que o F de primeiro estagio dele passa de 200.
     Ver onda2_3_forma_reduzida_ar.py.

Uso: python lp_base_nova.py ols | iv | regimes | comparar
"""
import os, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
PASTA = AQUI.parent
RAIZ = PASTA.parent
SAIDA = PASTA / "resultados"; SAIDA.mkdir(parents=True, exist_ok=True)

DATA_INICIO, DATA_FIM, DATA_FIM_IV, DATA_CORTE = "2003-01-01", "2025-09-01", "2025-06-01", "2016-10-01"
DATA_CORTE_FINAL = "2017-03-01"      # quebra comum estimada (onda3_3)
LAGS, H_MAX, Z_90 = 3, 12, 1.645
HORIZONTES = [0, 1, 2, 3, 6, 12]
CONTROLES = ["dlog_cambio", "dlog_atividade", "selic", "expectativa"]
# (C7) FASE 0 - controle global opcional, desligado por padrao.
# A especificacao do trabalho tem vetor de controles inteiramente domestico, ao passo
# que o tratamento e um preco internacional e o regime posterior contem 2021-22, quando
# petroleo, alimentos, fretes e bens globais subiram juntos. Ligar por
#   LP_CONTROLE_GLOBAL=cpi_eua   (CPI cheio dos EUA, ja na base)
#   LP_CONTROLE_GLOBAL=nao_energia  (indice de commodities nao energeticas, se disponivel)
# Sem a variavel de ambiente o comportamento e identico ao original.
CONTROLE_GLOBAL = os.environ.get("LP_CONTROLE_GLOBAL", "").strip().lower()
GLOBAIS = {"cpi_eua": "dlog_cpi_eua", "nao_energia": "dlog_nao_energia"}
if CONTROLE_GLOBAL:
    if CONTROLE_GLOBAL not in GLOBAIS:
        raise ValueError(f"LP_CONTROLE_GLOBAL invalido: {CONTROLE_GLOBAL}")
    CONTROLES = CONTROLES + [GLOBAIS[CONTROLE_GLOBAL]]
VARIAVEIS = {
    "gasolina_refinaria": ("Var_GasolinaABrasil_media", "GasolinaABrasil_media_nivel"),
    "gasolina_consumidor": ("Var_Gasolina", "Gasolina_nivel"),
    "etanol": ("Var_Etanol", "Etanol_nivel"),
    "diesel": ("Var_Oleo_diesel", "Oleo_diesel_nivel"),
    "ipca_geral": ("Var_IPCA_Geral", "IPCA_Geral_nivel"),
    "ipca_transportes": ("Var_IPCA_Trans", "IPCA_Trans_nivel"),
}
JUROS = {"antiga": "selic_base_original", "corrigida": "meta_selic_sgs432"}  # meta oficial SGS 432
PANDEMIA = pd.date_range("2020-03-01", "2020-12-01", freq="MS")


# ==================================================================
# MODELO PRINCIPAL -- a especificacao de referencia do trabalho
# ==================================================================
MODELO_PRINCIPAL = """
ESPECIFICACAO PRINCIPAL (decidida em 27/08, apos a auditoria)

  Equacao, para cada variavel e cada horizonte h = 0..12, estimada por OLS:

    100*[ln P(t+h) - ln P(t-1)]
        = a
        + b_pre(h) * choque(t) * 1{t <  mar/2017}
        + b_pos(h) * choque(t) * 1{t >= mar/2017}
        + d(h)     * 1{t >= mar/2017}
        + soma_{i=1..3} [ g_i * dlnP(t-i) + f_i * choque(t-i) + O_i * X(t-i) ]
        + 11 indicadoras mensais
        + 10 indicadoras de pandemia de janela, 1{t <= s <= t+h}, s em mar..dez/2020
        + u(t,h)

  choque(t)  variacao logaritmica mensal do Brent em USD, padronizada pelo desvio
             da janela de referencia fixa 2003-01 a 2025-09
  X(t-i)     cambio (dlog), atividade (dlog), meta Selic SGS 432 em 100*ln(1+r/100),
             expectativa de inflacao idem -- SO DEFASADOS, nunca em t
  amostra    2003-01 a 2025-09, mensal

  OBJETO PRINCIPAL: b_pre(h) e b_pos(h), e o teste de b_pre = b_pos.
  A versao de amostra cheia (um unico b(h), sem interacao) e DESCRITIVA: os testes
  conjuntos mostram que a media de amostra cheia mistura dois regimes e nao rejeita
  para IPCA Geral nem etanol, enquanto a igualdade entre regimes e rejeitada nas
  seis variaveis. A media nao e um parametro estrutural.

  POR QUE p = 3: AIC escolhe 2 e BIC escolhe 1 na equacao de h=0; adota-se 3
  seguindo o aumento de defasagem de Montiel Olea e Plagborg-Moller (2021), que da
  validade uniforme com regressores persistentes -- caso da meta Selic, cujo ADF e
  -2,73 (p = 0,070). O repasse estimado e o contraste entre regimes sao
  praticamente invariantes a p entre 1 e 6 (item 2.5).

  POR QUE mar/2017: quebra comum ao sistema, estimada por sup-Wald multivariado com
  critico por bootstrap (item 3.3), p = 0,000. Set/2016 tambem fica acima do critico
  e entra como robustez.

INFERENCIA -- DOIS PROCEDIMENTOS, E SO DOIS

  (1) HAC de Newey-West (kernel de Bartlett, banda h+1, correcao T/(T-K)).
      E o UNICO erro-padrao de coeficiente em todo o trabalho. Dele saem:
        - banda pontual de 90%   : beta(h) +- 1,645 * ep(h)
        - banda uniforme de 90%  : beta(h) +- c * ep(h), c pelo sup-t plug-in
                                   (Montiel Olea e Plagborg-Moller, 2019) a partir
                                   da correlacao implicada pela covariancia HAC
                                   conjunta entre horizontes
        - Wald por horizonte e Wald conjunto de h=0 a 6
      A banda uniforme NAO e um segundo erro-padrao: e o mesmo ep, outro critico.

  (2) Wild block bootstrap (blocos de 12 meses, pesos de Rademacher, residuo
      corrigido por alavancagem, 299 a 999 replicas). Usado APENAS onde nao existe
      distribuicao analitica:
        - valor critico do sup-Wald de quebra, cuja nula nao e qui-quadrado
        - p-valor ajustado por selecao na data estimada (item 3.3)
        - valor critico do Wald conjunto, onde a qui-quadrado sobrerejeita
        - intervalo da PARCELA do canal de combustiveis, que e uma razao (item 2.4)

  Fora desses dois, so o conjunto de Anderson-Rubin na secao de LP-IV -- que nao e
  um erro-padrao e sim inversao de teste, e existe para responder a critica de
  instrumento fraco. Fica confinado aquela secao.

ARQUIVOS QUE **NAO** SAO O MODELO PRINCIPAL
  `aux_comparacao_juros_*.xlsx` sao a checagem de que trocar a serie de juros nao
  move resultado. Rodam no corte antigo (set/2016) e SEM dummies de pandemia, de
  proposito, porque sao comparacao com o estado anterior. Nao tirar numero de la.
"""


def especificacao():
    """Imprime a especificacao de referencia. `python lp_base_nova.py espec`."""
    print(MODELO_PRINCIPAL)
    print(f"constantes em uso: LAGS={LAGS}  H_MAX={H_MAX}  corte={DATA_CORTE_FINAL}"
          f"  amostra={DATA_INICIO} a {DATA_FIM}  juros={JUROS['corrigida']}")
    print(f"controles (so defasados): {CONTROLES}")
    print(f"pandemia: {len(PANDEMIA)} meses, {PANDEMIA[0]:%Y-%m} a {PANDEMIA[-1]:%Y-%m}")


# ------------------------------------------------------------------
# (C4) HAC com correcao de amostra finita, em todo lugar
# ------------------------------------------------------------------
def hac_kw(h):
    """cov_kwds do statsmodels para o horizonte h."""
    return {"maxlags": max(1, h + 1), "use_correction": True}


def hac_var_row(c, u, banda, T, K, corr=True):
    """Variancia HAC de c'y, com c a LINHA de projecao ja pronta (vetor T).

    Qualquer estimador linear em y -- um coeficiente, uma diferenca entre
    coeficientes, ou a diferenca entre coeficientes de DUAS regressoes aninhadas
    estimadas na mesma amostra -- e c'y para algum c. Reduzir o sanduiche ao
    vetor v_t = c_t * u_t torna o calculo O(T * banda).
    """
    v = c * u
    s = float(v @ v)
    for l in range(1, banda + 1):
        s += 2.0 * (1.0 - l / (banda + 1)) * float(v[l:] @ v[:-l])
    if corr:
        s *= T / (T - K)
    return max(s, 0.0)


def hac_var_combo(X, u, a, banda, corr=True):
    """Variancia HAC de a'beta, com a = (X'X)^-1 e_j (ou qualquer combinacao).

    Reduz o sanduiche a um unico vetor v_t = (X a)_t * u_t, o que torna o
    calculo O(T * banda) e viabiliza recalcular o erro-padrao dentro do
    bootstrap. Confere com sm.OLS(...).fit(cov_type='HAC', use_correction=True)
    ate a oitava casa (ver `autoteste()`).
    """
    T, K = X.shape
    return hac_var_row(X @ a, u, banda, T, K, corr)


# ------------------------------------------------------------------
# (C1) regime sem deslocamento
# ------------------------------------------------------------------
def regime(datas, corte):
    """1{t >= corte}. Sem `.shift(1)`: a mesma convencao da busca de quebra."""
    return (pd.DatetimeIndex(datas) >= pd.Timestamp(corte)).astype(float)


# ------------------------------------------------------------------
# (C3) indicadoras de pandemia compativeis com a janela do horizonte
# ------------------------------------------------------------------
def controles_pandemia(d, h, modo="janela"):
    """Cria/atualiza as indicadoras de pandemia em `d` e devolve seus nomes.

    modo='janela'  -> 1{t <= s <= t+h}, a definicao correta para y(t,h)
    modo='impulso' -> 1{t = s}, a definicao antiga (so para comparacao)
    Em h=0 as duas coincidem.
    """
    mi = d["Data"].dt.year * 12 + d["Data"].dt.month
    nomes = []
    for s in PANDEMIA:
        c = f"pand_{s:%Y%m}"
        si = s.year * 12 + s.month
        d[c] = (((mi <= si) & (si <= mi + h)) if modo == "janela" else (mi == si)).astype(float)
        nomes.append(c)
    return nomes


def carregar(juros, data_fim=DATA_FIM, com_kanzig=False):
    b = pd.read_parquet(PASTA / "base/base_corrigida.parquet")
    b = b[(b.Data >= DATA_INICIO) & (b.Data <= DATA_FIM)].copy().reset_index(drop=True)
    if com_kanzig:
        k = pd.read_excel(RAIZ / "base_de_dados/oilSupplyNewsShocks_2025M06.xlsx", sheet_name="Monthly")
        k.columns = [str(c).strip() for c in k.columns]
        k["Data"] = pd.to_datetime(k["Date"].astype(str).str.strip().str.replace("M", "-", regex=False) + "-01",
                                   errors="coerce")
        k["ck"] = pd.to_numeric(k["Oil supply news shock"], errors="coerce")
        # (C5) instrumento externo bruto, zero fora dos meses de anuncio da OPEP
        k["cs"] = pd.to_numeric(k["Oil supply surprise series"], errors="coerce")
        b = b.merge(k[["Data", "ck", "cs"]], on="Data", how="left")
    b["dlog_petroleo_usd"] = 100 * np.log(b["Preco_Barril"]).diff()
    b["dlog_cambio"] = 100 * np.log(b["Cambio"]).diff()
    b["dlog_atividade"] = 100 * np.log(b["Atividade"]).diff()
    # (C6) TAXAS EM LOGARITMO. Nao se aplica ln direto ao valor percentual: converte-se
    # para decimal, soma-se 1 e loga-se, obtendo a taxa composta continuamente, e
    # reescala-se por 100 para manter a variavel em unidade comparavel a percentual.
    #   r = 13,75%  ->  100*ln(1 + 0,1375) = 12,883
    # A compressao e desprezivel na ponta baixa e chega a 11,3% com a Selic em 26,5%.
    b["selic"] = 100.0 * np.log1p(
        pd.to_numeric(b[JUROS[juros]], errors="coerce") / 100.0)
    b["expectativa"] = 100.0 * np.log1p(
        pd.to_numeric(b["Expectativa_inflacao"], errors="coerce") / 100.0)
    # (C7) controle global, construido apenas quando pedido por LP_CONTROLE_GLOBAL
    if CONTROLE_GLOBAL == "cpi_eua":
        b["dlog_cpi_eua"] = 100 * np.log(pd.to_numeric(b["CPI_EUA"], errors="coerce")).diff()
    elif CONTROLE_GLOBAL == "nao_energia":
        ne = pd.read_csv(RAIZ / "base_de_dados/commodities_nao_energia.csv",
                         parse_dates=["Data"])
        b = b.merge(ne[["Data", "indice_nao_energia"]], on="Data", how="left")
        b["dlog_nao_energia"] = 100 * np.log(
            pd.to_numeric(b["indice_nao_energia"], errors="coerce")).diff()
        if int(b["dlog_nao_energia"].iloc[1:].isna().sum()):
            raise ValueError("meses sem indice de commodities nao energeticas")
    # (C2) escala fixada na janela de referencia, antes de qualquer truncamento
    b["choque_petroleo"] = b["dlog_petroleo_usd"] / b["dlog_petroleo_usd"].std()
    b["petroleo_endogeno"] = b["choque_petroleo"]
    if com_kanzig:
        b["instrumento_kanzig"] = b["ck"] / b["ck"].std()
        b["instrumento_surpresa"] = b["cs"] / b["cs"].std()
    for nome, (_, niv) in VARIAVEIS.items():
        b[f"ln_{nome}"] = 100 * np.log(pd.to_numeric(b[niv], errors="coerce"))
        b[f"dlog_{nome}"] = b[f"ln_{nome}"].diff()
    dm = pd.get_dummies(b.Data.dt.month, prefix="mes", drop_first=True, dtype=float)
    b = pd.concat([b, dm], axis=1)
    for lag in range(1, LAGS + 1):
        b[f"choque_petroleo_lag{lag}"] = b["choque_petroleo"].shift(lag)
        b[f"petroleo_lag{lag}"] = b["petroleo_endogeno"].shift(lag)
        if com_kanzig:
            b[f"kanzig_lag{lag}"] = b["instrumento_kanzig"].shift(lag)
            b[f"surpresa_lag{lag}"] = b["instrumento_surpresa"].shift(lag)
        for c in CONTROLES:
            b[f"{c}_lag{lag}"] = b[c].shift(lag)
    # o truncamento vem DEPOIS de tudo que depende de escala ou de defasagem
    b = b[b.Data <= data_fim].copy().reset_index(drop=True)
    if com_kanzig and int(b["ck"].isna().sum()):
        raise ValueError("meses sem Kanzig")
    return b, list(dm.columns)


def carregar_pandemia(juros="corrigida", data_fim=DATA_FIM, com_kanzig=False):
    """carregar(), com as indicadoras de pandemia registradas a parte.

    Devolve (b, dum, pand), com `pand` a funcao que instala as indicadoras da
    janela de um horizonte h em um DataFrame e devolve os nomes. As dummies sao
    h-dependentes (C3), entao nao podem mais ser uma lista fixa de colunas.
    """
    b, dum = carregar(juros, data_fim, com_kanzig)
    return b, dum, controles_pandemia


def y_acum(b, nome, h):
    return b[f"ln_{nome}"].shift(-h) - b[f"ln_{nome}"].shift(1)


def rodar_ols():
    linhas = []
    for juros in JUROS:
        b, dum = carregar(juros)
        for nome in VARIAVEIS:
            d = b.copy()
            for lag in range(1, LAGS + 1):
                d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
            regs = ["choque_petroleo"]
            for lag in range(1, LAGS + 1):
                regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in CONTROLES]
            regs += dum
            for h in range(0, H_MAX + 1):
                d["__y"] = y_acum(d, nome, h)
                am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                r = sm.OLS(am["__y"], sm.add_constant(am[regs], has_constant="add")).fit(
                    cov_type="HAC", cov_kwds=hac_kw(h))
                cf, ep = r.params["choque_petroleo"], r.bse["choque_petroleo"]
                linhas.append({"juros": juros, "variavel": nome, "h": h, "coeficiente": cf,
                               "erro_padrao": ep, "p_valor": r.pvalues["choque_petroleo"],
                               "limite_inferior_90": cf - Z_90 * ep, "limite_superior_90": cf + Z_90 * ep,
                               "observacoes": int(r.nobs)})
        print(f"  ols {juros}")
    pd.DataFrame(linhas).to_excel(SAIDA / "aux_comparacao_juros_ols.xlsx", index=False)


def rodar_iv():
    from linearmodels.iv import IV2SLS
    linhas = []
    for juros in JUROS:
        b, dum = carregar(juros, DATA_FIM_IV, com_kanzig=True)
        for nome in VARIAVEIS:
            d = b.copy()
            for lag in range(1, LAGS + 1):
                d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
            exog = []
            for lag in range(1, LAGS + 1):
                exog += [f"y_lag{lag}", f"petroleo_lag{lag}", f"kanzig_lag{lag}"] + [f"{c}_lag{lag}" for c in CONTROLES]
            exog += dum
            for h in range(0, H_MAX + 1):
                d["__y"] = y_acum(d, nome, h)
                am = d[["__y", "petroleo_endogeno", "instrumento_kanzig"] + exog].replace(
                    [np.inf, -np.inf], np.nan).dropna()
                Xe = sm.add_constant(am[exog], has_constant="add")
                riv = IV2SLS(dependent=am["__y"], exog=Xe, endog=am[["petroleo_endogeno"]],
                             instruments=am[["instrumento_kanzig"]]).fit(
                    cov_type="kernel", kernel="bartlett", bandwidth=max(1, h + 1))
                Xp = sm.add_constant(am[["instrumento_kanzig"] + exog], has_constant="add")
                rp = sm.OLS(am["petroleo_endogeno"], Xp).fit(cov_type="HAC", cov_kwds=hac_kw(h))
                rfr = sm.OLS(am["__y"], Xp).fit(cov_type="HAC", cov_kwds=hac_kw(h))
                cf, ep = riv.params["petroleo_endogeno"], riv.std_errors["petroleo_endogeno"]
                linhas.append({"juros": juros, "variavel": nome, "h": h, "coeficiente_iv": cf,
                               "erro_padrao_iv": ep, "p_valor_iv": riv.pvalues["petroleo_endogeno"],
                               "f_primeira_etapa": float(np.asarray(rp.f_test("instrumento_kanzig = 0").fvalue).squeeze()),
                               "coef_forma_reduzida": rfr.params["instrumento_kanzig"],
                               "p_forma_reduzida": rfr.pvalues["instrumento_kanzig"],
                               "observacoes": int(riv.nobs)})
        print(f"  iv {juros}")
    pd.DataFrame(linhas).to_excel(SAIDA / "aux_comparacao_juros_iv.xlsx", index=False)


def rodar_regimes():
    linhas = []
    for juros in JUROS:
        b, dum = carregar(juros)
        b["regime_pos"] = regime(b.Data, DATA_CORTE)          # (C1)
        b["choque_pre"] = b["choque_petroleo"] * (1 - b["regime_pos"])
        b["choque_pos"] = b["choque_petroleo"] * b["regime_pos"]
        for nome in VARIAVEIS:
            d = b.copy()
            for lag in range(1, LAGS + 1):
                d[f"y_lag{lag}"] = d[f"dlog_{nome}"].shift(lag)
            regs = ["choque_pre", "choque_pos", "regime_pos"]
            for lag in range(1, LAGS + 1):
                regs += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in CONTROLES]
            regs += dum
            for h in range(0, H_MAX + 1):
                d["__y"] = y_acum(d, nome, h)
                am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
                r = sm.OLS(am["__y"], sm.add_constant(am[regs], has_constant="add")).fit(
                    cov_type="HAC", cov_kwds=hac_kw(h))
                w = r.wald_test("choque_pre = choque_pos", scalar=True)
                linhas.append({"juros": juros, "variavel": nome, "h": h,
                               "coef_pre": r.params["choque_pre"], "p_pre": r.pvalues["choque_pre"],
                               "coef_pos": r.params["choque_pos"], "p_pos": r.pvalues["choque_pos"],
                               "diferenca": r.params["choque_pos"] - r.params["choque_pre"],
                               "wald_p": float(w.pvalue), "observacoes": int(r.nobs)})
        print(f"  regimes {juros}")
    pd.DataFrame(linhas).to_excel(SAIDA / "aux_comparacao_juros_regimes.xlsx", index=False)


def comparar():
    def par(t, val, p):
        a = t[t.juros == "antiga"].set_index(["variavel", "h"])
        c = t[t.juros == "corrigida"].set_index(["variavel", "h"])
        o = pd.DataFrame({"antiga": a[val], "corrigida": c[val],
                          "dif": c[val] - a[val], "p_antiga": a[p], "p_corrigida": c[p]}).reset_index()
        o["muda_10"] = (o.p_antiga < 0.10) != (o.p_corrigida < 0.10)
        return o
    saidas = {}
    saidas["lp_ols"] = par(pd.read_excel(SAIDA / "aux_comparacao_juros_ols.xlsx"), "coeficiente", "p_valor")
    saidas["lp_iv"] = par(pd.read_excel(SAIDA / "aux_comparacao_juros_iv.xlsx"), "coeficiente_iv", "p_valor_iv")
    reg = pd.read_excel(SAIDA / "aux_comparacao_juros_regimes.xlsx")
    a = reg[reg.juros == "antiga"].set_index(["variavel", "h"])
    c = reg[reg.juros == "corrigida"].set_index(["variavel", "h"])
    saidas["regimes"] = pd.DataFrame({
        "pos_antiga": a.coef_pos, "pos_corrigida": c.coef_pos,
        "wald_p_antiga": a.wald_p, "wald_p_corrigida": c.wald_p,
        "muda_10": (a.wald_p < 0.10) != (c.wald_p < 0.10)}).reset_index()
    with pd.ExcelWriter(SAIDA / "aux_comparacao_juros_resumo.xlsx") as w:
        for k, v in saidas.items():
            v.to_excel(w, sheet_name=k, index=False)
    for k, v in saidas.items():
        n = int(v["muda_10"].sum())
        col = "dif" if "dif" in v else None
        extra = ""
        if col:
            g = v[v[["antiga", "corrigida"]].abs().min(axis=1) > 0.1]
            extra = f" | mediana |dif %| = {(100*g['dif'].abs()/g['antiga'].abs()).median():.2f}%"
        print(f"{k:10s}: {len(v)} linhas, vereditos a 10% que mudam: {n}{extra}")
    print("\n=== horizontes do artigo, LP-OLS ===")
    v = saidas["lp_ols"]
    print(v[v.h.isin(HORIZONTES)].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print("\n=== regimes, IPCA Geral ===")
    v = saidas["regimes"]
    print(v[(v.variavel == "ipca_geral") & (v.h.isin(HORIZONTES))].to_string(
        index=False, float_format=lambda x: f"{x:8.3f}"))


def autoteste():
    """Confere as correcoes C1-C4 contra referencias independentes."""
    print("=== autoteste do modulo ===")
    b, dum, pand = carregar_pandemia("corrigida")

    # C1 -- regime sem deslocamento
    r = regime(b.Data, DATA_CORTE_FINAL)
    print(f"C1  regime pos comeca em {b.loc[r == 1, 'Data'].min():%Y-%m} "
          f"(esperado {pd.Timestamp(DATA_CORTE_FINAL):%Y-%m}) -- "
          f"{'OK' if b.loc[r == 1, 'Data'].min() == pd.Timestamp(DATA_CORTE_FINAL) else 'FALHOU'}")

    # C2 -- escala do choque nao depende do truncamento
    b6, _ = carregar("corrigida", DATA_FIM_IV)
    dif = float((b.set_index("Data")["choque_petroleo"].reindex(b6.Data).to_numpy()
                 - b6["choque_petroleo"].to_numpy())[1:].max())
    print(f"C2  desvio maximo do choque entre amostra cheia e truncada: {dif:.2e} -- "
          f"{'OK' if abs(dif) < 1e-12 else 'FALHOU'}")

    # C3 -- dummies de janela: nesting em h=0 e posto cheio em h=12
    d = b.copy()
    n0 = pand(d, 0, "janela"); j0 = d[n0].to_numpy().copy()
    n0i = pand(d, 0, "impulso"); i0 = d[n0i].to_numpy().copy()
    nj = pand(d, 12, "janela"); J = d[nj].to_numpy()
    cob = int((J.sum(axis=1) > 0).sum())
    d2 = b.copy(); pand(d2, 12, "impulso")
    print(f"C3  h=0 janela == impulso: {'OK' if np.allclose(j0, i0) else 'FALHOU'} | "
          f"h=12 posto {np.linalg.matrix_rank(J)}/10, observacoes cobertas {cob} "
          f"(impulso cobria 10) -- {'OK' if np.linalg.matrix_rank(J) == 10 and cob > 10 else 'FALHOU'}")

    # C4 -- hac_var_combo == statsmodels com use_correction
    d = b.copy()
    for lag in range(1, LAGS + 1):
        d[f"y_lag{lag}"] = d["dlog_ipca_geral"].shift(lag)
    regs = ["choque_petroleo"] + [f"y_lag{l}" for l in range(1, LAGS + 1)]
    for lag in range(1, LAGS + 1):
        regs += [f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in CONTROLES]
    regs += dum
    pior = 0.0
    for h in [0, 3, 6, 12]:
        d["__y"] = y_acum(d, "ipca_geral", h)
        am = d[["__y"] + regs].replace([np.inf, -np.inf], np.nan).dropna()
        X = sm.add_constant(am[regs], has_constant="add")
        Xv = np.asarray(X, float); yv = np.asarray(am["__y"], float)
        A = np.linalg.pinv(Xv.T @ Xv); j = list(X.columns).index("choque_petroleo")
        u = yv - Xv @ (np.linalg.pinv(Xv) @ yv)
        meu = np.sqrt(hac_var_combo(Xv, u, A[:, j], max(1, h + 1)))
        ref = float(sm.OLS(am["__y"], X).fit(cov_type="HAC", cov_kwds=hac_kw(h)).bse["choque_petroleo"])
        pior = max(pior, abs(meu / ref - 1))
    print(f"C4  hac_var_combo x statsmodels: desvio relativo maximo {pior:.2e} -- "
          f"{'OK' if pior < 1e-8 else 'FALHOU'}")

    # C5 -- os dois instrumentos de Kanzig sao objetos diferentes
    bk, _ = carregar("corrigida", DATA_FIM_IV, com_kanzig=True)
    print(f"C5  correl(news shock, dlog Brent) = "
          f"{bk[['instrumento_kanzig','choque_petroleo']].corr().iloc[0,1]:.3f} | "
          f"correl(surpresa bruta, dlog Brent) = "
          f"{bk[['instrumento_surpresa','choque_petroleo']].corr().iloc[0,1]:.3f} | "
          f"meses nao-nulos da surpresa: {int((bk.instrumento_surpresa.abs() > 1e-12).sum())}/{len(bk)}")


if __name__ == "__main__":
    {"ols": rodar_ols, "iv": rodar_iv, "regimes": rodar_regimes, "comparar": comparar,
     "autoteste": autoteste, "espec": especificacao}[
        sys.argv[1] if len(sys.argv) > 1 else "espec"]()
