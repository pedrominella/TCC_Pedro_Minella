# -*- coding: utf-8 -*-
"""
PLACEBO DA REFINARIA COM PRECOS DE FIM DE MES (01/10/2026)

Questao. Com a serie de refinaria da ANP sem tributos, o placebo em t-1 (variacao do preco no
mes anterior ao choque) e significativo. Duas explicacoes concorrentes:
  (a) efeito mecanico de medias mensais (Working, 1960): Brent e preco sao medias de dados
      diarios; um movimento do Brent no fim do mes t-1 entra na media do preco em t-1 (se o
      preco reage em dias) e na variacao da media do Brent em t;
  (b) reacao do preco a movimentos do Brent anteriores ao mes do choque.
Com precos de fim de mes (pontuais), (a) desaparece por construcao; (b) nao.

Versoes, todas com o Brent diario do FRED (DCOILBRENTEU) para comparar igual com igual:
  media/media : choque = variacao da media mensal do Brent; refinaria = media do mes civil
  fim/fim     : choque = variacao do Brent no ultimo dia util do mes; refinaria = preco da
                semana que contem o ultimo dia do mes
  e as duas combinacoes cruzadas.
A especificacao e a do placebo do trabalho (onda2_bandas_placebo): amostra completa, choque em
t, defasagens a partir de t-k-1, sazonais e indicadora de pandemia do mes da dependente,
erro-padrao HAC. Escala do choque: desvio-padrao na janela de referencia, como no trabalho.

Uso: python teste_placebo_fim_de_mes.py
"""
import warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")
import lp_base_nova as L
import onda2_bandas_placebo as PB
import serie_refinaria_sem_tributos as S

PASTA = Path(__file__).resolve().parent.parent
BRENT_DIARIO = PASTA.parent / "base_de_dados" / "brent_diario_fred.csv"
NOME = "gasolina_refinaria"


def brent_mensal():
    f = pd.read_csv(BRENT_DIARIO); f.columns = ["data", "brent"]
    f["data"] = pd.to_datetime(f["data"]); f["brent"] = pd.to_numeric(f["brent"], errors="coerce")
    f = f.dropna(); f["mes"] = f.data.dt.to_period("M").dt.to_timestamp()
    return f.groupby("mes").brent.mean(), f.sort_values("data").groupby("mes").brent.last()


def refinaria_fim_de_mes():
    w = S.semanal(); td = S.tributo_diario()
    w["semt"] = w.Brasil - [td[r.ini:r.fim].mean() / 1000 for r in w.itertuples()]
    fins = pd.date_range("2002-01-31", "2025-12-31", freq="ME")
    out = {}
    for f in fins:
        x = w[(w.ini <= f) & (w.fim >= f)]
        if len(x): out[f.to_period("M").to_timestamp()] = float(x["semt"].iloc[-1])
    return pd.Series(out)


def estimar(b, dum, choque, ln_ref, com_choque_mes_dependente=False):
    """placebos k=1..3 e respostas h=0..3 da refinaria, especificacao do placebo do trabalho.

    com_choque_mes_dependente=True acrescenta ao placebo k o choque do proprio mes da
    dependente, choque(t-k), que o desenho do trabalho deixa de fora ao deslocar todo o conjunto
    de condicionamento para antes de t-k. Com choques autocorrelacionados, a omissao faz o
    coeficiente de choque(t) absorver parte do efeito de choque(t-k).
    """
    d = b.copy()
    d["choque_petroleo"] = d["Data"].map(choque)
    d[f"ln_{NOME}"] = d["Data"].map(ln_ref)
    d[f"dlog_{NOME}"] = d[f"ln_{NOME}"].diff()
    kmax = L.LAGS + max(PB.PLACEBOS)
    for lag in range(1, kmax + 1):
        d[f"y_lag{lag}"] = d[f"dlog_{NOME}"].shift(lag)
        d[f"choque_petroleo_lag{lag}"] = d["choque_petroleo"].shift(lag)
        for c in L.CONTROLES:
            d[f"{c}_lag{lag}"] = d[c].shift(lag)

    def defas(desloc):
        r = []
        for lag in range(desloc + 1, desloc + L.LAGS + 1):
            r += [f"y_lag{lag}", f"choque_petroleo_lag{lag}"] + [f"{c}_lag{lag}" for c in L.CONTROLES]
        return r
    res = []
    mi = d["Data"].dt.year * 12 + d["Data"].dt.month
    for k in PB.PLACEBOS:
        d["__y"] = d[f"dlog_{NOME}"].shift(k)
        pk = []
        for s in L.PANDEMIA:
            c = f"pand_{s:%Y%m}"; d[c] = (mi - k == s.year * 12 + s.month).astype(float); pk.append(c)
        regs = ["choque_petroleo"] + ([f"choque_petroleo_lag{k}"] if com_choque_mes_dependente else []) + defas(k) + dum + pk
        g = PB.ajustar(PB._amostra(d, regs), regs, 0, "choque_petroleo")
        res.append(("placebo", -k, g["valor"], g["ep"], g["nobs"]))
    for h in range(0, 4):
        d["__y"] = L.y_acum(d, NOME, h)
        regs = ["choque_petroleo"] + defas(0) + dum + L.controles_pandemia(d, h)
        g = PB.ajustar(PB._amostra(d, regs), regs, h, "choque_petroleo")
        res.append(("resposta", h, g["valor"], g["ep"], g["nobs"]))
    t = pd.DataFrame(res, columns=["tipo", "h", "coeficiente", "erro_padrao", "obs"])
    t["p_valor"] = 2 * stats.norm.sf(np.abs(t.coeficiente / t.erro_padrao))
    return t


def main():
    b, dum, pand = L.carregar_pandemia("corrigida")
    janela = (b.Data >= L.DATA_INICIO) & (b.Data <= L.DATA_FIM)
    media, fim = brent_mensal()
    choques = {}
    for rot, s in [("media", media), ("fim", fim)]:
        dl = 100 * np.log(s).diff()
        choques[rot] = dl / dl[b.Data[janela]].std()
    ref_media = 100 * np.log(b.set_index("Data")["GasolinaABrasil_media_nivel"])
    ref_fim = 100 * np.log(refinaria_fim_de_mes())
    # checagem: com o choque da base, a versao media/media reproduz o placebo do trabalho
    base_chk = estimar(b, dum, b.set_index("Data")["choque_petroleo"], ref_media)
    ref_pl = pd.read_excel(PASTA / "resultados" / "onda2_bandas_placebo_final.xlsx", sheet_name="bandas")
    ref_pl = ref_pl[(ref_pl.variavel == NOME) & (ref_pl.tipo == "placebo")].set_index("h").coeficiente
    dif = (base_chk[base_chk.tipo == "placebo"].set_index("h").coeficiente - ref_pl).abs().max()
    print(f"checagem contra onda2_bandas_placebo (choque da base, media/media): dif max {dif:.2e}")
    for rot, ch in [("base", b.set_index("Data")["choque_petroleo"]), ("media FRED", choques["media"]), ("fim FRED", choques["fim"])]:
        print(f"autocorrelacao de ordem 1 do choque ({rot}): {ch.loc[L.DATA_INICIO:L.DATA_FIM].autocorr(1):+.3f}")
    linhas = []
    for rot_c, rot_r, ch, rf in [("media", "media", choques["media"], ref_media), ("fim", "fim", choques["fim"], ref_fim),
                                 ("media", "fim", choques["media"], ref_fim), ("fim", "media", choques["fim"], ref_media)]:
        for ctrl in (False, True):
            t = estimar(b, dum, ch, rf, ctrl); t.insert(0, "choque", rot_c); t.insert(1, "refinaria", rot_r)
            t.insert(2, "controla_choque_do_mes_da_dependente", ctrl); linhas.append(t)
    t = pd.concat(linhas, ignore_index=True)
    t.to_excel(PASTA / "resultados" / "teste_placebo_fim_de_mes.xlsx", index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
