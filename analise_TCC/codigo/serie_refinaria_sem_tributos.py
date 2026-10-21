# -*- coding: utf-8 -*-
"""
SERIE DE GASOLINA A NA REFINARIA SEM TRIBUTOS FEDERAIS (01/10/2026)

A serie semanal da ANP (precos medios ponderados de produtores e importadores) inclui,
segundo a propria ANP, as parcelas de Cide, PIS/Pasep e Cofins, e exclui o ICMS. Mudancas
desses tributos entram, portanto, como se fossem mudancas do preco praticado. Este script
subtrai o tributo federal vigente em cada dia, conforme o calendario documentado em
base_de_dados/brasil/BR_tributos_federais_gasolina.csv (cada linha com o ato legal).

Passos:
  1. le as duas planilhas semanais da ANP (2002-2012 e desde 2013), Gasolina A / Gasolina A
     Comum, coluna Brasil, R$/litro;
  2. tributo da semana = media diaria do total (R$/m3) nos dias da semana, dividido por 1000;
  3. preco sem tributo = preco da ANP - tributo da semana;
  4. agrega ao mes pela media das semanas segundo o mes do dia inicial (mesma regra da serie
     com tributos), e confere que a serie com tributos reproduz a coluna ja em uso;
  5. grava as colunas novas em base_de_dados/brasil/BR_gasolinaA_anp_mensal.csv.

Uso: python serie_refinaria_sem_tributos.py
"""
import numpy as np, pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
P = RAIZ / "base_de_dados" / "brasil"
ARQS = [P / "anp_2002_2012.xls", P / "anp_2013.xls"]
PRODUTOS = ["Gasolina A (R$/litro)", "Gasolina A Comum (R$/litro)"]
CALENDARIO = P / "BR_tributos_federais_gasolina.csv"
MENSAL = P / "BR_gasolinaA_anp_mensal.csv"
COLS = ["Norte", "Nordeste", "Centro-Oeste", "Sul", "Sudeste", "Brasil"]


def semanal():
    partes = []
    for f in ARQS:
        d = pd.read_excel(f, header=None).iloc[9:, :9]
        d.columns = ["produto", "ini", "fim"] + COLS
        d["ini"] = pd.to_datetime(d["ini"], errors="coerce")
        d["fim"] = pd.to_datetime(d["fim"], errors="coerce", dayfirst=True)
        d = d.dropna(subset=["ini"])
        d["produto"] = d["produto"].astype(str).str.strip()
        d["Brasil"] = pd.to_numeric(d["Brasil"], errors="coerce")
        partes.append(d[d.produto.isin(PRODUTOS)])
    w = pd.concat(partes).sort_values("ini").reset_index(drop=True)
    w["fim"] = w["fim"].fillna(w["ini"] + pd.Timedelta(days=6))
    return w.dropna(subset=["Brasil"])


def tributo_diario():
    c = pd.read_csv(CALENDARIO, parse_dates=["inicio", "fim"])
    dias = pd.date_range(c.inicio.min(), c.fim.max())
    s = pd.Series(np.nan, index=dias)
    for r in c.itertuples():
        s[r.inicio:r.fim] = r._5                       # total_R$_m3
    if s.isna().any():
        raise ValueError(f"dias sem tributo no calendario: {s[s.isna()].index[:5].tolist()}")
    if not np.allclose(c["cide_R$_m3"] + c["pis_cofins_liquido_R$_m3"], c["total_R$_m3"]):
        raise ValueError("total diferente de Cide + PIS/Cofins em alguma linha")
    return s


def construir():
    w = semanal()
    td = tributo_diario()
    w["tributo_R$_litro"] = [td[r.ini:r.fim].mean() / 1000 for r in w.itertuples()]
    w["sem_tributo"] = w["Brasil"] - w["tributo_R$_litro"]
    w["mes"] = w["ini"].dt.to_period("M").dt.to_timestamp()
    m = w.groupby("mes").agg(com=("Brasil", "mean"), sem=("sem_tributo", "mean"),
                             trib=("tributo_R$_litro", "mean"))
    base = pd.read_csv(MENSAL, parse_dates=["mes"]).set_index("mes")
    dif = (base["gasolina_A_anp_mes_inicio_semana"] - m["com"]).abs().max()
    if dif > 1e-9:
        raise ValueError(f"a serie com tributos nao reproduz a coluna em uso (dif max {dif})")
    base["tributo_federal_medio_R$_litro"] = m["trib"]
    base["gasolina_A_anp_sem_tributos_mes_inicio_semana"] = m["sem"]
    # agregacao por calendario: cada dia recebe o preco sem tributo da semana que o contem e o
    # mes e a media dos seus dias, alinhada ao Brent e ao IPCA, que sao medias do mes civil.
    # A regra "mes do dia inicial" adianta a serie em ate 6 dias (semana que comeca no fim do
    # mes leva dias do mes seguinte).
    dias = pd.DataFrame([(dd, r.sem_tributo, r.Brasil) for r in w.itertuples()
                         for dd in pd.date_range(r.ini, r.fim)], columns=["dia", "sem", "com"])
    dias = dias.drop_duplicates("dia", keep="last")
    cal = dias.groupby(dias.dia.dt.to_period("M").dt.to_timestamp())[["sem", "com"]].mean()
    dif_cal = (base["gasolina_A_refinaria_anp"] - cal["com"]).abs().max()
    if dif_cal > 1e-9:
        raise ValueError(f"a agregacao por calendario nao reproduz gasolina_A_refinaria_anp (dif {dif_cal})")
    base["gasolina_A_anp_sem_tributos_calendario"] = cal["sem"]
    base.reset_index().to_csv(MENSAL, index=False)
    print(f"serie com tributos reproduzida (dif max {dif:.1e}); colunas novas gravadas em {MENSAL.name}")
    return w, base


if __name__ == "__main__":
    construir()
