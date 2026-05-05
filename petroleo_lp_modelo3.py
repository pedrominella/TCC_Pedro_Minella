# -*- coding: utf-8 -*-
"""
petroleo_lp_modelo2.py

MODELO 2 - LOCAL PROJECTIONS PARA O TCC

O que foi mudado em relação ao modelo anterior
------------------------------------------------
1. O horizonte principal foi reduzido para 12 meses.
   - O código ainda permite estimar até 24 meses, mas o foco interpretativo fica em 0 a 12 meses,
     porque horizontes muito longos em Local Projections costumam gerar intervalos de confiança maiores.

2. As variáveis de preço passam a ser usadas em variação logarítmica.
   - Petróleo, câmbio e combustíveis são transformados em Δlog.
   - Isso melhora a interpretação econômica e reduz o risco de regressões com séries em nível não estacionárias.

3. O modelo usa erros robustos HAC/Newey-West.
   - Como a Local Projection estima uma regressão para cada horizonte, os resíduos podem ter autocorrelação.
   - O erro padrão Newey-West torna os intervalos de confiança mais adequados.

4. O modelo inclui defasagens da variável dependente, do choque e dos controles.
   - Isso reduz o risco de o choque capturar apenas persistência da própria inflação ou dos combustíveis.

5. O modelo foi separado por blocos de transmissão.
   - Bloco A: petróleo -> combustíveis.
   - Bloco B: combustíveis -> IPCA Geral e IPCA Transporte.
   - Bloco C: petróleo -> IPCA com controle pelo canal dos combustíveis.
   - Bloco D: comparação por regimes da Petrobras, quando houver amostra suficiente.

6. O modelo evita especificações muito carregadas.
   - A ideia é começar com uma especificação limpa e depois usar controles adicionais como robustez.

O que contém neste modelo
-------------------------
- Leitura automática do Excel.
- Tratamento da coluna de data.
- Criação de variações logarítmicas para petróleo, câmbio, combustíveis e atividade, quando possível.
- Criação de inflação mensal para IPCA Geral e IPCA Transporte, caso as séries estejam em nível.
- Estimação de Local Projections acumuladas e não acumuladas.
- Erros robustos Newey-West.
- Gráficos individuais.
- Tabelas CSV com coeficientes, erros-padrão, intervalos de confiança e p-valores.
- Estimações para amostra completa e subamostras de regimes da Petrobras.

Observação importante
---------------------
Este código foi escrito para ser robusto a nomes próximos de colunas, mas talvez você precise ajustar
os nomes no dicionário CONFIG_COLUNAS conforme o nome exato das variáveis na sua planilha.
"""

# =============================================================================
# 0. PACOTES
# =============================================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURAÇÕES GERAIS
# =============================================================================

# Ajuste aqui o caminho do seu arquivo.
# Mantive um caminho provável com base nos seus códigos anteriores.
ARQUIVO = r"C:\Users\pedro\OneDrive\Documentos\TCC\IPCA.xlsx"
ABA = 0

# Pasta de saída.
OUTPUT_DIR = Path("output_petroleo_lp_modelo3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Horizonte máximo.
# Interprete principalmente até 12 meses.
H_MAX = 24
H_PRINCIPAL = 12

# Número de defasagens.
LAGS_Y = 3          # defasagens da variável resposta
LAGS_SHOCK = 3      # defasagens do choque
LAGS_CONTROLS = 3   # defasagens dos controles

# Intervalo de confiança.
CONF = 0.90
Z_CRIT = 1.645  # 90%

# Erro robusto Newey-West.
# Regra simples: usar o próprio horizonte como maxlags, com mínimo 1.
USAR_HAC = True

# Choque padronizado.
# Se True, o choque é multiplicado pelo desvio-padrão da variável de impulso.
# Assim a resposta é interpretada como efeito de um choque de 1 desvio-padrão.
PADRONIZAR_CHOQUE = True

# Criar dummies mensais.
USAR_DUMMIES_MENSAIS = True

# =============================================================================
# 2. NOMES DAS COLUNAS
# =============================================================================

"""
Ajuste os nomes abaixo se sua planilha estiver com nomes diferentes.

GasolinaA é a gasolina na refinaria/Petrobras.
Gasolina é a gasolina ao consumidor.
Oleo_diesel é o diesel ao consumidor.
"""

CONFIG_COLUNAS = {
    "data": ["Data", "data", "DATE", "Date"],

    "petroleo": ["Preco_Barril", "Petroleo", "Brent", "DCOILBRENTEU", "preco_barril"],
    "cambio": ["Cambio", "cambio", "USDBRL", "Dolar", "Taxa_Cambio"],

    "gasolina_refinaria": [
        "GasolinaABrasil_media", "GasolinaA", "GasolinaA_nivel",
        "Gasolina_A", "Gasolina_Refinaria", "Preco_Refinaria"
    ],
    "gasolina": ["Gasolina", "Gasolina_nivel", "Gasolina_consumidor", "Preco_Gasolina"],
    "etanol": ["Etanol", "Etanol_nivel", "Preco_Etanol"],
    "diesel": ["Oleo_diesel", "Oleo_diesel_nivel", "Diesel", "Preco_Diesel"],

    "ipca_geral": ["IPCA_Geral_nivel", "IPCA_Brasil", "Var_IPCA_Brasil", "IPCA_Geral", "IPCA"],
    "ipca_transporte": ["IPCA_Trans_nivel", "Var_IPCA_trans", "IPCA_Transporte", "IPCA_Trans"],

    "atividade": ["Atividade", "IBC_BR", "IBC_Br", "IBC-BR", "IBC"],
    "selic": ["Selic", "SELIC", "Meta_Selic", "selic"],
    "expectativa": ["Expectativa_Inflacao", "Focus_IPCA_12m", "IPCA_Focus_12m", "Expectativa"],
    "stringency": ["Stringency", "stringency", "Stringency_Index", "Oxford_Stringency"]
}


# =============================================================================
# 3. FUNÇÕES AUXILIARES
# =============================================================================

def encontrar_coluna(df, candidatos, obrigatoria=False, nome_logico=""):
    """
    Encontra uma coluna no DataFrame com base em uma lista de possíveis nomes.
    """
    cols = list(df.columns)

    # busca exata
    for c in candidatos:
        if c in cols:
            return c

    # busca ignorando maiúsculas/minúsculas
    cols_lower = {str(c).lower(): c for c in cols}
    for c in candidatos:
        if str(c).lower() in cols_lower:
            return cols_lower[str(c).lower()]

    if obrigatoria:
        raise ValueError(
            f"Não encontrei a coluna obrigatória '{nome_logico}'. "
            f"Candidatos testados: {candidatos}. "
            f"Colunas disponíveis: {cols}"
        )

    return None


def safe_log_diff(s):
    """
    Calcula Δlog apenas quando os valores são positivos.
    Se houver valor zero ou negativo, retorna NaN.
    """
    s = pd.to_numeric(s, errors="coerce")
    out = np.log(s).diff()
    out[s <= 0] = np.nan
    return out


def diff_se_precisa(s):
    """
    Para IPCA:
    - Se a série parece nível/base 100, usa variação percentual aproximada via Δlog * 100.
    - Se a série parece já ser inflação mensal, mantém a própria série.
    """
    s = pd.to_numeric(s, errors="coerce")

    med = s.dropna().median()
    std = s.dropna().std()

    # Heurística:
    # séries em nível/base 100 geralmente têm mediana > 20.
    # séries de inflação mensal geralmente ficam perto de 0 ou 1.
    if med > 20:
        return 100 * safe_log_diff(s)
    else:
        return s


def criar_lags(df, var, n_lags, prefix=None):
    """
    Cria defasagens de uma variável.
    """
    if prefix is None:
        prefix = var

    lag_cols = []
    for L in range(1, n_lags + 1):
        col = f"{prefix}_lag{L}"
        df[col] = df[var].shift(L)
        lag_cols.append(col)
    return lag_cols


def preparar_base():
    """
    Lê a planilha e cria variáveis transformadas.
    """
    print("=" * 100)
    print("1) LEITURA E PREPARAÇÃO DA BASE")
    print("=" * 100)

    df = pd.read_excel(ARQUIVO, sheet_name=ABA)
    df.columns = [str(c).strip() for c in df.columns]

    col_data = encontrar_coluna(df, CONFIG_COLUNAS["data"], obrigatoria=True, nome_logico="data")
    df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
    df = df.dropna(subset=[col_data]).sort_values(col_data).reset_index(drop=True)
    df = df.rename(columns={col_data: "Data"})
    df["mes"] = df["Data"].dt.month
    df["ano"] = df["Data"].dt.year

    mapa = {}
    for nome_logico, candidatos in CONFIG_COLUNAS.items():
        if nome_logico == "data":
            continue
        mapa[nome_logico] = encontrar_coluna(df, candidatos, obrigatoria=False, nome_logico=nome_logico)

    print("\nColunas identificadas:")
    for k, v in mapa.items():
        print(f"- {k}: {v}")

    # Transformações principais
    transformadas = {}

    if mapa["petroleo"]:
        df["dln_petroleo"] = 100 * safe_log_diff(df[mapa["petroleo"]])
        transformadas["petroleo"] = "dln_petroleo"

    if mapa["cambio"]:
        df["dln_cambio"] = 100 * safe_log_diff(df[mapa["cambio"]])
        transformadas["cambio"] = "dln_cambio"

    if mapa["gasolina_refinaria"]:
        df["dln_gasolina_refinaria"] = 100 * safe_log_diff(df[mapa["gasolina_refinaria"]])
        transformadas["gasolina_refinaria"] = "dln_gasolina_refinaria"

    if mapa["gasolina"]:
        df["dln_gasolina"] = 100 * safe_log_diff(df[mapa["gasolina"]])
        transformadas["gasolina"] = "dln_gasolina"

    if mapa["etanol"]:
        df["dln_etanol"] = 100 * safe_log_diff(df[mapa["etanol"]])
        transformadas["etanol"] = "dln_etanol"

    if mapa["diesel"]:
        df["dln_diesel"] = 100 * safe_log_diff(df[mapa["diesel"]])
        transformadas["diesel"] = "dln_diesel"

    if mapa["atividade"]:
        df["dln_atividade"] = 100 * safe_log_diff(df[mapa["atividade"]])
        transformadas["atividade"] = "dln_atividade"

    if mapa["ipca_geral"]:
        df["ipca_geral_mensal"] = diff_se_precisa(df[mapa["ipca_geral"]])
        transformadas["ipca_geral"] = "ipca_geral_mensal"

    if mapa["ipca_transporte"]:
        df["ipca_transporte_mensal"] = diff_se_precisa(df[mapa["ipca_transporte"]])
        transformadas["ipca_transporte"] = "ipca_transporte_mensal"

    if mapa["selic"]:
        # Selic pode estar em taxa, nível base 100 ou meta mensal.
        # Aqui entra em nível, não em log, porque taxa de juros pode ter zero ou comportamento específico.
        df["selic_controle"] = pd.to_numeric(df[mapa["selic"]], errors="coerce")
        transformadas["selic"] = "selic_controle"

    if mapa["expectativa"]:
        df["expectativa_controle"] = pd.to_numeric(df[mapa["expectativa"]], errors="coerce")
        transformadas["expectativa"] = "expectativa_controle"

    if mapa["stringency"]:
        df["stringency_controle"] = pd.to_numeric(df[mapa["stringency"]], errors="coerce")
        transformadas["stringency"] = "stringency_controle"

    # Dummies mensais
    dummy_cols = []
    if USAR_DUMMIES_MENSAIS:
        dummies = pd.get_dummies(df["mes"], prefix="mes", drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        dummy_cols = list(dummies.columns)

    print("\nVariáveis transformadas criadas:")
    for k, v in transformadas.items():
        print(f"- {k}: {v}")

    return df, transformadas, dummy_cols


def local_projection(
    df,
    y,
    shock,
    controls=None,
    h_max=24,
    acumulada=True,
    nome_modelo="modelo",
    subpasta=None
):
    """
    Estima Local Projections para uma variável resposta y e um choque shock.

    y: variável dependente em t+h.
    shock: variável de impulso em t.
    controls: lista de controles contemporâneos e defasados.
    acumulada:
        True  -> resposta acumulada de y entre t e t+h.
        False -> resposta pontual em t+h.
    """

    controls = controls or []

    base = df.copy()

    # Padronização do choque
    shock_usado = shock
    if PADRONIZAR_CHOQUE:
        sd = base[shock].std(skipna=True)
        if pd.notna(sd) and sd > 0:
            shock_usado = f"{shock}_std"
            base[shock_usado] = base[shock] / sd

    regressores_fixos = []

    # Defasagens da resposta
    regressores_fixos += criar_lags(base, y, LAGS_Y, prefix=y)

    # Defasagens do choque
    regressores_fixos += criar_lags(base, shock_usado, LAGS_SHOCK, prefix=shock_usado)

    # Controles contemporâneos e defasados
    for c in controls:
        if c in base.columns:
            regressores_fixos.append(c)
            regressores_fixos += criar_lags(base, c, LAGS_CONTROLS, prefix=c)

    resultados = []

    for h in range(0, h_max + 1):
        temp = base.copy()

        if acumulada:
            # Soma acumulada de y de t até t+h.
            cols_futuras = []
            for j in range(0, h + 1):
                col_fut = f"{y}_lead{j}"
                temp[col_fut] = temp[y].shift(-j)
                cols_futuras.append(col_fut)
            temp[f"y_h{h}"] = temp[cols_futuras].sum(axis=1, min_count=h + 1)
        else:
            temp[f"y_h{h}"] = temp[y].shift(-h)

        X_cols = [shock_usado] + regressores_fixos
        X_cols = [c for c in X_cols if c in temp.columns]

        temp_reg = temp[[f"y_h{h}"] + X_cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp_reg) < max(40, len(X_cols) + 10):
            resultados.append({
                "h": h, "coef": np.nan, "se": np.nan, "t": np.nan,
                "pvalor": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "nobs": len(temp_reg)
            })
            continue

        Y = temp_reg[f"y_h{h}"]
        X = sm.add_constant(temp_reg[X_cols], has_constant="add")

        try:
            model = sm.OLS(Y, X)
            if USAR_HAC:
                maxlags = max(1, h)
                res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
            else:
                res = model.fit(cov_type="HC1")

            coef = res.params.get(shock_usado, np.nan)
            se = res.bse.get(shock_usado, np.nan)
            t = res.tvalues.get(shock_usado, np.nan)
            pvalor = res.pvalues.get(shock_usado, np.nan)

            resultados.append({
                "h": h,
                "coef": coef,
                "se": se,
                "t": t,
                "pvalor": pvalor,
                "ci_low": coef - Z_CRIT * se,
                "ci_high": coef + Z_CRIT * se,
                "nobs": int(res.nobs)
            })

        except Exception as e:
            print(f"Erro em {nome_modelo}, h={h}: {e}")
            resultados.append({
                "h": h, "coef": np.nan, "se": np.nan, "t": np.nan,
                "pvalor": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "nobs": len(temp_reg)
            })

    tab = pd.DataFrame(resultados)

    # Salvar tabela
    if subpasta is None:
        subpasta = OUTPUT_DIR
    else:
        subpasta = OUTPUT_DIR / subpasta
    subpasta.mkdir(parents=True, exist_ok=True)

    sufixo = "acumulada" if acumulada else "pontual"
    csv_path = subpasta / f"lp_{sufixo}_{nome_modelo}.csv"
    tab.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(tab["h"], tab["coef"], marker="o", label="Resposta estimada")
    ax.fill_between(tab["h"], tab["ci_low"], tab["ci_high"], alpha=0.2, label=f"IC {int(CONF*100)}%")
    ax.axhline(0, linewidth=1)
    ax.axvline(H_PRINCIPAL, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_title(f"LP {sufixo} - {nome_modelo}")
    ax.set_xlabel("Horizonte h, em meses")
    ax.set_ylabel("Resposta estimada")
    ax.grid(True, alpha=0.3)
    ax.legend()

    png_path = subpasta / f"lp_{sufixo}_{nome_modelo}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    return tab


def estimar_blocos(df, vars_t, dummy_cols):
    """
    Estima os blocos principais do modelo.
    """

    print("\n" + "=" * 100)
    print("2) ESTIMAÇÃO DOS BLOCOS DE LOCAL PROJECTIONS")
    print("=" * 100)

    # Controles macro básicos
    controles_macro = []
    for k in ["cambio", "atividade", "selic", "expectativa", "stringency"]:
        if k in vars_t:
            controles_macro.append(vars_t[k])

    controles_macro = controles_macro + dummy_cols

    # ----------------------------
    # Bloco A: petróleo -> combustíveis
    # ----------------------------
    combustiveis = ["gasolina_refinaria", "gasolina", "etanol", "diesel"]

    if "petroleo" in vars_t:
        for comb in combustiveis:
            if comb in vars_t:
                nome = f"Petroleo_para_{comb}"
                print(f"Estimando Bloco A: {nome}")
                local_projection(
                    df=df,
                    y=vars_t[comb],
                    shock=vars_t["petroleo"],
                    controls=controles_macro,
                    h_max=H_MAX,
                    acumulada=True,
                    nome_modelo=nome,
                    subpasta="A_petroleo_para_combustiveis"
                )
                local_projection(
                    df=df,
                    y=vars_t[comb],
                    shock=vars_t["petroleo"],
                    controls=controles_macro,
                    h_max=H_MAX,
                    acumulada=False,
                    nome_modelo=nome,
                    subpasta="A_petroleo_para_combustiveis"
                )

    # ----------------------------
    # Bloco B: combustíveis -> inflação
    # ----------------------------
    inflacoes = ["ipca_geral", "ipca_transporte"]

    for comb in combustiveis:
        if comb not in vars_t:
            continue

        for infl in inflacoes:
            if infl not in vars_t:
                continue

            controles = controles_macro.copy()
            if "petroleo" in vars_t:
                controles.append(vars_t["petroleo"])

            nome = f"{comb}_para_{infl}"
            print(f"Estimando Bloco B: {nome}")
            local_projection(
                df=df,
                y=vars_t[infl],
                shock=vars_t[comb],
                controls=controles,
                h_max=H_MAX,
                acumulada=True,
                nome_modelo=nome,
                subpasta="B_combustiveis_para_inflacao"
            )
            local_projection(
                df=df,
                y=vars_t[infl],
                shock=vars_t[comb],
                controls=controles,
                h_max=H_MAX,
                acumulada=False,
                nome_modelo=nome,
                subpasta="B_combustiveis_para_inflacao"
            )

    # ----------------------------
    # Bloco C: petróleo -> inflação com e sem controle por combustível
    # ----------------------------
    if "petroleo" in vars_t:
        for infl in inflacoes:
            if infl not in vars_t:
                continue

            # Sem canal
            nome = f"Petroleo_para_{infl}_sem_canal"
            print(f"Estimando Bloco C: {nome}")
            local_projection(
                df=df,
                y=vars_t[infl],
                shock=vars_t["petroleo"],
                controls=controles_macro,
                h_max=H_MAX,
                acumulada=True,
                nome_modelo=nome,
                subpasta="C_petroleo_para_inflacao_com_sem_canal"
            )

            # Com canal
            for comb in combustiveis:
                if comb not in vars_t:
                    continue

                controles = controles_macro + [vars_t[comb]]
                nome = f"Petroleo_para_{infl}_controle_{comb}"
                print(f"Estimando Bloco C: {nome}")
                local_projection(
                    df=df,
                    y=vars_t[infl],
                    shock=vars_t["petroleo"],
                    controls=controles,
                    h_max=H_MAX,
                    acumulada=True,
                    nome_modelo=nome,
                    subpasta="C_petroleo_para_inflacao_com_sem_canal"
                )


def estimar_regimes(df, vars_t, dummy_cols):
    """
    Estima versões por regimes de política de preços da Petrobras.
    Usa uma divisão simples para preservar graus de liberdade.
    """

    print("\n" + "=" * 100)
    print("3) ESTIMAÇÃO POR REGIMES DA PETROBRAS")
    print("=" * 100)

    regimes = {
        "2003_2014_contencao": ("2003-01-01", "2014-12-31"),
        "2015_2026_maior_alinhamento": ("2015-01-01", "2026-12-31"),
    }

    controles_macro = []
    for k in ["cambio", "atividade", "selic", "expectativa", "stringency"]:
        if k in vars_t:
            controles_macro.append(vars_t[k])
    controles_macro = controles_macro + dummy_cols

    # Para regimes, fazer poucos modelos principais para não poluir a saída.
    modelos_regime = []

    if "petroleo" in vars_t and "gasolina_refinaria" in vars_t:
        modelos_regime.append(("gasolina_refinaria", "petroleo", "Petroleo_para_GasolinaA"))

    if "petroleo" in vars_t and "gasolina" in vars_t:
        modelos_regime.append(("gasolina", "petroleo", "Petroleo_para_Gasolina"))

    if "ipca_transporte" in vars_t and "gasolina" in vars_t:
        modelos_regime.append(("ipca_transporte", "gasolina", "Gasolina_para_IPCA_Transporte"))

    if "ipca_transporte" in vars_t and "gasolina_refinaria" in vars_t:
        modelos_regime.append(("ipca_transporte", "gasolina_refinaria", "GasolinaA_para_IPCA_Transporte"))

    for nome_regime, (inicio, fim) in regimes.items():
        df_r = df[(df["Data"] >= pd.to_datetime(inicio)) & (df["Data"] <= pd.to_datetime(fim))].copy()

        if len(df_r) < 80:
            print(f"Regime {nome_regime} ignorado por poucas observações: {len(df_r)}")
            continue

        print(f"\nRegime: {nome_regime} | Observações brutas: {len(df_r)}")

        for y_key, shock_key, nome_base in modelos_regime:
            y = vars_t[y_key]
            shock = vars_t[shock_key]

            controles = controles_macro.copy()
            # Se o choque for combustível, controlar por petróleo.
            if shock_key != "petroleo" and "petroleo" in vars_t:
                controles.append(vars_t["petroleo"])

            nome = f"{nome_base}_{nome_regime}"
            print(f"Estimando regime: {nome}")
            local_projection(
                df=df_r,
                y=y,
                shock=shock,
                controls=controles,
                h_max=H_PRINCIPAL,
                acumulada=True,
                nome_modelo=nome,
                subpasta=f"D_regimes/{nome_regime}"
            )


def criar_resumo_resultados():
    """
    Junta os CSVs gerados em uma tabela única de resumo.
    """
    print("\n" + "=" * 100)
    print("4) CRIANDO RESUMO DOS RESULTADOS")
    print("=" * 100)

    arquivos = list(OUTPUT_DIR.rglob("*.csv"))
    linhas = []

    for arq in arquivos:
        try:
            tab = pd.read_csv(arq)
            for h_ref in [3, 6, 12, 24]:
                if h_ref in tab["h"].values:
                    row = tab.loc[tab["h"] == h_ref].iloc[0].to_dict()
                    row["arquivo"] = str(arq)
                    row["h_ref"] = h_ref
                    linhas.append(row)
        except Exception:
            pass

    if linhas:
        resumo = pd.DataFrame(linhas)
        resumo_path = OUTPUT_DIR / "resumo_resultados_h3_h6_h12_h24.csv"
        resumo.to_csv(resumo_path, index=False, encoding="utf-8-sig")
        print(f"Resumo salvo em: {resumo_path}")
    else:
        print("Nenhum resultado encontrado para resumir.")


# =============================================================================
# 4. EXECUÇÃO
# =============================================================================

def main():
    df, vars_t, dummy_cols = preparar_base()

    # Checagem mínima
    obrigatorias = ["petroleo"]
    for ob in obrigatorias:
        if ob not in vars_t:
            raise ValueError(f"Variável obrigatória ausente: {ob}")

    estimar_blocos(df, vars_t, dummy_cols)
    estimar_regimes(df, vars_t, dummy_cols)
    criar_resumo_resultados()

    print("\n" + "=" * 100)
    print("MODELO FINALIZADO")
    print("=" * 100)
    print(f"Arquivos salvos em: {OUTPUT_DIR.resolve()}")
    print("\nInterprete principalmente os horizontes até 12 meses.")
    print("Use os resultados de 24 meses apenas como robustez, pois a incerteza tende a ser maior.")


if __name__ == "__main__":
    main()
