# ======================================================================================
# LOCAL PROJECTIONS ACUMULADO - MODELO TOP
# Tema: Petróleo, combustíveis e inflação brasileira
# Autor: Pedro Franck
# ======================================================================================
# ======================================================================================
# RESUMO DO CÓDIGO
# ======================================================================================

# 1. Lê e organiza a base mensal do TCC a partir de 2003.

# 2. Padroniza os nomes das colunas e cria dummies mensais para sazonalidade.

# 3. Cria uma dummy de regime Petrobras, igual a 1 a partir de 2015.

# 4. Transforma petróleo, câmbio, combustíveis e atividade em diferença logarítmica.

# 5. Mantém IPCA Geral e IPCA Transporte como variações mensais,
#    pois o LP acumulado deve somar taxas, não índices em nível.

# 6. Cria choques residuais padronizados para petróleo e combustíveis:
#    resíduo da equação auxiliar dividido pelo desvio-padrão.

# 7. Estima Local Projections acumulado de 0 a 24 meses.

# 8. A variável dependente acumulada é:
#    y_t + y_{t+1} + ... + y_{t+h}.

# 9. Inclui defasagens da variável dependente e dos controles macroeconômicos.

# 10. Usa lag augmentation, adicionando uma defasagem extra ao modelo.

# 11. Usa erros-padrão HAC/Newey-West para corrigir autocorrelação
#     e heterocedasticidade nos resíduos.

# 12. Estima combustíveis → IPCA Geral e IPCA Transporte.

# 13. Estima petróleo → combustíveis.

# 14. Estima petróleo → IPCA como efeito total, sem controlar pelos combustíveis.

# 15. Estima petróleo → IPCA como efeito direto, controlando por cada combustível.

# 16. Roda os modelos para 2003–2026, 2003–2014 e 2015–2026.

# 17. Estima interação entre choque do petróleo e regime pós-2015.

# 18. Gera gráficos das respostas acumuladas e tabela-resumo dos resultados.

# 19. O horizonte de 12 meses é o principal; 24 meses é usado como robustez.

# ======================================================================================
# INTERPRETAÇÃO ECONÔMICA
# ======================================================================================

# A cadeia testada é:
# Petróleo → Petrobras/refinaria → combustíveis → IPCA Transporte → IPCA Geral

# Se combustíveis afetam mais o IPCA Transporte do que o IPCA Geral,
# o repasse é concentrado no grupo de transportes.

# Se o efeito total do petróleo for maior que o efeito direto,
# o canal dos combustíveis é relevante.

# Se o efeito direto some após controlar pelos combustíveis,
# o petróleo afeta a inflação principalmente via combustíveis.

# Se o repasse for maior após 2015,
# pode haver maior alinhamento dos preços domésticos aos preços internacionais.

# Se a banda de confiança cruza zero,
# o efeito deve ser interpretado com cautela.

# Se o efeito aparece até 12 meses e perde força depois,
# o repasse é concentrado no curto e médio prazo.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ======================================================================================
# 1) CONFIGURAÇÕES GERAIS
# ======================================================================================

ARQUIVO = r"C:\Users\pedro\OneDrive\Documentos\TCC\IPCA.xlsx"
ABA = 0

PASTA_OUTPUT = "output_lp_modelo2"
os.makedirs(PASTA_OUTPUT, exist_ok=True)

# Horizonte principal e robustez
H_MAIN = 12
H_ROBUST = 24

# Defasagens principais
P_LAGS = 6

# Lag augmentation: uma defasagem extra para robustez da inferência
P_AUG = P_LAGS + 1

# Banda de confiança
# 1.645 = 90%
# 1.96 = 95%
# 1.00 = aproximadamente 68%
Z_CONF = 1.645

# Regimes Petrobras
REGIMES = {
    "2003_2026": ("2003-01-01", "2026-12-31"),
    "2003_2014": ("2003-01-01", "2014-12-31"),
    "2015_2026": ("2015-01-01", "2026-12-31"),
}

# ======================================================================================
# 2) FUNÇÕES AUXILIARES
# ======================================================================================

def limpar_nome_colunas(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def achar_coluna(df, candidatos, obrigatoria=True):
    """
    Procura uma coluna no dataframe a partir de uma lista de nomes possíveis.
    """
    for c in candidatos:
        if c in df.columns:
            return c

    if obrigatoria:
        raise ValueError(
            f"Não encontrei nenhuma das colunas esperadas: {candidatos}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    return None


def criar_log_diff(df, coluna, novo_nome):
    """
    Cria diferença do log: dln_x = log(x_t) - log(x_{t-1}).
    Só funciona corretamente para séries positivas.
    """
    df = df.copy()

    if coluna not in df.columns:
        print(f"[AVISO] Coluna {coluna} não encontrada. Não criei {novo_nome}.")
        return df

    serie = pd.to_numeric(df[coluna], errors="coerce")

    if (serie.dropna() <= 0).any():
        print(f"[AVISO] Coluna {coluna} tem valores <= 0. Não apliquei log.")
        df[novo_nome] = np.nan
    else:
        df[novo_nome] = np.log(serie).diff()

    return df


def criar_lags(df, variaveis, p):
    """
    Cria defasagens de 1 até p para cada variável.
    """
    df = df.copy()

    for var in variaveis:
        if var in df.columns:
            for lag in range(1, p + 1):
                df[f"{var}_lag{lag}"] = df[var].shift(lag)

    return df


def criar_acumulado_futuro(df, var_y, h, nome_y_acum):
    """
    Cria a variável dependente acumulada:
    y_t + y_{t+1} + ... + y_{t+h}

    Isso deve ser usado com variáveis de variação/taxa, como inflação mensal.
    Não use com índice em nível/base 100.
    """
    df = df.copy()
    acumulado = pd.Series(0.0, index=df.index)

    for j in range(0, h + 1):
        acumulado = acumulado + df[var_y].shift(-j)

    df[nome_y_acum] = acumulado
    return df


def estimar_choque_residualizado(
    df,
    var_base,
    nome_choque,
    controles=None,
    p_lags=6
):
    """
    Estima um choque residualizado e padronizado.

    Exemplo:
    dln_petroleo_t = alfa + lags(dln_petroleo) + controles + erro_t

    Depois:
    choque_t = erro_t / desvio_padrao(erro_t)

    Isso transforma a variação observada em um componente inesperado.
    """
    df = df.copy()

    if controles is None:
        controles = []

    if var_base not in df.columns:
        print(f"[AVISO] Variável {var_base} não encontrada. Não criei {nome_choque}.")
        df[nome_choque] = np.nan
        return df

    # Lags da própria variável
    for lag in range(1, p_lags + 1):
        df[f"{var_base}_shock_lag{lag}"] = df[var_base].shift(lag)

    x_vars = [f"{var_base}_shock_lag{lag}" for lag in range(1, p_lags + 1)]

    # Controles contemporâneos simples
    for c in controles:
        if c in df.columns:
            x_vars.append(c)

    # Dummies mensais
    dummies_meses = [col for col in df.columns if col.startswith("mes_")]
    x_vars += dummies_meses

    dados = df[[var_base] + x_vars].dropna()

    if len(dados) < 40:
        print(f"[AVISO] Poucas observações para criar choque {nome_choque}.")
        df[nome_choque] = np.nan
        return df

    y = dados[var_base]
    X = sm.add_constant(dados[x_vars], has_constant="add")

    modelo = sm.OLS(y, X).fit()
    residuos = modelo.resid

    choque = residuos / residuos.std()

    df[nome_choque] = np.nan
    df.loc[choque.index, nome_choque] = choque

    print(f"Choque criado: {nome_choque} | R² da equação auxiliar: {modelo.rsquared:.3f}")

    return df


def rodar_lp_acumulado(
    df,
    y_var,
    shock_var,
    nome_modelo,
    controles_lagados=None,
    controles_contemporaneos=None,
    p_aug=7,
    h_max=24,
    z_conf=1.645,
    interaction_regime=False
):
    """
    Roda Local Projections acumulado para h = 0 até h_max.

    Modelo base:
    y_acum_{t,h} = alfa_h + beta_h choque_t
                   + lags(y)
                   + lags(controles)
                   + controles contemporâneos
                   + erro_{t+h}

    Erro-padrão:
    HAC/Newey-West com maxlags = max(h+1, p_aug)

    Se interaction_regime=True:
    inclui choque_t * regime_pos_2015.
    """

    if controles_lagados is None:
        controles_lagados = []

    if controles_contemporaneos is None:
        controles_contemporaneos = []

    resultados = []

    df_model = df.copy()

    # Criar lags da resposta e dos controles
    vars_para_lags = [y_var] + controles_lagados
    df_model = criar_lags(df_model, vars_para_lags, p_aug)

    for h in range(0, h_max + 1):

        nome_y_acum = f"{y_var}_acum_h{h}"
        df_model = criar_acumulado_futuro(df_model, y_var, h, nome_y_acum)

        x_vars = [shock_var]

        # Interação de regime
        if interaction_regime:
            if "regime_pos_2015" in df_model.columns:
                df_model[f"{shock_var}_x_regime"] = df_model[shock_var] * df_model["regime_pos_2015"]
                x_vars.append(f"{shock_var}_x_regime")

        # Lags da variável resposta
        for lag in range(1, p_aug + 1):
            lag_name = f"{y_var}_lag{lag}"
            if lag_name in df_model.columns:
                x_vars.append(lag_name)

        # Lags dos controles
        for var in controles_lagados:
            for lag in range(1, p_aug + 1):
                lag_name = f"{var}_lag{lag}"
                if lag_name in df_model.columns:
                    x_vars.append(lag_name)

        # Controles contemporâneos/exógenos
        for var in controles_contemporaneos:
            if var in df_model.columns:
                x_vars.append(var)

        # Dummies mensais
        dummies_meses = [col for col in df_model.columns if col.startswith("mes_")]
        x_vars += dummies_meses

        dados = df_model[[nome_y_acum] + x_vars].dropna()

        if len(dados) < 50:
            resultados.append({
                "modelo": nome_modelo,
                "h": h,
                "coef": np.nan,
                "se": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "nobs": len(dados)
            })
            continue

        y = dados[nome_y_acum]
        X = sm.add_constant(dados[x_vars], has_constant="add")

        maxlags_hac = max(h + 1, p_aug)

        modelo = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags_hac}
        )

        beta = modelo.params.get(shock_var, np.nan)
        se = modelo.bse.get(shock_var, np.nan)

        lower = beta - z_conf * se
        upper = beta + z_conf * se

        resultados.append({
            "modelo": nome_modelo,
            "h": h,
            "coef": beta,
            "se": se,
            "lower": lower,
            "upper": upper,
            "nobs": int(modelo.nobs)
        })

    return pd.DataFrame(resultados)


def plotar_lp(resultado, titulo, caminho):
    """
    Plota a resposta acumulada do LP.
    """
    r = resultado.copy()
    r = r.dropna(subset=["coef", "lower", "upper"])

    plt.figure(figsize=(12, 6))
    plt.fill_between(r["h"], r["lower"], r["upper"], alpha=0.2)
    plt.plot(r["h"], r["coef"], marker="o", linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.axvline(12, linestyle="--", linewidth=1, alpha=0.6)

    plt.title(titulo, fontsize=16)
    plt.xlabel("Horizonte h, em meses")
    plt.ylabel("Resposta acumulada estimada")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()


def criar_tabela_resumo(resultados):
    """
    Cria tabela-resumo por modelo:
    - pico da resposta acumulada
    - horizonte do pico
    - efeito em 6, 12 e 24 meses
    - significância em 12 meses
    - se a banda cruza zero
    """
    linhas = []

    for modelo, grupo in resultados.groupby("modelo"):
        g = grupo.dropna(subset=["coef"]).copy()

        if g.empty:
            continue

        # Pico em valor absoluto
        idx_pico_abs = g["coef"].abs().idxmax()
        linha_pico = g.loc[idx_pico_abs]

        def pegar_coef_h(h):
            linha = g[g["h"] == h]
            if linha.empty:
                return np.nan
            return float(linha["coef"].iloc[0])

        def significancia_h(h):
            linha = g[g["h"] == h]
            if linha.empty:
                return "NA"

            low = float(linha["lower"].iloc[0])
            up = float(linha["upper"].iloc[0])

            if low > 0:
                return "positivo_significativo"
            elif up < 0:
                return "negativo_significativo"
            else:
                return "nao_significativo"

        linhas.append({
            "modelo": modelo,
            "pico_abs_coef": float(linha_pico["coef"]),
            "h_pico_abs": int(linha_pico["h"]),
            "coef_h6": pegar_coef_h(6),
            "coef_h12": pegar_coef_h(12),
            "coef_h24": pegar_coef_h(24),
            "significancia_h6": significancia_h(6),
            "significancia_h12": significancia_h(12),
            "significancia_h24": significancia_h(24),
            "nobs_min": int(g["nobs"].min()),
            "nobs_max": int(g["nobs"].max())
        })

    return pd.DataFrame(linhas)


# ======================================================================================
# 3) LEITURA E PREPARAÇÃO DA BASE
# ======================================================================================

print("=" * 100)
print("1) LEITURA DA BASE")
print("=" * 100)

df = pd.read_excel(ARQUIVO, sheet_name=ABA)
df = limpar_nome_colunas(df)

col_data = achar_coluna(df, ["Data", "data", "DATE", "Date"])
df[col_data] = pd.to_datetime(df[col_data])
df = df.sort_values(col_data).reset_index(drop=True)

# Renomear colunas principais para nomes padronizados
col_ipca_geral = achar_coluna(df, ["Var_IPCA_Brasil", "IPCA_Geral", "IPCA_Brasil", "Var_IPCA_Geral"])
col_ipca_trans = achar_coluna(df, ["Var_IPCA_trans", "Var_IPCA_Trans", "IPCA_Transporte", "IPCA_Trans"])

col_petroleo = achar_coluna(df, ["Preco_Barril", "Brent", "Petroleo", "Preco_petroleo"])
col_cambio = achar_coluna(df, ["Cambio", "Câmbio", "USD_BRL"])
col_atividade = achar_coluna(df, ["Atividade", "IBC_BR", "IBC_Br"], obrigatoria=False)
col_selic = achar_coluna(df, ["Selic", "SELIC"], obrigatoria=False)
col_focus = achar_coluna(df, ["Expectativa_Inflacao", "Focus_IPCA_12m", "Focus_Inflacao"], obrigatoria=False)
col_stringency = achar_coluna(df, ["Stringency", "stringency", "Stringency_Index"], obrigatoria=False)

col_gasolinaA = achar_coluna(df, ["GasolinaABrasil_media", "GasolinaA", "Gasolina_Refinaria"], obrigatoria=False)
col_gasolina = achar_coluna(df, ["Gasolina", "Gasolina_Consumidor"], obrigatoria=False)
col_etanol = achar_coluna(df, ["Etanol", "Etanol_Consumidor"], obrigatoria=False)
col_diesel = achar_coluna(df, ["Oleo_diesel", "Diesel", "Oleo_Diesel"], obrigatoria=False)

df = df.rename(columns={
    col_data: "Data",
    col_ipca_geral: "IPCA_Geral",
    col_ipca_trans: "IPCA_Transporte",
    col_petroleo: "Petroleo",
    col_cambio: "Cambio"
})

if col_atividade:
    df = df.rename(columns={col_atividade: "Atividade"})
if col_selic:
    df = df.rename(columns={col_selic: "Selic"})
if col_focus:
    df = df.rename(columns={col_focus: "Expectativa_Inflacao"})
if col_stringency:
    df = df.rename(columns={col_stringency: "Stringency"})
if col_gasolinaA:
    df = df.rename(columns={col_gasolinaA: "GasolinaA"})
if col_gasolina:
    df = df.rename(columns={col_gasolina: "Gasolina"})
if col_etanol:
    df = df.rename(columns={col_etanol: "Etanol"})
if col_diesel:
    df = df.rename(columns={col_diesel: "Oleo_diesel"})

# Converter para numérico
for c in df.columns:
    if c != "Data":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Cortar amostra a partir de 2003
df = df[df["Data"] >= "2003-01-01"].copy()

# Dummies mensais
df["mes"] = df["Data"].dt.month
mes_dummies = pd.get_dummies(df["mes"], prefix="mes", drop_first=True).astype(int)
df = pd.concat([df, mes_dummies], axis=1)

# Dummy de regime Petrobras
df["regime_pos_2015"] = (df["Data"] >= "2015-01-01").astype(int)

print("Colunas finais disponíveis:")
print(df.columns.tolist())

# ======================================================================================
# 4) TRANSFORMAÇÕES
# ======================================================================================

print("=" * 100)
print("2) TRANSFORMAÇÕES")
print("=" * 100)

# Diferenças logarítmicas para séries positivas
series_log = {
    "Petroleo": "dln_petroleo",
    "Cambio": "dln_cambio",
    "GasolinaA": "dln_gasolinaA",
    "Gasolina": "dln_gasolina",
    "Etanol": "dln_etanol",
    "Oleo_diesel": "dln_diesel",
    "Atividade": "dln_atividade"
}

for original, novo in series_log.items():
    if original in df.columns:
        df = criar_log_diff(df, original, novo)

# IMPORTANTE:
# IPCA_Geral e IPCA_Transporte devem ser variações mensais, não índice em nível.
# Se estiverem em percentual, exemplo 0.50 para 0,50%, mantenha assim.
# Se estiverem em decimal, exemplo 0.005 para 0,50%, também funciona,
# mas a interpretação muda de escala.

# ======================================================================================
# 5) CRIAÇÃO DOS CHOQUES RESIDUAIS PADRONIZADOS
# ======================================================================================

print("=" * 100)
print("3) CRIAÇÃO DOS CHOQUES RESIDUAIS PADRONIZADOS")
print("=" * 100)

controles_choque_petroleo = []
for c in ["dln_cambio", "dln_atividade", "Selic", "Expectativa_Inflacao", "Stringency"]:
    if c in df.columns:
        controles_choque_petroleo.append(c)

df = estimar_choque_residualizado(
    df=df,
    var_base="dln_petroleo",
    nome_choque="shock_petroleo",
    controles=controles_choque_petroleo,
    p_lags=P_LAGS
)

# Choques dos combustíveis
combustiveis = {
    "Etanol": "dln_etanol",
    "Gasolina": "dln_gasolina",
    "GasolinaA": "dln_gasolinaA",
    "Oleo_diesel": "dln_diesel"
}

for nome_comb, var_comb in combustiveis.items():
    if var_comb in df.columns:
        controles_fuel = []
        for c in ["shock_petroleo", "dln_cambio", "dln_atividade", "Selic", "Expectativa_Inflacao", "Stringency"]:
            if c in df.columns:
                controles_fuel.append(c)

        df = estimar_choque_residualizado(
            df=df,
            var_base=var_comb,
            nome_choque=f"shock_{nome_comb}",
            controles=controles_fuel,
            p_lags=P_LAGS
        )

# ======================================================================================
# 6) CONFIGURAÇÃO DOS CONTROLES DO LP
# ======================================================================================

controles_lagados_base = []

for c in [
    "dln_petroleo",
    "dln_cambio",
    "dln_atividade",
    "Selic",
    "Expectativa_Inflacao",
    "Stringency"
]:
    if c in df.columns:
        controles_lagados_base.append(c)

controles_contemporaneos_base = []

# Deixe stringency contemporâneo se existir, pois é controle exógeno/institucional.
if "Stringency" in df.columns:
    controles_contemporaneos_base.append("Stringency")

# ======================================================================================
# 7) RODAR OS MODELOS
# ======================================================================================

print("=" * 100)
print("4) ESTIMAÇÃO DOS LOCAL PROJECTIONS")
print("=" * 100)

todos_resultados = []

for nome_regime, (data_ini, data_fim) in REGIMES.items():

    df_regime = df[(df["Data"] >= data_ini) & (df["Data"] <= data_fim)].copy()

    print(f"\nRodando regime: {nome_regime} | Observações: {len(df_regime)}")

    # ----------------------------------------------------------------------
    # A) Combustíveis -> IPCA Transporte e IPCA Geral
    # ----------------------------------------------------------------------
    for nome_comb in combustiveis.keys():
        shock_comb = f"shock_{nome_comb}"

        if shock_comb not in df_regime.columns:
            continue

        for y_var in ["IPCA_Geral", "IPCA_Transporte"]:

            nome_modelo = f"{nome_regime}_{nome_comb}_para_{y_var}"

            res = rodar_lp_acumulado(
                df=df_regime,
                y_var=y_var,
                shock_var=shock_comb,
                nome_modelo=nome_modelo,
                controles_lagados=controles_lagados_base,
                controles_contemporaneos=controles_contemporaneos_base,
                p_aug=P_AUG,
                h_max=H_ROBUST,
                z_conf=Z_CONF,
                interaction_regime=False
            )

            todos_resultados.append(res)

            caminho = os.path.join(PASTA_OUTPUT, f"lp_acumulada_{nome_modelo}.png")
            titulo = f"LP acumulada - {nome_modelo}"
            plotar_lp(res, titulo, caminho)

    # ----------------------------------------------------------------------
    # B) Petróleo -> combustíveis
    # ----------------------------------------------------------------------
    if "shock_petroleo" in df_regime.columns:

        for nome_comb, var_comb in combustiveis.items():

            if var_comb not in df_regime.columns:
                continue

            nome_modelo = f"{nome_regime}_Petroleo_para_{nome_comb}"

            controles_lagados = [c for c in controles_lagados_base if c != var_comb]

            res = rodar_lp_acumulado(
                df=df_regime,
                y_var=var_comb,
                shock_var="shock_petroleo",
                nome_modelo=nome_modelo,
                controles_lagados=controles_lagados,
                controles_contemporaneos=controles_contemporaneos_base,
                p_aug=P_AUG,
                h_max=H_ROBUST,
                z_conf=Z_CONF,
                interaction_regime=False
            )

            todos_resultados.append(res)

            caminho = os.path.join(PASTA_OUTPUT, f"lp_acumulada_{nome_modelo}.png")
            titulo = f"LP acumulada - {nome_modelo}"
            plotar_lp(res, titulo, caminho)

    # ----------------------------------------------------------------------
    # C) Petróleo -> IPCA: efeito total
    # ----------------------------------------------------------------------
    for y_var in ["IPCA_Geral", "IPCA_Transporte"]:

        nome_modelo = f"{nome_regime}_Petroleo_para_{y_var}_efeito_total"

        res = rodar_lp_acumulado(
            df=df_regime,
            y_var=y_var,
            shock_var="shock_petroleo",
            nome_modelo=nome_modelo,
            controles_lagados=controles_lagados_base,
            controles_contemporaneos=controles_contemporaneos_base,
            p_aug=P_AUG,
            h_max=H_ROBUST,
            z_conf=Z_CONF,
            interaction_regime=False
        )

        todos_resultados.append(res)

        caminho = os.path.join(PASTA_OUTPUT, f"lp_acumulada_{nome_modelo}.png")
        titulo = f"LP acumulada - {nome_modelo}"
        plotar_lp(res, titulo, caminho)

    # ----------------------------------------------------------------------
    # D) Petróleo -> IPCA: efeito direto controlando cada combustível
    # ----------------------------------------------------------------------
    for y_var in ["IPCA_Geral", "IPCA_Transporte"]:

        for nome_comb, var_comb in combustiveis.items():

            if var_comb not in df_regime.columns:
                continue

            nome_modelo = f"{nome_regime}_Petroleo_para_{y_var}_controle_{nome_comb}"

            controles_lagados_direto = controles_lagados_base.copy()

            if var_comb not in controles_lagados_direto:
                controles_lagados_direto.append(var_comb)

            res = rodar_lp_acumulado(
                df=df_regime,
                y_var=y_var,
                shock_var="shock_petroleo",
                nome_modelo=nome_modelo,
                controles_lagados=controles_lagados_direto,
                controles_contemporaneos=controles_contemporaneos_base,
                p_aug=P_AUG,
                h_max=H_ROBUST,
                z_conf=Z_CONF,
                interaction_regime=False
            )

            todos_resultados.append(res)

            caminho = os.path.join(PASTA_OUTPUT, f"lp_acumulada_{nome_modelo}.png")
            titulo = f"LP acumulada - {nome_modelo}"
            plotar_lp(res, titulo, caminho)

# ----------------------------------------------------------------------
# E) Modelo com interação de regime, somente na amostra cheia
# ----------------------------------------------------------------------

print("\nRodando modelos com interação de regime Petrobras na amostra cheia...")

df_full = df[(df["Data"] >= "2003-01-01") & (df["Data"] <= "2026-12-31")].copy()

for y_var in ["IPCA_Geral", "IPCA_Transporte"]:

    nome_modelo = f"2003_2026_Petroleo_para_{y_var}_interacao_regime"

    res = rodar_lp_acumulado(
        df=df_full,
        y_var=y_var,
        shock_var="shock_petroleo",
        nome_modelo=nome_modelo,
        controles_lagados=controles_lagados_base,
        controles_contemporaneos=controles_contemporaneos_base,
        p_aug=P_AUG,
        h_max=H_ROBUST,
        z_conf=Z_CONF,
        interaction_regime=True
    )

    todos_resultados.append(res)

    caminho = os.path.join(PASTA_OUTPUT, f"lp_acumulada_{nome_modelo}.png")
    titulo = f"LP acumulada - {nome_modelo}"
    plotar_lp(res, titulo, caminho)

# ======================================================================================
# 8) SALVAR RESULTADOS E TABELA-RESUMO
# ======================================================================================

print("=" * 100)
print("5) SALVANDO RESULTADOS")
print("=" * 100)

resultados_finais = pd.concat(todos_resultados, ignore_index=True)

caminho_resultados = os.path.join(PASTA_OUTPUT, "resultados_lp_completos.xlsx")
resultados_finais.to_excel(caminho_resultados, index=False)

tabela_resumo = criar_tabela_resumo(resultados_finais)

caminho_resumo = os.path.join(PASTA_OUTPUT, "tabela_resumo_lp.xlsx")
tabela_resumo.to_excel(caminho_resumo, index=False)

print(f"Resultados completos salvos em: {caminho_resultados}")
print(f"Tabela-resumo salva em: {caminho_resumo}")
print(f"Gráficos salvos na pasta: {PASTA_OUTPUT}")

print("\nResumo dos principais modelos:")
print(tabela_resumo.head(20))

print("\nFim da estimação.")