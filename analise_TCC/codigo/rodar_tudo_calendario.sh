#!/usr/bin/env bash
# Roda de novo todos os resultados do TCC: refinaria ANP sem tributos federais, agregada por calendario (01/10/2026).
# Cada passo grava a saida em ../resultados/_log_calendario/<passo>.txt e o tempo em ../resultados/_log_calendario/tempos.txt
cd "$(dirname "$0")"
LOG=../resultados/_log_calendario; mkdir -p "$LOG"; : > "$LOG/tempos.txt"
export PYTHONIOENCODING=utf-8 MPLBACKEND=Agg
passo() {
  nome="$1"; shift
  ini=$(date +%s)
  python "$@" > "$LOG/$nome.txt" 2>&1
  st=$?
  echo "$nome status=$st segundos=$(( $(date +%s) - ini ))" | tee -a "$LOG/tempos.txt"
}
passo lp_ols                lp_base_nova.py ols
passo lp_iv                 lp_base_nova.py iv
passo lp_regimes            lp_base_nova.py regimes
passo lp_comparar           lp_base_nova.py comparar
passo o23_rodar             onda2_3_forma_reduzida_ar.py rodar
passo o23_fig               onda2_3_forma_reduzida_ar.py figuras
passo o24_rodar             onda2_4_canal_combustiveis.py rodar
passo o24_fig               onda2_4_canal_combustiveis.py figuras
passo o25_rodar             onda2_5_lags_residuos.py rodar
passo o25_selecao           onda2_5_lags_residuos.py selecao
passo o25_fig               onda2_5_lags_residuos.py figuras
passo o26_rodar             onda2_6_resposta_nao_acumulada.py rodar
passo o26_ident             onda2_6_resposta_nao_acumulada.py identidade
passo o26_fig               onda2_6_resposta_nao_acumulada.py figuras
passo o26b                  onda2_6b_decomposicao_por_regime.py
passo o27_rodar             onda2_7_segunda_volta.py rodar
passo o27_decompor          onda2_7_segunda_volta.py decompor
passo o27_fig               onda2_7_segunda_volta.py figuras
passo o27b                  onda2_7b_voltas_por_regime.py
passo o28_rodar             onda2_8_teulings_zubanov.py rodar
passo o28_fig               onda2_8_teulings_zubanov.py figuras
passo o2plac_rodar          onda2_bandas_placebo.py rodar
passo o2plac_fig            onda2_bandas_placebo.py figuras
passo o31_buscar            onda3_1_quebra_endogena.py buscar
passo o31_fig               onda3_1_quebra_endogena.py figuras
passo o32_pandemia          onda3_2_pandemia_subamostras.py pandemia
passo o32_sub               onda3_2_pandemia_subamostras.py subamostras
passo o33_escolher          onda3_3_data_comum.py escolher
passo o33_estimar           onda3_3_data_comum.py estimar 2017-03
passo o33_fig               onda3_3_data_comum.py figuras
passo o34_rodar             onda3_4_diagnostico_refinaria.py rodar
passo o34_fig               onda3_4_diagnostico_refinaria.py figuras
passo o41                   onda4_1_tres_regimes.py
passo o42                   onda4_2_identificacao.py
passo o43                   onda4_3_assimetria.py
passo o44                   onda4_4_paridade_iv.py
passo o45                   onda4_5_comparacao_figuras.py
passo banca2_bp             banca2_bp_bootstrap.py
passo banca2_pand           banca2_pandemia_assimetria.py
passo banca2_supt           banca2_supt_assimetria.py
passo banca2_trib           banca2_tributos_bp_reais.py
passo t_crack               teste_crack_spread.py
passo t_greve               teste_greve_icms.py
passo t_oxford              teste_oxford_covid.py
passo artigo_rodar          artigo_set2016.py rodar
passo f_acum                fig_acumulado_vs_mensal.py
passo f_comb                fig_combustiveis_regime.py
passo f_compos              fig_composicao_choques.py
passo f_contraf             fig_contrafactual.py
passo f_corte               fig_corte_alternativo.py
passo f_paridade            fig_paridade_golfo.py
passo f_plato               fig_plato_quebra.py
passo f_repint              fig_repasse_internacional.py
passo f_resid               fig_residuos_principal.py
passo f_respreg             fig_resposta_por_regime.py
passo f_series              fig_series_historicas.py
passo f_var                 fig_var_cholesky.py
passo f_voltas              fig_voltas_e_residuos.py
passo f_artigo              figuras_artigo.py
passo o6_interagido         onda6_regime_interagido.py
passo o5_tudo               onda5_alternativos.py tudo
echo FIM | tee -a "$LOG/tempos.txt"
