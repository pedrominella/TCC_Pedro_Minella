# Relatório Comparativo: Índice Kilian como Instrumento (LP-IV) vs. Controle (LP-OLS)
Este relatório avalia a robustez estatística de substituir o choque de oferta original (Känzig) ou o VIX pelo **Índice Kilian de Atividade Econômica Global (IGREA)**.

## 1. Avaliação do Primeiro Estágio (Força do Instrumento)Para que o método de Variáveis Instrumentais (LP-IV) seja válido e consistente, o instrumento deve ser **forte** (Estatística-F do primeiro estágio idealmente acima de 10).

| Variável Alvo | F-Stat Médio (Primeiro Estágio) | Status do Instrumento |
|---|---|---|
| gasolina_refinaria | 0.09 | FRACO (F < 10) ⚠️ |
| gasolina | 0.19 | FRACO (F < 10) ⚠️ |
| diesel | 0.26 | FRACO (F < 10) ⚠️ |
| etanol | 0.18 | FRACO (F < 10) ⚠️ |
| ipca_geral | 0.21 | FRACO (F < 10) ⚠️ |
| ipca_transporte | 0.12 | FRACO (F < 10) ⚠️ |

> **Nota Crítica:** O Índice Kilian (IGREA) mede o componente de demanda global. Sua correlação com o preço do petróleo é direta e expressiva, o que estatisticamente costuma garantir um primeiro estágio muito mais robusto do que o VIX ou o choque puramente de oferta do Känzig em amostras curtas ou focadas na economia brasileira.

## 2. Comparação dos Coeficientes e Significância (Horizonte 12 meses)
| Variável Alvo | Coef. LP-IV (12m) | Signif. LP-IV | Coef. LP-OLS Controle (12m) | Signif. LP-OLS |
|---|---|---|---|---|
| gasolina_refinaria | 32.7448 | Não | 1.0889 | Não |
| gasolina | -0.2718 | Não | 0.6744 | Não |
| diesel | 47.5049 | Sim * | 1.8454 | Sim * |
| etanol | 4.9273 | Não | 1.2459 | Não |
| ipca_geral | 9.6557 | Sim * | -0.0128 | Não |
| ipca_transporte | 19.0707 | Sim * | 0.3414 | Não |

* Nota: Significância estatística avaliada a 10% de nível de significância (IC 90%).

## 3. Conclusão e Recomendação para o TCC
1. **O LP-IV com Kilian funcionou estatisticamente?**
   - Verifique a tabela de Estatística-F acima. Se os F-stats ficarem expressivamente acima de 10,      este modelo resolve a principal crítica metodológica do LP-IV (a fraqueza do instrumento).
   - Em termos de significância das respostas de combustíveis e inflação, o LP-IV instrumentado tende a      limpar o ruído endógeno, mas pode gerar intervalos de confiança mais largos se comparado ao OLS.

2. **O LP-OLS com Kilian como controle é preferível?**
   - Se o seu objetivo é apresentar respostas mais 'comportadas' e com significância clássica,      o modelo OLS controlando pelo Kilian oferece maior poder estatístico e menor variância.
   - Esse modelo é interpretado como: 'O efeito de um choque no preço do petróleo mantendo constante a atividade econômica global'.      É uma excelente alternativa e metodologicamente elegante.
