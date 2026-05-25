import docx
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = docx.Document('TCC_Pedro_v7_resultados_finais.docx')
paras = doc.paragraphs

# Lista de pares: (texto final do paragrafo anterior, equação a inserir)
equacoes = {
    'medida pode ser escrita como:': 
        'NOPI_t = max(0, P_t - max(P_{t-1}, ..., P_{t-12}))',
    'atividade econômica global e preço real do petróleo:': 
        'A_0 X_t = α + A_1 X_{t-1} + ... + ε_t',
    'Esse mecanismo pode ser representado por:': 
        'P^{BRL}_t = P^{USD}_t × E_t',
    'fatores externos, como câmbio e commodities:': 
        'π_t = c + α π^e_t + β π_{t-1} + γ ỹ_t + δ ΔE_t + θ ΔP^{oil}_t + ε_t',
    'preço da gasolina ao consumidor pode ser representada, de modo simplificado, como:': 
        'P^{consumidor}_t = P^{refinaria}_t + P^{etanol}_t + Tributos_t + Margens_t',
    'o combustível ao consumidor é o coeficiente de pass-through:': 
        'PT_h = ΔP^{consumidor}_{t+h} / ΔP^{oil}_t',
    'por uma local projection em painel:': 
        'π_{i, t+h} = α_{i,h} + λ_{t,h} + β_h Shock_{i,t} + Γ X_{i,t} + ε_{i, t+h}',
    'com controles exógenos pode ser escrita como:': 
        'Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + B X_t + u_t',
    'ao choque j no horizonte h pode ser representada por:': 
        'IRF(h) = ∂Y_{t+h} / ∂ε_{j,t}',
    'cada horizonte . A forma canônica é:': 
        'Y_{t+h} = α_h + β_h Shock_t + Γ_h X_t + ε_{t+h}',
    'A primeira etapa isola a componente exógena do choque:': 
        'x_t = γ_0 + γ_1 z_t + Γ W_t + v_t',
    'resposta da variável de interesse à parte instrumentada do choque:': 
        'Y_{t+h} = α_h + β_h x̂_t + Θ_h W_t + ε_{t+h}',
    'Uma forma comum de especificar esse modelo é:': 
        'Y_{t+h} = I_t [ α_{1,h} + β_{1,h} Shock_t ] + (1 - I_t) [ α_{2,h} + β_{2,h} Shock_t ] + Γ_h X_t + ε_{t+h}',
    'completamente independente para cada horizonte, pode-se escrever:': 
        'β_h = Σ_{k=1}^K c_k B_k(h)',
    'De modo simplificado, pode-se assumir:': 
        'β_h ~ N(μ, Σ)',
    'que a resposta ao choque mude ao longo do tempo:': 
        'Y_{t+h} = α_{t,h} + β_{t,h} Shock_t + Γ_{t,h} X_t + ε_{t+h}',
    'da variável k pode ser escrita como:': 
        'FEVD_{j,k}(h) = (Σ_{i=0}^h β_{i,j,k}^2 σ_j^2) / (Σ_m Σ_{i=0}^h β_{i,m,k}^2 σ_m^2)',
    'combustível pode estimar o efeito do petróleo em reais sobre o preço doméstico do combustível:': 
        'P^{fuel}_{t+h} = α_h + β_h P^{oil}_t + Γ_h X_t + ε_{t+h}',
    'efeito do preço do combustível sobre a inflação. Uma especificação possível é:': 
        'π_{t+h} = α_h + β_h P^{fuel}_t + Γ_h X_t + ε_{t+h}',
    'antes e depois de mudanças na política comercial da Petrobras:': 
        'π_{t+h} = α_h + β_{1,h} P^{fuel}_t × D_t + β_{2,h} P^{fuel}_t × (1-D_t) + Γ_h X_t + ε_{t+h}',
    'Uma panel LP pode ser escrita como:': 
        'π_{r, t+h} = α_{r,h} + λ_{t,h} + β_h P^{fuel}_{r,t} + Γ X_{r,t} + ε_{r, t+h}',
    'A variação logarítmica mensal é definida por:': 
        'Δy_t = ln(Y_t) - ln(Y_{t-1})',
    'Quando a série estava em nível, a inflação mensal foi calculada por:': 
        'π_t = (IPCA_t / IPCA_{t-1} - 1) × 100',
    'a resposta da variável dependente a um choque de um desvio-padrão:': 
        'Shock_t = Δx_t / σ_{Δx}',
    'conhecido como ADF. Sua forma simplificada pode ser escrita como:': 
        'Δy_t = α + βt + γ y_{t-1} + Σ_{i=1}^p δ_i Δy_{t-i} + ε_t',
    'O teste KPSS, cuja hipótese nula é a estacionariedade. Em forma simplificada:': 
        'y_t = c_t + δt + u_t,  com  c_t = c_{t-1} + v_t',
    'A forma vetorial associada ao teste pode ser expressa como:': 
        'ΔY_t = Π Y_{t-1} + Σ_{i=1}^{p-1} Γ_i ΔY_{t-i} + ε_t',
    'A especificação básica da Local Projection estimada neste trabalho é:': 
        'y_{t+h} - y_{t-1} = α_h + β_h Shock_t + Σ_{i=1}^p γ_{h,i} Δy_{t-i} + Σ_{j=1}^q θ_{h,j} X_{t-j} + ε_{t+h}',
    'do choque, para os horizontes de h=0 até h=12.': 
        'P^{fuel}_{t+h} - P^{fuel}_{t-1} = α_h + β_h ΔP^{oil}_t + Controles + ε_{t+h}'
}

# Iterate to fix equations
fixed_count = 0
for i in range(1, len(paras)):
    prev_text = paras[i-1].text.strip()
    curr_text = paras[i].text.strip()
    
    # Check if this paragraph is empty and follows a colon
    if prev_text.endswith(':') and curr_text == '':
        for key, eq in equacoes.items():
            if key in prev_text:
                paras[i].text = eq
                # Format equation paragraph: bold, italic, centered
                for run in paras[i].runs:
                    run.italic = True
                paras[i].alignment = WD_ALIGN_PARAGRAPH.CENTER
                fixed_count += 1
                break

doc.save('TCC_Pedro_v7_resultados_finais.docx')
print(f'Pronto! {fixed_count} equações matematicas foram restauradas no documento.')
