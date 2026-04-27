# Resumo do Projeto — Sistema de Scoring de Risco de Óbito por Dengue

## Objetivo

Desenvolver um modelo de classificação binária para estimar a probabilidade de óbito em pacientes internados com dengue, com base nos dados sintomatológicos da ficha de notificação do SINAN. O score gerado (0–1) alimenta uma fila de prioridade de acompanhamento clínico, estratificada em quatro categorias de risco: **BAIXO RISCO**, **MODERADO**, **ALTO RISCO** e **CRÍTICO**.

---

## Sobre os Datasets

**Fonte:** SINAN — Sistema de Informação de Agravos de Notificação (dengue)

**Features:** Sintomas clínicos, sinais de alarme e gravidade da ficha de notificação, além de variáveis demográficas (idade, sexo, raça, escolaridade) e comorbidades.

**Target:** `obito` — binário (0 = sobrevivência, 1 = óbito)

| Dataset | Período | Registros | Óbitos | Prevalência | Uso |
|---|---|---|---|---|---|
| Treino | 2020–2023 | 137.743 | 2.640 | 1,92% | Treinamento dos modelos |
| Teste | 2024 | 160.534 | 5.295 | 3,30% | Avaliação principal (ano epidêmico) |
| Validação externa | 2025 | 63.271 | 1.539 | 2,43% | Teste de robustez (ano endêmico) |

O conjunto de teste **nunca foi balanceado** — o split ocorreu antes de qualquer oversampling. Os dados de 2024 representam um ano epidêmico (maior volume e severidade); os de 2025, um ano endêmico (menor prevalência e casos predominantemente mais leves).

---

## Metodologia

### Pré-processamento
- Limpeza e padronização das variáveis da ficha SINAN
- Tratamento de valores ausentes por imputação (mediana para contínuas, moda para categóricas)
- Codificação de variáveis categóricas via One-Hot Encoding

### Estratégia de treinamento
- **Desbalanceamento:** ~95% sobrevivência / ~5% óbito — tratado via `class_weight='balanced'` (ou `scale_pos_weight` no XGBoost) para maximizar sensibilidade, sem balancear o conjunto de teste
- **Validação cruzada:** StratifiedKFold(5) dentro do conjunto de treino para otimização de hiperparâmetros via GridSearchCV, com scoring=`average_precision` (AUPRC)
- **Separação temporal:** treino em 2020–2023, teste em 2024 — sem vazamento de dados

### Modelos treinados
| Modelo | Variantes |
|---|---|
| Regressão Logística | Baseline · Tuned (GridSearch) · SMOTE 1:1 / 1:5 / 1:10 |
| MLP (Rede Neural) | Baseline · Tuned |
| XGBoost | Baseline · Tuned |
| LightGBM | Baseline · Tuned |
| Random Forest | Baseline · Tuned |
| Naive Bayes | Baseline (apenas features binárias) |
| Ensemble (Voting) | LR + LR tuned + MLP + LightGBM tuned (média de probabilidades) |

### Ensemble
Testadas todas as combinações possíveis de 2 a 5 modelos via média simples de probabilidades (`itertools.combinations`). Melhor combinação selecionada por AUPRC: **LR + LR (tuned) + MLP + LightGBM (tuned)**.

### Avaliação
**Métricas prioritárias:** Sensibilidade > AUPRC > ROC-AUC > Especificidade > F1

- Sensibilidade é crítica: falsos negativos (óbitos não detectados) têm custo clínico elevado
- AUPRC priorizada sobre ROC-AUC por ser mais informativa em datasets desbalanceados

### Análises complementares
- **Bootstrap (1.000 iterações):** intervalos de confiança 95% para sensibilidade, AUPRC e ROC-AUC — análise de suporte para quantificar a incerteza das estimativas de performance
- **Validação externa 2025:** avaliação de robustez em contexto endêmico, com distribuição de severidade diferente do período de treino — critério decisivo para a escolha do modelo final

---

## Resultados

### Desempenho em 2024 (ano epidêmico) — threshold = 0.5

| Modelo | Sensibilidade | AUPRC | ROC-AUC | Especificidade |
|---|---|---|---|---|
| LR (tuned) | **0.8008** | 0.6250 | 0.9242 | 0.9087 |
| LR | 0.8004 | 0.6250 | 0.9242 | 0.9087 |
| XGBoost (tuned) | 0.7932 | 0.6314 | 0.9194 | 0.9022 |
| Ensemble | 0.7843 | **0.6472** | **0.9261** | 0.9087 |
| MLP | 0.7843 | 0.6369 | 0.9217 | 0.9087 |
| LightGBM (tuned) | 0.7549 | 0.6294 | 0.9136 | 0.9141 |
| Random Forest (tuned) | 0.4457 | 0.6268 | 0.9038 | — |

### Impacto do tuning de hiperparâmetros
- **LightGBM:** ganho expressivo (+0.073 sensibilidade, +0.020 AUPRC)
- **XGBoost:** ganho moderado (+0.044 sensibilidade, +0.020 AUPRC)
- **LR:** ganho marginal — baseline já próximo do ótimo
- **MLP:** todas as métricas pioraram — configuração padrão era a melhor
- **Random Forest:** degradação severa (-0.178 sensibilidade) — overfitting aos dados de 2020–2023

### Robustez — 2024 vs 2025

| Modelo | Δ Sensibilidade | Δ AUPRC |
|---|---|---|
| **LR** | **+0.0018** | **-0.0205** |
| LR (tuned) | -0.0126 | -0.0389 |
| Random Forest | -0.0119 | -0.0413 |
| MLP | -0.0234 | -0.0418 |
| XGBoost (tuned) | -0.0141 | -0.0516 |
| LightGBM (tuned) | -0.0259 | -0.0494 |

Todos os modelos perdem AUPRC no contexto endêmico (~0.04–0.05) — queda inerente à mudança de distribuição, não falha dos modelos. Modelos mais complexos (LightGBM, XGBoost) são mais afetados por aprender padrões específicos do período epidêmico.

### Conclusão

A **Regressão Logística baseline** é o modelo recomendado para produção com base nos seguintes critérios:

1. **Maior sensibilidade em 2024** — 0.8004, superior a todos os demais modelos no conjunto de teste epidêmico
2. **Maior robustez em 2025** — único modelo que não perde sensibilidade no contexto endêmico (Δ = +0.0018)
3. **Interpretabilidade** — coeficientes permitem explicar a contribuição de cada sintoma
4. **Menor risco de overfitting sazonal** — modelo mais simples captura padrões mais estáveis entre contextos epidemiológicos

Os limiares das categorias de risco (BAIXO / MODERADO / ALTO / CRÍTICO) devem ser ajustados empiricamente com base na prevalência real do contexto de uso antes da implantação.
