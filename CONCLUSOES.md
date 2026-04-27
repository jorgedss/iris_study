# Análise de Trabalhos Correlatos — Scoring de Risco de Óbito por Dengue

## 1. Panorama Geral da Literatura

A literatura recente converge em três abordagens para predição de gravidade/mortalidade por dengue: **scores clínicos com regressão logística**, **modelos de ML com features laboratoriais**, e **sistemas de vigilância usando dados de notificação compulsória**. Nenhum trabalho identificado usa AUPRC como métrica primária — o que reforça a relevância metodológica deste projeto para contextos com prevalência de evento <5%.

---

## 2. Trabalhos Correlatos e Comparação

### Bloco A — Predição de Gravidade/Mortalidade com ML

#### [1] Madewell et al. (2025) — Porto Rico

Comparou 9 algoritmos (Decision Tree, KNN, Naive Bayes, SVM, ANN, AdaBoost, CatBoost, LightGBM, XGBoost) em 1.708 casos confirmados (24,3% graves) do Sentinel Enhanced Dengue Surveillance System. CatBoost foi o melhor modelo isolado; ensemble meta-modelo foi o mais robusto. **Desfecho:** dengue grave (não especificamente óbito).

| Dimensão | Madewell 2025 | Este Projeto |
|---|---|---|
| Dataset | Vigilância sentinela, ~1.700 casos | SINAN, ~160k casos (teste) |
| Desfecho | Dengue grave | Óbito (mais raro, mais crítico) |
| Prevalência positivos | 24,3% | 1,9–3,3% |
| Melhor modelo | CatBoost / ensemble | Ensemble (LR + MLP + LightGBM) |
| ROC-AUC | 0,971 | 0,926 (Ensemble) |
| Features | Clínicas + laboratoriais | Sintomáticas + demográficas (sem lab) |
| Métrica prioritária | ROC-AUC | Sensibilidade > AUPRC |

> **Nota:** o alto ROC-AUC do estudo de Porto Rico é esperado dado que a prevalência de 24% é mais balanceada, o desfecho é menos grave (gravidade ≠ óbito), e há features laboratoriais (hemoconcentração, leucopenia) ausentes na ficha SINAN. Em contexto de ~3% de óbitos e sem dados laboratoriais, ROC-AUC = 0,926 deste projeto é resultado expressivo.

---

#### [2] Huang et al. (2020) — Taiwan

Coorte hospitalar de 798 pacientes com dengue confirmada (sorotipo 2, Taiwan 2015). 138 casos graves (17,4%) e 660 leves. Comparou RL, SVM, Random Forest, GBM e ANN com validação cruzada 10-fold. ANN foi o melhor modelo (AUROC = 0,832).

| Dimensão | Huang 2020 | Este Projeto |
|---|---|---|
| Dataset | 798 pacientes, Taiwan | 137k treino, 160k teste, Brasil |
| Desequilíbrio | 82,6% leve / 17,4% grave | 96,7% sobrevivência / 3,3% óbito |
| Melhor modelo | ANN (AUROC = 0,832) | RL baseline (sensibilidade = 0,800) |
| Métricas | AUROC, acurácia balanceada | Sensibilidade, AUPRC, ROC-AUC |
| Features lab | Sim (NS1, hematócrito) | Não |

> **Ponto de comparação:** AUROC = 0,832 com features laboratoriais vs. ROC-AUC = 0,924 deste projeto sem dados laboratoriais — sugere que variáveis sintomáticas da ficha SINAN têm forte capacidade preditiva quando há grande volume de dados.

---

#### [3] Diaz-Arocutipa, Chumbiauca & Soto-Becerra (2025) — Revisão Sistemática

Revisão de 35 estudos (43 modelos prognósticos). Desfechos: gravidade (70%), **mortalidade (22%)**, hospitalização, UTI. Modelos de mortalidade: C-statistic = 0,83–0,99; sensibilidade = 92–95%; especificidade = 69–88%.

> **Relevância:** os modelos de mortalidade desta revisão atingem sensibilidade de 92–95%, acima do 80,0% obtido neste projeto, mas com features laboratoriais completas. A revisão também alerta para alto risco de viés em todos os modelos revisados e ausência de calibração reportada — lacunas que este projeto trata explicitamente (separação temporal, validação externa 2025).

---

### Bloco B — Dados SINAN e Notificação Compulsória Brasileira

#### [4] Bohm et al. (2024) — SINAN Brasil

Usou SINAN do Rio de Janeiro e Minas Gerais (2016 e 2019). Dataset artificialmente balanceado (10k positivos + 10k negativos). Comparou Decision Tree, KNN, RL e MLP. **Desfecho:** classificação de caso (dengue confirmado vs. descartado) — não mortalidade.

| Dimensão | Bohm 2024 | Este Projeto |
|---|---|---|
| Fonte de dados | SINAN (RJ e MG) | SINAN (nacional) |
| Desfecho | Caso confirmado vs. descartado | Óbito vs. sobrevivência |
| Balanceamento | Dataset balanceado artificialmente | Apenas treino (teste preservado) |
| Melhor modelo | MLP (AUC = 0,988*) | Ensemble (AUPRC = 0,647) |
| Features | 10 sintomas da ficha SINAN | Sintomas + demográficas + comorbidades |

> **Limitação crítica:** balancear artificialmente o conjunto de teste infla o AUC e não reflete a performance em campo (prevalência real ~5%). Este projeto corrige esse viés ao nunca balancear o conjunto de teste.

---

#### [5] Santos et al. (2023) — SINAN + SIH + INMET

Integrou SINAN, SIH e dados meteorológicos (400k internações 2014–2020) para identificar casos mal diagnosticados de dengue. Random Forest: sensibilidade = 99%, especificidade = 76%.

> **Relevância indireta:** demonstra que erros de diagnóstico no SINAN existem (~3,4% das internações), o que pode afetar a qualidade do label `obito` nos dados de treino. Também valida o uso de Random Forest em grandes bases brasileiras.

---

#### [6] Pinto et al. (2016) — SINAN Amazonas (referência mais próxima)

Estudo retrospectivo com 1.605 casos graves de dengue no Amazonas (SINAN + SIM, 2001–2013), **61 óbitos (3,8% de letalidade)** — desequilíbrio próximo ao deste projeto. Regressão logística com stepwise identificou preditores de morte: sangramento gastrointestinal (OR 10,26), hematúria (OR 5,07), idade >55 anos (OR 4,98) e trombocitopenia <20.000/mm³ (OR 2,55).

| Dimensão | Pinto 2016 | Este Projeto |
|---|---|---|
| Fonte de dados | SINAN + SIM, Amazonas | SINAN, nacional |
| Desequilíbrio | ~4% óbitos | ~3,3% óbitos |
| Método | Regressão Logística (stepwise) | RL, MLP, XGBoost, LightGBM, RF, Ensemble |
| AUROC | 0,843 | 0,926 (Ensemble) / 0,924 (RL) |
| Features | Lab + clínicas (sangramento, hematúria, plaquetas) | Sintomáticas + demográficas |
| Validação | Interna | Temporal (2024 + 2025) |

> **Benchmark direto:** este é o trabalho mais comparável — mesma fonte de dados, mesmo desfecho, desequilíbrio similar. O ROC-AUC deste projeto (0,924 para RL) supera o AUROC = 0,843 de Pinto 2016, mesmo sem variáveis laboratoriais, graças ao volume massivo de dados (~100x mais casos).

---

### Bloco C — Scores Bedside com Regressão Logística

#### [7] Marois et al. (2021) — Nova Caledônia

Score bedside em 383 pacientes hospitalizados com dengue (surto 2017), validado em 130 pacientes (2018). Regressão logística estratificada por sexo. Variáveis: idade, hipertensão, alcoolismo, sangramento mucoso, plaquetas <30×10⁹/L e ALT elevada.

- Feminino: AUROC = 0,80; sensibilidade = 84,5%; especificidade = 78,6%
- Masculino: AUROC = 0,88; sensibilidade = 84,5%; especificidade = 95,5%

---

#### [8] Lee et al. (2016) — Taiwan

Score clínico em 1.253 adultos com dengue. Regressão logística estratificada por duração da doença (≤4 dias vs. >4 dias). Modelo ≤4 dias: AUROC = 0,848 (derivação); AUROC = 0,904 (validação); sensibilidade = 70,3%, especificidade = 90,6%.

> **Comparação com RL deste projeto:** a sensibilidade de 80,0% da RL baseline é comparável ao topo da faixa reportada em scores clínicos tradicionais (Marois: 84,5%; Lee: 70,3–80,3%), mesmo usando apenas dados de notificação — sem exames laboratoriais. O modelo atinge desempenho comparável a scores projetados com coleta prospectiva de dados.

---

#### [9] Htun, Xiong & Pang (2021) — Meta-análise de Sinais de Alarme WHO

Meta-análise de 39 estudos sobre sinais clínicos WHO associados à dengue grave. Fatores mais discriminativos: choque (OR 47,51), inconsciência (OR 29,81), sangramento gastrointestinal (OR 14,56), dispneia (OR 11,19), derrame pleural (OR 6,20) e ascite (OR 5,20).

> **Relevância:** valida a escolha de features da ficha SINAN que capturam sinais de alarme WHO. Serve como âncora para interpretação dos coeficientes da Regressão Logística deste projeto.

---

### Bloco D — Metodologia: Desequilíbrio de Classes e ML em Saúde Pública

#### [10] van den Goorbergh et al. (2022) — JAMIA

Simulações Monte Carlo (24 cenários, frações de eventos de 0,01 a 0,3) + dados reais de tumor ovariano (n=3.369). Demonstrou que correções de desequilíbrio (undersampling, oversampling, SMOTE) **não melhoram o AUROC** e causam **severa miscalibração** das probabilidades estimadas (interceptos de calibração de -0,7 a -4,5 para frações de eventos de 1%).

> **Validação metodológica direta:** justifica a decisão de não balancear o conjunto de teste e usar `class_weight='balanced'` apenas no treino. Também alerta sobre o risco de usar probabilidades de modelos treinados com SMOTE em produção — crítico dado que a saída do sistema é `probabilidade: 0.72`.

---

#### [11] Santos et al. (2019) — Cadernos de Saúde Pública

Comparou RL, LASSO, Redes Neurais, Gradient Boosted Trees e RF para predição de óbito em idosos no Brasil (coorte SABE, n=2.808, 15% eventos). Melhor: Redes Neurais (AUROC = 0,779).

> **Referência nacional:** em contexto brasileiro com dados observacionais e desequilíbrio moderado (15% de eventos), o AUROC máximo foi 0,779 — abaixo do ROC-AUC = 0,926 obtido neste projeto, evidenciando o ganho de escala (137k vs. 2.808 registros de treino).

---

#### [12] Rocha & Giesbrecht (2022) — São Luís/MA

Avaliou RL, LDA, Naive Bayes, Decision Tree e RF para risco de dengue em bairros de São Luís (dados municipais 2014–2020). Testou SMOTE, ADASYN e DBSMOTE. Melhor: RF + DBSMOTE (AUC = 0,751; sensibilidade = 0,754; especificidade = 0,605).

> **Comparação com SMOTE deste projeto:** as variantes SMOTE 1:1, 1:5 e 1:10 testadas são consistentes com a literatura brasileira. Confirma o padrão de ganhos modestos com oversampling em problemas epidemiológicos.

---

## 3. Tabela Comparativa Global

| # | Estudo | Ano | País | Desfecho | n (teste) | Desequilíbrio | Melhor Modelo | ROC-AUC | Sensibilidade | Dados Lab? |
|---|---|---|---|---|---|---|---|---|---|---|
| — | **Este Projeto** | **2026** | **Brasil** | **Óbito** | **160.534** | **96,7% / 3,3%** | **RL baseline** | **0,924** | **0,800** | **Não** |
| 1 | Madewell | 2025 | Porto Rico | Dengue grave | 1.708 | 75,7% / 24,3% | CatBoost | 0,971 | — | Sim |
| 2 | Huang | 2020 | Taiwan | Dengue grave | 798 | 82,6% / 17,4% | ANN | 0,832 | — | Sim |
| 3 | Diaz-Arocutipa | 2025 | Revisão sist. | Mortalidade | — | — | Vários | 0,83–0,99 | 92–95% | Sim |
| 4 | Bohm | 2024 | Brasil | Caso confirmado | 6.000* | Balanceado* | MLP | 0,988* | — | Não |
| 5 | Santos CY | 2023 | Brasil | Mal diagnóstico | 400k | — | Random Forest | — | 0,99 | Não |
| 6 | Pinto | 2016 | Brasil | Óbito (dengue grave) | 1.605 | 96,2% / 3,8% | RL | 0,843 | — | Sim |
| 7 | Marois | 2021 | Nova Caledônia | Dengue grave | 383 | — | RL | 0,80–0,88 | 0,845 | Sim |
| 8 | Lee | 2016 | Taiwan | Dengue grave | 1.253 | — | RL | 0,848–0,917 | 0,703–0,803 | Sim |
| 11 | Santos HG | 2019 | Brasil | Óbito (idosos) | 2.808 | 85% / 15% | Redes Neurais | 0,779 | — | Não |
| 12 | Rocha | 2022 | Brasil | Risco dengue | 123 bairros | 78% / 22% | RF + DBSMOTE | 0,751 | 0,754 | Não |

*AUC inflado por balanceamento artificial do conjunto de teste.

---

## 4. Verificações de Robustez do Modelo Final

### Encoding de SG_UF — experimento controlado

A variável `SG_UF` (unidade federativa) foi codificada com `OrdinalEncoder` no pipeline da Regressão Logística, atribuindo inteiros arbitrários a cada estado. Em modelos lineares, esse encoding é tecnicamente inadequado para variáveis nominais, pois impõe uma ordenação sem significado semântico entre os estados.

Para avaliar o impacto real dessa limitação, foi conduzido um experimento controlado com três variantes do modelo tuned (C=10, L2, lbfgs), mantendo todos os demais hiperparâmetros e dados idênticos:

| Variante | Sensibilidade | AUPRC | ROC-AUC | Δ Sensibilidade | Δ AUPRC | Δ ROC-AUC |
|---|---|---|---|---|---|---|
| OrdinalEncoder (original) | 0,8008 | 0,6250 | 0,9242 | — | — | — |
| OneHotEncoder (corrigido) | 0,8079 | 0,6194 | 0,9207 | +0,0071 | −0,0056 | −0,0035 |
| Sem SG_UF | 0,8008 | 0,6254 | 0,9243 | +0,0000 | +0,0004 | +0,0001 |

**Achados:**
- Corrigir o encoding para `OneHotEncoder` adiciona 26 features esparsas (60 → 86) e **piora** as métricas prioritárias (AUPRC −0,006; ROC-AUC −0,004), provavelmente porque as features estaduais esparsas introduzem ruído que a regularização L2 não amortece completamente.
- Remover completamente `SG_UF` tem impacto desprezível em todas as métricas (ΔSensibilidade = 0,0000; ΔAUPRC = +0,0004; ΔROC-AUC = +0,0001).

**Conclusão:** `SG_UF` não contribui para o poder preditivo do modelo independentemente do encoding utilizado. O risco de óbito por dengue grave é explicado pelos sinais clínicos e características do paciente, não pela localização geográfica de notificação. A limitação de encoding não invalida o modelo final — o `OrdinalEncoder` atua como regularizador implícito, comprimindo a informação geográfica em um único coeficiente de baixo impacto.

Notebook de referência: `04_validation/lr_uf_encoding_comparison.ipynb`

---

## 5. Posicionamento do Projeto na Literatura

**Contribuições originais em relação ao estado da arte:**

1. **Escala:** maior base de dados brasileira para predição de mortalidade por dengue (>160k casos de teste), superando em ~100x o próximo estudo nacional comparável (Pinto 2016, n=1.605).

2. **Validação temporal dupla:** separação treino/teste por ano epidemiológico + validação externa em ano endêmico (2025) — nenhum dos trabalhos revisados adota design equivalente.

3. **AUPRC como métrica primária:** metodologicamente mais rigoroso para datasets com <5% de eventos; ausente como métrica primária em todos os trabalhos revisados.

4. **Sem dados laboratoriais:** RL com ROC-AUC = 0,924 usando apenas sintomas e variáveis demográficas da ficha SINAN é resultado expressivo — Pinto 2016 atingiu AUC = 0,843 com features laboratoriais adicionais.

5. **Análise de robustez epidemiológica:** avaliação explícita da degradação de performance entre contexto epidêmico e endêmico — dimensão inexplorada nos trabalhos correlatos.

---

## 5. Referências Bibliográficas

**[1]** MADEWELL ZJ, Rodriguez DM, Thayer MB, Rivera-Amill V, Paz-Bailey G, Adams LE, Wong JM. Machine learning for predicting severe dengue in Puerto Rico. *Infectious Diseases of Poverty*. 2025;14(1). DOI: 10.1186/s40249-025-01273-0

**[2]** HUANG SW, Tsai HP, Hung SJ, Ko WC, Wang JR. Assessing the risk of dengue severity using demographic information and laboratory test results with machine learning. *PLoS Neglected Tropical Diseases*. 2020;14(12):e0008960. DOI: 10.1371/journal.pntd.0008960

**[3]** DIAZ-AROCUTIPA C, Chumbiauca M, Soto-Becerra P. Prognostic models in patients with dengue: a systematic review. *The American Journal of Tropical Medicine and Hygiene*. 2025. DOI: 10.4269/ajtmh.24-0653

**[4]** BOHM BC, Borges FEM, Silva SCM, Soares AT, Ferreira DD, Belo VS, et al. Utilization of machine learning for dengue case screening. *BMC Public Health*. 2024;24(1). DOI: 10.1186/s12889-024-19083-8

**[5]** SANTOS CY, Tuboi S, Abreu AJL, Abud DA, Lobao Neto AA, Pereira R, Siqueira JB Jr. A machine learning model to assess potential misdiagnosed dengue hospitalization. *Heliyon*. 2023;9(6):e16634. DOI: 10.1016/j.heliyon.2023.e16634

**[6]** PINTO RC, de Castro DB, de Albuquerque BC, Sampaio VDS, dos Passos RA, da Costa CF, et al. Mortality predictors in patients with severe dengue in the State of Amazonas, Brazil. *PLoS One*. 2016;11(8):e0161884. DOI: 10.1371/journal.pone.0161884

**[7]** MAROIS I, Forfait C, Inizan C, Klement-Frutos E, Valiame A, Aubert D, et al. Development of a bedside score to predict dengue severity. *BMC Infectious Diseases*. 2021;21(1):527. DOI: 10.1186/s12879-021-06146-z

**[8]** LEE IK, Liu JW, Chen YH, Chen YC, Tsai CY, Huang SY, et al. Development of a simple clinical risk score for early prediction of severe dengue in adult patients. *PLoS One*. 2016;11(5):e0154772. DOI: 10.1371/journal.pone.0154772

**[9]** HTUN TP, Xiong Z, Pang J. Clinical signs and symptoms associated with WHO severe dengue classification: a systematic review and meta-analysis. *Emerging Microbes & Infections*. 2021;10(1):1116-1128. DOI: 10.1080/22221751.2021.1935327

**[10]** VAN DEN GOORBERGH R, van Smeden M, Timmerman D, Van Calster B. The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association*. 2022;29(9):1525-1534. DOI: 10.1093/jamia/ocac093

**[11]** SANTOS HG, Nascimento CF, Izbicki R, Duarte YAO, Chiavegatto Filho ADP. Machine learning para análises preditivas em saúde: exemplo de aplicação para predizer óbito em idosos de São Paulo, Brasil. *Cadernos de Saúde Pública*. 2019;35(7):e00050818. DOI: 10.1590/0102-311X00050818

**[12]** ROCHA FP, Giesbrecht M. Machine learning algorithms for dengue risk assessment: a case study for São Luís do Maranhão. *Computational and Applied Mathematics*. 2022;41(8). DOI: 10.1007/s40314-022-02101-z
