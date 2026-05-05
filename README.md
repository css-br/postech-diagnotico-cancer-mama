# TechChallenge A — Diagnóstico de Câncer de Mama com ML

Projeto de classificação de câncer de mama usando Machine Learning e Redes Neurais Convolucionais, desenvolvido como parte do POSTECH FIAP Fase 1.

Dois notebooks implementam abordagens complementares:
- **diagnostico_cancer_mama.ipynb** — dados estruturados (Logistic Regression + Random Forest)
- **diagnostico_imagem_cnn.ipynb** — dados de imagem (CNN com transfer learning MobileNetV2)

---

## Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) instalado e em execução
- Python 3.8+ instalado localmente (apenas para executar o `copy_sample.py`)

---

## Estrutura do projeto

```
TechChallengeA/
├── data/
│   ├── data.csv                     # Dataset Breast Cancer Wisconsin
│   └── cnn-data/                    # Dataset CBIS-DDSM (gerado pelo copy_sample.py)
│       ├── csv/
│       │   ├── mass_case_description_train_set.csv
│       │   ├── mass_case_description_test_set.csv
│       │   ├── train_sample.csv     # Amostra de treino (usada pelo notebook CNN)
│       │   └── test_sample.csv      # Amostra de teste (usada pelo notebook CNN)
│       └── jpeg/
│           └── <UID>/               # Imagens de mamografia (ROI recortado)
├── diagnostico_cancer_mama.ipynb    # Notebook: diagnostico_cancer_mama, dados estruturados
├── diagnostico_imagem_cnn.ipynb     # Notebook: [EXTRA] diagnostico_imagem_cnn: CNN por imagem
├── copy_sample.py                   # Script de amostragem do CBIS-DDSM
├── requirements.txt                 # Dependências Python
├── Dockerfile                       # Container único para ambos os notebooks
└── README.md
```

---

## Notebook: diagnostico_cancer_mama — Dados Estruturados

### Dataset
**Breast Cancer Wisconsin (Diagnostic)** — 569 amostras, 30 features numéricas extraídas de biópsias.
Arquivo incluído em `data/data.csv`.
Fonte: [Kaggle — UCI ML Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)

### Pipeline
- Análise exploratória e pré-processamento
- Redução de dimensionalidade com PCA
- Treinamento e avaliação de dois classificadores

### Modelos
| Modelo | Foco |
|--------|------|
| Logistic Regression | Baseline linear |
| Random Forest | Ensemble, importância de features |

---

## Notebook: [EXTRA] diagnostico_imagem_cnn — Diagnóstico por Imagem (CNN)

### Dataset
**CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) — recortes de mamografia (ROI de massa) classificados como BENIGNO/MALIGNO.
Fonte: [Kaggle — CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

### Pipeline
- Transfer learning com **MobileNetV2** (pesos ImageNet)
- Fase 1: treino do cabeçalho com backbone congelado
- Fase 2: fine-tuning das últimas 30 camadas
- Visualização de predições (acerto/erro por imagem)
- **Grad-CAM**: mapa de calor das regiões de atenção do modelo

### Modelo
| Componente | Detalhe |
|------------|---------|
| Backbone | MobileNetV2 (ImageNet, congelado na fase 1) |
| Cabeçalho | GlobalAvgPool → BatchNorm → Dense(128) → Dropout(0.4) → Dense(1, sigmoid) |
| Otimizador | Adam (lr=1e-3 fase 1, lr=1e-4 fase 2) |
| Loss | Binary Crossentropy |

---

## Como executar

### Passo 1 — Preparar o dataset CBIS-DDSM

1. Baixe o dataset ZIP em: [kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
2. Extraia o ZIP em qualquer pasta (ex: `F:\archive`)
3. Configure `SOURCE_DIR` no arquivo `copy_sample.py` com o caminho de extração
4. Execute o script de amostragem:

```bash
pip install pandas
python copy_sample.py
```

O script copia 50 imagens por classe (200 no total) para `data/cnn-data/`, criando os arquivos `train_sample.csv` e `test_sample.csv` prontos para o notebook.

---

### Passo 2 — Construir a imagem Docker

```bash
docker build -t cancer-mana-ml .
```

> O build inclui TensorFlow e todas as dependências. Leva alguns minutos na primeira vez.

---

### Passo 3 — Iniciar o container

**Apenas Notebook: diagnostico_cancer_mama (dados estruturados):**
```bash
docker run -p 8888:8888 cancer-mana-ml
```

**Ambos os notebooks (com dataset de imagens):**

Windows (PowerShell):
```powershell
docker run -p 8888:8888 -v "${PWD}/data:/app/data" cancer-mama-ml
```

Linux/Mac:
```bash
docker run -p 8888:8888 -v "$(pwd)/data:/app/data" cancer-mama-ml
```

---

### Passo 4 — Acessar o Jupyter Notebook

Abra o navegador em:
```
http://localhost:8888
```

Execute os notebooks com **Kernel > Restart & Run All**.

---

## Métricas avaliadas

| Métrica | Notebook: diagnostico_cancer_mama | Notebook: [EXTRA] diagnostico_imagem_cnn |
|---------|-----------|-----------|
| Accuracy | ✓ | ✓ |
| Recall | ✓ (foco principal) | ✓ |
| Precision | ✓ | ✓ |
| F1-score | ✓ | ✓ |
| ROC-AUC | ✓ | ✓ |
| Matriz de confusão | ✓ | ✓ |
| Visualização de predições | — | ✓ |
| Grad-CAM | — | ✓ |

> **Recall** é a métrica principal em ambos os notebooks — falsos negativos (malignos classificados como benignos) têm alto custo clínico.

---

## Executando sem Docker (alternativa local)

Caso o Docker não esteja disponível, é possível executar os notebooks diretamente:

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Instalar dependências
pip install -r requirements.txt

# Iniciar Jupyter
jupyter notebook
```

---

## Relatório Técnico

### Análise Exploratória

**Notebook: diagnostico_cancer_mama — Breast Cancer Wisconsin:**
- Dataset com 569 amostras, 30 features numéricas (média, erro padrão e pior valor de 10 características de biópsias)
- Distribuição de classes: ~63% Benigno (B), ~37% Maligno (M) — leve desbalanceamento gerenciável
- Alta correlação entre features de média e pior valor (redundância esperada → justifica uso de PCA)
- Ausência de valores nulos; coluna `id` descartada por não ser preditiva
- Separação clara entre classes em algumas features (ex: `radius_mean`, `concavity_mean`) evidenciada em pairplots

**Notebook: [EXTRA] diagnostico_imagem_cnn — CBIS-DDSM:**
- Dataset de mamografias com ROI de massas (recorte da região de interesse)
- Amostra utilizada: 200 imagens (50 treino benigno, 50 treino maligno, 50 teste benigno, 50 teste maligno)
- Imagens em escala de cinza convertidas para RGB (requisito do MobileNetV2)
- Redimensionamento para 224×224 pixels (padrão ImageNet)
- Desbalanceamento controlado pela estratégia de amostragem balanceada no `copy_sample.py`

---

### Estratégias de Pré-processamento

**Dados Estruturados:**
- Remoção de colunas irrelevantes (`id`, `Unnamed: 32`)
- Encoding do target: M → 1, B → 0
- Padronização via `StandardScaler` (média 0, desvio padrão 1) — necessário para Logistic Regression e PCA
- Redução de dimensionalidade com PCA preservando 95% da variância, reduzindo de 30 para ~10 componentes principais

**Imagens:**
- Normalização de pixels para [0, 1] dividindo por 255
- Data augmentation na fase de treino: rotação, flip horizontal, zoom e deslocamento — aumenta robustez sem necessidade de mais dados
- Sem augmentation na fase de validação/teste (apenas normalização)

---

### Modelos e Justificativas

**Notebook: diagnostico_cancer_mama — Por que esses modelos?**

| Modelo | Justificativa |
|--------|---------------|
| **Logistic Regression** | Baseline interpretável; funciona bem com features normalizadas e linearmente separáveis; coeficientes indicam importância de cada componente PCA |
| **Random Forest** | Ensemble robusto a outliers; não requer normalização; captura relações não-lineares; importância de features permite interpretabilidade clínica |

Ambos foram escolhidos por serem modelos consolidados em dados tabulares médicos, com boa interpretabilidade — fator relevante em contextos clínicos.

**Notebook: [EXTRA] diagnostico_imagem_cnn — Por que CNN com Transfer Learning?**

| Escolha | Justificativa |
|---------|---------------|
| **MobileNetV2** | Arquitetura eficiente treinada no ImageNet; extrai features visuais genéricas (bordas, texturas, formas) transferíveis para imagens médicas |
| **Fine-tuning em 2 fases** | Fase 1 treina apenas o cabeçalho (evita destruir pesos pré-treinados com gradientes grandes); Fase 2 ajusta camadas superiores do backbone com learning rate baixo |
| **Grad-CAM** | Visualiza quais regiões da imagem influenciaram a predição, adicionando interpretabilidade clínica ao modelo |

Transfer learning é a abordagem padrão quando o dataset é pequeno (apenas 200 imagens aqui), pois aproveita o conhecimento visual já aprendido em milhões de imagens.

---

### Resultados e Interpretação

**Notebook: diagnostico_cancer_mama — Dados Estruturados:**

Os modelos foram avaliados com foco em **Recall** (sensibilidade), pois em diagnóstico oncológico o custo de um falso negativo (maligno classificado como benigno) é muito maior do que um falso positivo.

- **Logistic Regression com PCA**: Recall elevado após otimização do threshold de decisão; PCA reduziu overfitting e tempo de treino sem perda significativa de performance
- **Random Forest**: Ligeiramente superior em F1-score; importância de features destacou `concavity_mean` e `area_worst` como principais preditores — alinhado com literatura clínica

**Notebook: [EXTRA] diagnostico_imagem_cnn — Diagnóstico por Imagem:**

- A fase de fine-tuning melhorou métricas em relação ao treino apenas do cabeçalho
- Grad-CAM confirmou que o modelo aprende a focar nas regiões de massa e não em artefatos da imagem
- Com apenas 200 imagens, os resultados demonstram a viabilidade da abordagem; um dataset completo (>10k imagens) tenderia a performance significativamente superior
- O desbalanceamento controlado na amostragem evitou viés em favor da classe majoritária

**Métrica principal — Recall:** Em ambos os notebooks o modelo foi otimizado para maximizar Recall, aceitando redução em Precision quando necessário. Esta é a escolha correta para triagem clínica, onde é preferível investigar mais casos suspeitos do que deixar passar um maligno real.

---

## Observação

Os modelos saõ ferramentas auxiliares que podem aumentar a eficiência
e reduzir erros humanos por fadiga, mas nunca devem substituir o julgamento clínico profissional.
Decisões de tratamento devem sempre ser tomadas por um especialista qualificado com base no quadro completo do paciente.
