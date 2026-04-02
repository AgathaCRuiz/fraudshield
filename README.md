<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=FraudShield&fontSize=80&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=ML%20pipeline%20para%20detecção%20de%20fraudes%20em%20cartões%20de%20crédito&descAlignY=60&descSize=16" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Author](https://img.shields.io/badge/by-AgathaCRuiz-a855f7?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AgathaCRuiz)

<br/>

> **Pipeline completo de Machine Learning** — da exploração dos dados ao deploy em produção —  
> para detecção de fraudes em transações financeiras com técnicas avançadas de balanceamento,  
> análise de custo e explicabilidade via SHAP.

<br/>

</div>

---

## ✦ Sobre o Projeto

Fraudes em cartões de crédito representam um problema crítico para instituições financeiras. Este projeto aborda o desafio de ponta a ponta: desde a análise exploratória dos dados até a disponibilização de um modelo em produção via API REST.

O dataset utilizado é o [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), com **284.807 transações** e apenas **0,17% de fraudes** — um exemplo clássico de classificação altamente desbalanceada.

<br/>

## 📊 Resultados

<div align="center">

| Modelo | AUC-ROC | AUPRC | F1 (Fraude) |
|:---|:---:|:---:|:---:|
| 🥇 **XGBoost** | **0.9823** | **0.8641** | **0.8512** |
| 🥈 LightGBM | 0.9801 | 0.8530 | 0.8398 |
| 🥉 CatBoost | 0.9785 | 0.8471 | 0.8301 |
| Random Forest | 0.9721 | 0.8203 | 0.7980 |
| Logistic Regression | 0.9501 | 0.7012 | 0.7640 |
| Isolation Forest ★ | 0.7734 | 0.1502 | 0.4210 |

★ modelo **não supervisionado** — não usa labels de fraude no treino

</div>

> **AUPRC** (Area Under Precision-Recall Curve) é a métrica principal. Em dados tão desbalanceados,  
> a acurácia e o AUC-ROC podem ser enganosos — a curva Precision-Recall conta a história real.

<br/>

## 🗂️ Estrutura do Projeto

```
fraud_detection/
│
├── 📓 notebooks/                  # Análise em ordem de execução
│   ├── 01_exploracao.ipynb        # EDA: padrões, distribuições, correlações
│   ├── 02_features.ipynb          # Feature engineering + exporta dados processados
│   ├── 03_modelos.ipynb           # Treina e compara todos os modelos
│   └── 04_resultados.ipynb        # Custo financeiro, SHAP, prep para deploy
│
├── 🐍 src/                        # Código Python reutilizável
│   ├── data/
│   │   └── loader.py              # Carregamento e split dos dados
│   ├── features/
│   │   └── engineering.py         # Criação de features (usado tb na API)
│   ├── models/
│   │   ├── train.py               # Funções de treino de cada modelo
│   │   └── evaluate.py            # Métricas e análise de custo financeiro
│   ├── visualization/
│   │   └── plots.py               # Funções de plot reutilizáveis
│   └── api/
│       └── app.py                 # API FastAPI para produção
│
├── 📁 data/
│   ├── raw/                       # Dados originais — nunca edite
│   └── processed/                 # Pós feature engineering
│
├── 🤖 models/                     # Artefatos serializados (.pkl)
├── 📈 reports/figures/            # Gráficos exportados
├── 🧪 tests/
│   └── test_features.py           # Testes unitários com pytest
│
├── .env.example                   # Template de variáveis de ambiente
├── .gitignore
├── requirements.txt
└── README.md
```

<br/>

## 🔄 Fluxo do Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE COMPLETO                           │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────┐
  │  data/   │───▶│  notebook    │───▶│  notebook    │───▶│models/ │
  │  raw/    │    │  01 + 02     │    │     03       │    │ *.pkl  │
  │  .csv    │    │  EDA +       │    │  Treino de   │    │        │
  └──────────┘    │  Features    │    │  5 modelos   │    └───┬────┘
                  └──────┬───────┘    └──────────────┘        │
                         │                                     │
                         ▼                                     ▼
                  ┌──────────────┐    ┌──────────────┐    ┌────────────┐
                  │  data/       │    │  notebook    │    │  FastAPI   │
                  │  processed/  │───▶│     04       │───▶│  /prever   │
                  │  X_train...  │    │  Custo +     │    │  endpoint  │
                  └──────────────┘    │  SHAP        │    └────────────┘
                                      └──────────────┘
```

<br/>

## ⚙️ Features Criadas

| Feature | Origem | Por quê? |
|:---|:---|:---|
| `Amount_log` | `log1p(Amount)` | Reduz assimetria da distribuição de valores |
| `Amount_scaled` | `StandardScaler(Amount)` | Necessário para modelos lineares |
| `Hour` | `(Time // 3600) % 24` | Fraudes têm padrão horário distinto |
| `Is_night` | `Hour entre 22h e 6h` | Transações noturnas têm maior taxa de fraude |

<br/>

## 💰 Análise de Custo Financeiro

O projeto vai além das métricas tradicionais de ML e quantifica o impacto financeiro real de cada decisão de threshold.

```
Custo por Falso Negativo (fraude não detectada)  →  R$ 500
Custo por Falso Positivo (bloqueio indevido)     →  R$  10
```

O threshold ótimo **não é 0.5** — é calculado minimizando o custo total, o que muda completamente a estratégia de decisão do modelo. Os valores de custo são configuráveis via `.env`.

<br/>

## 🚀 Como Executar

### Pré-requisitos

- Python 3.11+
- Git

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/AgathaCRuiz/fraudshield.git
cd fraudshield

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com seus valores
```

### Notebooks — ordem de execução

```bash
# Abra no VSCode com a extensão Jupyter instalada
code .

# Ou via Jupyter Lab
jupyter lab notebooks/
```

| # | Notebook | O que faz |
|:---:|:---|:---|
| 01 | `01_exploracao.ipynb` | EDA: visualiza distribuições, correlações e padrões temporais |
| 02 | `02_features.ipynb` | Cria features e exporta para `data/processed/` |
| 03 | `03_modelos.ipynb` | Treina todos os modelos e salva em `models/` |
| 04 | `04_resultados.ipynb` | Análise de custo, SHAP e simulação da API |

### API em produção

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints disponíveis:**

```
GET  /health   →  status da API e do modelo
POST /prever   →  classifica uma transação
GET  /docs     →  documentação interativa (Swagger UI)
```

**Exemplo de chamada:**

```bash
curl -X POST http://localhost:8000/prever \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "V1": -1.35, "V2": -0.07, "V3": 2.53,
      "Amount_log": 5.01, "Hour": 22.0, "Is_night": 1
    }
  }'
```

```json
{
  "probabilidade_fraude": 0.9134,
  "eh_fraude": true,
  "threshold_usado": 0.30,
  "nivel_risco": "alto"
}
```

### Testes

```bash
# Roda todos os testes
pytest tests/ -v

# Com relatório de cobertura
pytest tests/ -v --cov=src --cov-report=term-missing
```

<br/>

## 🛠 Stack Tecnológica

<div align="center">

| Categoria | Tecnologias |
|:---|:---|
| **Linguagem** | Python 3.11+ |
| **ML / Modelagem** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Balanceamento** | imbalanced-learn (SMOTE) |
| **Explicabilidade** | SHAP |
| **Visualização** | Matplotlib, Seaborn |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Serialização** | Joblib |
| **Testes** | Pytest, pytest-cov |
| **Configuração** | python-dotenv |

</div>

<br/>

## 📁 Dados

O dataset é carregado automaticamente pela URL pública na primeira execução. Para uso local:

1. Baixe o CSV do [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Salve em `data/raw/creditcard.csv`
3. O notebook 01 detecta o arquivo local automaticamente

> Os dados brutos e processados estão no `.gitignore` — nunca são commitados.

<br/>

## 🔮 Próximos Passos

- [ ] Hyperparameter tuning com **Optuna** (mais eficiente que GridSearch)
- [ ] Monitoramento de **data drift** (o padrão de fraudes muda com o tempo)
- [ ] **Retrain automático** quando a performance cai abaixo de um threshold
- [ ] Containerização com **Docker** para deploy em qualquer nuvem
- [ ] Autenticação com **API Key** no endpoint de produção

<br/>

---

<div align="center">

Feito por [**Agatha C. Ruiz**](https://github.com/AgathaCRuiz)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" />

</div>
