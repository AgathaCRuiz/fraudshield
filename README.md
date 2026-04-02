# 🔍 Detecção de Fraudes em Transações

Projeto completo de Machine Learning para detecção de fraudes em cartões de crédito,
estruturado seguindo as melhores práticas da indústria.

## 📁 Estrutura do Projeto

```
fraud_detection/
│
├── data/
│   ├── raw/                  # Dados originais — nunca edite esses arquivos
│   └── processed/            # Dados após limpeza e feature engineering
│
├── notebooks/
│   ├── 01_exploracao.ipynb   # Análise exploratória (EDA)
│   ├── 02_features.ipynb     # Feature engineering
│   ├── 03_modelos.ipynb      # Treinamento e comparação de modelos
│   └── 04_resultados.ipynb   # Análise de resultados e SHAP
│
├── src/                      # Código Python reutilizável (importado pelos notebooks)
│   ├── data/
│   │   └── loader.py         # Carrega e divide os dados
│   ├── features/
│   │   └── engineering.py    # Criação de features
│   ├── models/
│   │   ├── train.py          # Treinamento dos modelos
│   │   └── evaluate.py       # Métricas e análise de custo
│   ├── visualization/
│   │   └── plots.py          # Funções de visualização reutilizáveis
│   └── api/
│       └── app.py            # API FastAPI para deploy
│
├── models/                   # Modelos serializados (.pkl)
├── reports/
│   └── figures/              # Gráficos exportados
├── tests/                    # Testes unitários
│   └── test_features.py
│
├── requirements.txt          # Dependências do projeto
├── .env.example              # Variáveis de ambiente (tokens, caminhos)
└── README.md
```

## 🚀 Como Começar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/fraud_detection.git
cd fraud_detection

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Abra os notebooks na ordem
# No VSCode: instale a extensão "Jupyter" e abra a pasta
```

## 📓 Ordem de Execução dos Notebooks

| # | Notebook | O que faz |
|---|---|---|
| 01 | `exploracao.ipynb` | Carrega os dados, visualiza distribuições, entende o problema |
| 02 | `features.ipynb` | Cria e valida as features, exporta para `data/processed/` |
| 03 | `modelos.ipynb` | Treina LR, RF, XGBoost, LightGBM, CatBoost e Isolation Forest |
| 04 | `resultados.ipynb` | Análise de custo, SHAP, escolhe o melhor modelo |

## 🛠 Stack

- **Python 3.11+**
- **scikit-learn** — modelos e métricas
- **XGBoost / LightGBM / CatBoost** — gradient boosting
- **imbalanced-learn** — SMOTE
- **SHAP** — explicabilidade
- **FastAPI + Uvicorn** — deploy da API
- **Matplotlib / Seaborn** — visualizações
- **Pytest** — testes unitários
