"""
src/models/train.py
────────────────────
Define e treina todos os modelos do projeto.
Cada modelo retorna um objeto já treinado e pronto para avaliação.
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def compute_scale_pos_weight(y_train) -> float:
    """
    Calcula o scale_pos_weight para modelos de gradient boosting.
    É a razão entre negativos e positivos — quanto maior, mais o modelo
    penaliza erros na classe minoritária (fraude).
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"scale_pos_weight = {spw:.1f}  ({neg:,} legítimas / {pos} fraudes)")
    return spw


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Aplica SMOTE apenas no conjunto de treino.
    Cria amostras sintéticas da classe minoritária para balancear.

    ⚠️ Nunca aplique SMOTE no conjunto de teste —
    ele deve refletir a distribuição real do mundo.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"Após SMOTE: {len(y_res):,} amostras | {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """
    Regressão Logística com StandardScaler em Pipeline.

    Pipeline garante que a normalização seja aplicada automaticamente,
    tanto no treino quanto na predição — evita data leakage.

    class_weight='balanced': multiplica o peso dos erros
    na classe fraude proporcionalmente ao desbalanceamento.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,            # evita ConvergenceWarning
            class_weight="balanced",  # lida com desbalanceamento
            solver="lbfgs",
            random_state=random_state,
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("Regressão Logística treinada ✅")
    return pipeline


def train_random_forest(X_res, y_res, random_state: int = 42):
    """
    Random Forest treinado com dados balanceados via SMOTE.

    n_jobs=-1: usa todos os núcleos disponíveis (treinamento paralelo).
    class_weight='balanced': peso adicional além do SMOTE.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_res, y_res)
    print("Random Forest treinado ✅")
    return model


def train_xgboost(X_train, y_train, X_test, y_test, spw: float, random_state: int = 42):
    """
    XGBoost com scale_pos_weight para compensar o desbalanceamento.

    eval_metric='aucpr': otimiza a AUPRC durante o treino,
    mais adequada para dados desbalanceados do que logloss.

    subsample e colsample_bytree < 1: técnica de regularização
    para reduzir overfitting.
    """
    model = XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="aucpr",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("XGBoost treinado ✅")
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, spw: float, random_state: int = 42):
    """
    LightGBM — mais rápido que XGBoost em datasets grandes.

    Usa estratégia leaf-wise (cresce pela folha com maior ganho)
    em vez de level-wise, o que reduz o erro mais rapidamente.

    num_leaves=31: controla complexidade das árvores (padrão do LightGBM).
    """
    model = LGBMClassifier(
        scale_pos_weight=spw,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,             # suprime logs de progresso
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    print("LightGBM treinado ✅")
    return model


def train_catboost(X_train, y_train, spw: float, random_state: int = 42):
    """
    CatBoost — excelente para variáveis categóricas (não temos aqui, mas vale saber).

    Usa ordered boosting: embaralha as amostras em cada iteração
    para reduzir overfitting sem precisar de conjunto de validação.
    """
    model = CatBoostClassifier(
        scale_pos_weight=spw,
        iterations=200,
        learning_rate=0.05,
        depth=5,
        random_state=random_state,
        verbose=0,              # suprime logs de progresso
    )
    model.fit(X_train, y_train)
    print("CatBoost treinado ✅")
    return model


def train_isolation_forest(X_train, y_train, random_state: int = 42):
    """
    Isolation Forest — único modelo NÃO supervisionado do projeto.

    Aprende apenas com transações normais e detecta anomalias
    por terem comportamento diferente do padrão.

    Quando usar:
    - Quando não há labels de fraude disponíveis
    - Como segundo modelo para validar os supervisionados
    - Para detectar novos tipos de fraude nunca vistos antes

    contamination: proporção esperada de anomalias no dataset.
    Aqui usamos a proporção real de fraudes do treino.
    """
    contamination = float(y_train.mean())

    # Treinamos APENAS com transações normais (sem fraudes)
    X_normais = X_train[y_train == 0]

    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_normais)
    print(f"Isolation Forest treinado ✅  (contamination={contamination:.4f})")
    return model
