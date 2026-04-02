"""
src/data/loader.py
──────────────────
Responsável por carregar e dividir os dados brutos.
Centralizar aqui significa que todos os notebooks usam
a mesma lógica de carregamento — troca o caminho em um
lugar só.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()  # lê variáveis do arquivo .env

# URL padrão — pode ser sobrescrita pela variável de ambiente DATA_URL
DEFAULT_URL = os.getenv(
    "DATA_URL",
    "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
)

# Colunas que serão removidas antes de treinar
# (Time e Amount originais são substituídas por versões processadas)
COLS_DROP = ["Time", "Amount"]
TARGET_COL = "Class"


def load_raw(path_or_url: str = DEFAULT_URL) -> pd.DataFrame:
    """
    Carrega o CSV bruto a partir de um caminho local ou URL.

    Parâmetros
    ----------
    path_or_url : str
        Caminho para um arquivo .csv local ou URL pública.

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados brutos, sem nenhuma transformação.
    """
    print(f"Carregando dados de: {path_or_url}")
    df = pd.read_csv(path_or_url)
    print(f"  Shape: {df.shape} | Fraudes: {df[TARGET_COL].sum()} ({df[TARGET_COL].mean()*100:.2f}%)")
    return df


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Separa features (X) e alvo (y) e divide em treino/teste.

    Usa stratify=y para garantir a mesma proporção de fraudes
    em ambos os conjuntos — importante em dados desbalanceados.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame já com as features processadas.
    test_size : float
        Proporção do conjunto de teste (padrão: 20%).
    random_state : int
        Semente para reprodutibilidade.

    Retorna
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(COLS_DROP + [TARGET_COL], axis=1, errors="ignore")
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"Treino: {len(X_train):,} amostras | Fraudes: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Teste:  {len(X_test):,} amostras  | Fraudes: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test
