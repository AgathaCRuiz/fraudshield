"""
src/features/engineering.py
────────────────────────────
Criação e transformação de features.
Separar aqui garante que o mesmo pré-processamento
seja aplicado no treino, no teste e na API de produção.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as transformações de feature engineering ao DataFrame.

    Transforma as colunas originais e cria novas features derivadas.
    Não remove nenhuma coluna — isso fica para o loader.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com as colunas originais do dataset.

    Retorna
    -------
    pd.DataFrame
        DataFrame com as novas features adicionadas.
    """
    df = df.copy()  # nunca modifica o original (boa prática)

    # ── Transformação do valor da transação ───────────────────────────────
    # log1p(x) = log(x + 1): comprime a escala e funciona com Amount = 0
    df["Amount_log"] = np.log1p(df["Amount"])

    # Normalização: média 0, desvio padrão 1 — necessário para modelos lineares
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    # ── Features temporais ────────────────────────────────────────────────
    # Time está em segundos desde a primeira transação do dataset
    # Convertemos para hora do dia (0–23) para capturar padrões horários
    df["Hour"] = (df["Time"] // 3600) % 24

    # Transações noturnas são estatisticamente mais suspeitas
    df["Is_night"] = ((df["Hour"] >= 22) | (df["Hour"] <= 6)).astype(int)

    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """
    Retorna a lista de features que devem ser usadas no modelo.
    Exclui as colunas originais que foram substituídas por versões processadas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame após build_features().

    Retorna
    -------
    list[str]
        Lista com os nomes das colunas a usar como input do modelo.
    """
    # Removemos Time, Amount (originais) e Class (alvo)
    excluir = {"Time", "Amount", "Class"}
    return [col for col in df.columns if col not in excluir]
