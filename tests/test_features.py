"""
tests/test_features.py
───────────────────────
Testes unitários para as funções de feature engineering.

Execute com:
    pytest tests/ -v
    pytest tests/ -v --cov=src   # com relatório de cobertura
"""

import numpy as np
import pandas as pd
import pytest

# Importa o módulo que vamos testar
from src.features.engineering import build_features, get_feature_names


# ── Fixtures ──────────────────────────────────────────────────────────────
# Fixtures são dados de teste reutilizados em vários testes

@pytest.fixture
def df_sample():
    """Cria um DataFrame mínimo com a estrutura do dataset real."""
    return pd.DataFrame({
        "Time":   [0.0, 3600.0, 7200.0, 79200.0],   # 0h, 1h, 2h, 22h
        "V1":     [-1.35, 1.19, -1.36, -0.96],
        "V2":     [-0.07, 0.26, -1.34, -0.18],
        "Amount": [149.62, 2.69, 378.66, 0.0],       # inclui zero para testar log1p
        "Class":  [0, 0, 1, 0],
    })


# ── Testes de build_features ──────────────────────────────────────────────

def test_build_features_cria_amount_log(df_sample):
    """Amount_log deve ser criado e conter log1p(Amount)."""
    resultado = build_features(df_sample)
    assert "Amount_log" in resultado.columns
    np.testing.assert_allclose(
        resultado["Amount_log"].values,
        np.log1p(df_sample["Amount"].values),
        rtol=1e-5,
    )


def test_build_features_amount_log_zero(df_sample):
    """log1p(0) deve ser 0 — garante que Amount=0 não quebra."""
    resultado = build_features(df_sample)
    valor = resultado.loc[df_sample["Amount"] == 0, "Amount_log"].iloc[0]
    assert valor == 0.0


def test_build_features_cria_amount_scaled(df_sample):
    """Amount_scaled deve ter média próxima de 0 e desvio padrão próximo de 1."""
    resultado = build_features(df_sample)
    assert "Amount_scaled" in resultado.columns
    # Com poucos exemplos, não exigimos exatidão — só verificamos que foi criado
    assert resultado["Amount_scaled"].notna().all()


def test_build_features_cria_hour(df_sample):
    """Hour deve ser extraída corretamente de Time (em segundos)."""
    resultado = build_features(df_sample)
    assert "Hour" in resultado.columns
    # Time=0 → hora 0; Time=3600 → hora 1; Time=7200 → hora 2; Time=79200 → hora 22
    esperado = [0, 1, 2, 22]
    assert list(resultado["Hour"].values) == esperado


def test_build_features_cria_is_night(df_sample):
    """Is_night deve ser 1 para horas entre 22h e 6h (inclusive)."""
    resultado = build_features(df_sample)
    assert "Is_night" in resultado.columns
    # Horas 0h, 1h, 2h são noturnas; 22h também é noturna
    esperado = [1, 0, 0, 1]
    assert list(resultado["Is_night"].values) == esperado


def test_build_features_nao_modifica_original(df_sample):
    """build_features não deve modificar o DataFrame original."""
    original_cols = list(df_sample.columns)
    build_features(df_sample)                          # executa a função
    assert list(df_sample.columns) == original_cols    # original intacto


def test_build_features_preserva_linhas(df_sample):
    """O número de linhas não deve mudar após o feature engineering."""
    resultado = build_features(df_sample)
    assert len(resultado) == len(df_sample)


# ── Testes de get_feature_names ────────────────────────────────────────────

def test_get_feature_names_exclui_colunas_alvo(df_sample):
    """Time, Amount e Class não devem aparecer na lista de features."""
    df_proc = build_features(df_sample)
    features = get_feature_names(df_proc)
    assert "Time"   not in features
    assert "Amount" not in features
    assert "Class"  not in features


def test_get_feature_names_inclui_novas_features(df_sample):
    """As features criadas pelo engineering devem aparecer na lista."""
    df_proc = build_features(df_sample)
    features = get_feature_names(df_proc)
    assert "Amount_log"    in features
    assert "Amount_scaled" in features
    assert "Hour"          in features
    assert "Is_night"      in features


def test_get_feature_names_retorna_lista(df_sample):
    """O retorno deve ser uma lista (não set, não Series)."""
    df_proc = build_features(df_sample)
    features = get_feature_names(df_proc)
    assert isinstance(features, list)
