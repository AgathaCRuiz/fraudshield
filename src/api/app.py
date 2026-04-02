"""
src/api/app.py
───────────────
API FastAPI para servir o modelo em produção.

Para rodar localmente:
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

Documentação interativa (gerada automaticamente pelo FastAPI):
    http://localhost:8000/docs
"""

import os
from typing import Dict

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# ── Carrega artefatos na inicialização ──────────────────────────────────────
# Fazemos isso uma vez ao subir a API, não a cada requisição (muito mais eficiente)
MODEL_PATH     = os.getenv("MODEL_PATH",     "models/modelo_fraude.pkl")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold_otimo.pkl")
FEATURES_PATH  = os.getenv("FEATURES_PATH",  "models/features.pkl")

try:
    modelo        = joblib.load(MODEL_PATH)
    threshold     = float(joblib.load(THRESHOLD_PATH))
    feature_names = joblib.load(FEATURES_PATH)
    print(f"✅ Modelo carregado | Threshold: {threshold:.2f} | Features: {len(feature_names)}")
except FileNotFoundError as e:
    # Em desenvolvimento, o modelo pode não existir ainda
    # A API ainda sobe, mas os endpoints de predição retornam 503
    print(f"⚠️  Artefatos não encontrados: {e}")
    modelo = threshold = feature_names = None


# ── Definição da aplicação ─────────────────────────────────────────────────
app = FastAPI(
    title="API de Detecção de Fraudes",
    description="Classifica transações financeiras como fraude ou legítima usando XGBoost.",
    version="1.0.0",
)


# ── Schemas de entrada e saída ─────────────────────────────────────────────
class TransacaoInput(BaseModel):
    """Dados de entrada: dicionário com o valor de cada feature."""

    features: Dict[str, float] = Field(
        ...,
        description="Valores das features da transação (mesmo nome e ordem do treino).",
        example={"V1": -1.35, "V2": -0.07, "Amount_log": 5.01, "Hour": 14.0},
    )


class PredictionOutput(BaseModel):
    """Dados de saída: resultado da predição com metadados."""

    probabilidade_fraude: float = Field(description="Score de risco de 0 a 1")
    eh_fraude: bool             = Field(description="Decisão com base no threshold otimizado")
    threshold_usado: float      = Field(description="Threshold de decisão aplicado")
    nivel_risco: str            = Field(description="baixo / médio / alto")


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Utilidades"])
def health_check():
    """Verifica se a API e o modelo estão operacionais."""
    return {
        "status": "ok" if modelo is not None else "degraded",
        "modelo_carregado": modelo is not None,
        "threshold": threshold,
        "n_features": len(feature_names) if feature_names else 0,
    }


@app.post("/prever", response_model=PredictionOutput, tags=["Predição"])
def prever(transacao: TransacaoInput):
    """
    Recebe os dados de uma transação e retorna a probabilidade de fraude.

    O threshold de decisão é o otimizado pela análise de custo financeiro
    (não o padrão 0.5 do scikit-learn).
    """
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o notebook 03_modelos.ipynb primeiro.",
        )

    # Garante que as features estão na mesma ordem que no treino
    # (ordem errada geraria predições incorretas silenciosamente)
    try:
        X = np.array([transacao.features[f] for f in feature_names]).reshape(1, -1)
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Feature ausente no payload: {e}. Features esperadas: {feature_names}",
        )

    prob      = float(modelo.predict_proba(X)[0][1])
    eh_fraude = prob >= threshold

    # Nível de risco para facilitar triagem humana
    if prob < 0.3:
        nivel = "baixo"
    elif prob < 0.7:
        nivel = "médio"
    else:
        nivel = "alto"

    return PredictionOutput(
        probabilidade_fraude=round(prob, 4),
        eh_fraude=eh_fraude,
        threshold_usado=round(threshold, 2),
        nivel_risco=nivel,
    )
