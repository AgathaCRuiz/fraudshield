"""
src/models/evaluate.py
───────────────────────
Funções de avaliação de modelos e análise de custo financeiro.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from dotenv import load_dotenv

load_dotenv()

# Custos padrão — sobrescritos pelas variáveis de ambiente se existirem
COST_FN = float(os.getenv("COST_FALSE_NEGATIVE", 500))  # fraude não detectada
COST_FP = float(os.getenv("COST_FALSE_POSITIVE", 10))   # bloqueio indevido


def evaluate_model(model, X_test, y_test, model_name: str = "Modelo") -> dict:
    """
    Avalia um modelo supervisionado e retorna um dicionário de métricas.

    Parâmetros
    ----------
    model : objeto com .predict_proba() e .predict()
    X_test : array-like com as features de teste
    y_test : array-like com os rótulos reais
    model_name : str — nome para exibição

    Retorna
    -------
    dict com AUC-ROC, AUPRC, F1, probabilidades e predições
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc  = roc_auc_score(y_test, probs)
    ap   = average_precision_score(y_test, probs)
    f1   = f1_score(y_test, preds)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, preds, target_names=["Legítima", "Fraude"]))
    print(f"AUC-ROC : {auc:.4f}")
    print(f"AUPRC   : {ap:.4f}")

    return {
        "nome":   model_name,
        "auc":    auc,
        "auprc":  ap,
        "f1":     f1,
        "probs":  probs,
        "preds":  preds,
    }


def evaluate_isolation_forest(model, X_test, y_test) -> dict:
    """
    Avalia o Isolation Forest, que não tem predict_proba.
    Usa o score_samples negado como proxy de probabilidade de anomalia.
    """
    # score_samples retorna valores negativos: mais negativo = mais anômalo
    # Invertemos o sinal para que "maior = mais fraude"
    scores = -model.score_samples(X_test)

    raw_preds = model.predict(X_test)
    preds = (raw_preds == -1).astype(int)   # -1 → anomalia → fraude

    auc = roc_auc_score(y_test, scores)
    ap  = average_precision_score(y_test, scores)
    f1  = f1_score(y_test, preds)

    print(f"\n{'='*50}")
    print("  Isolation Forest (não supervisionado)")
    print(f"{'='*50}")
    print(classification_report(y_test, preds, target_names=["Legítima", "Fraude"]))
    print(f"AUC-ROC : {auc:.4f}")
    print(f"AUPRC   : {ap:.4f}")

    return {
        "nome":  "Isolation Forest",
        "auc":   auc,
        "auprc": ap,
        "f1":    f1,
        "probs": scores,
        "preds": preds,
    }


def cost_analysis(
    y_test,
    y_probs,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
) -> pd.DataFrame:
    """
    Analisa o custo financeiro total para cada threshold de decisão.

    Em dados desbalanceados, o threshold padrão (0.5) raramente é
    o financeiramente ótimo. Esta função ajuda a encontrar o melhor.

    Parâmetros
    ----------
    y_test   : rótulos reais
    y_probs  : probabilidades preditas pelo modelo
    cost_fn  : custo por fraude não detectada (Falso Negativo)
    cost_fp  : custo por bloqueio indevido (Falso Positivo)

    Retorna
    -------
    pd.DataFrame com colunas: threshold, precision, recall, f1, fn, fp, custo
    """
    print(f"Custo por FN (fraude não detectada): R${cost_fn:,.0f}")
    print(f"Custo por FP (bloqueio indevido):    R${cost_fp:,.0f}\n")

    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall":    recall_score(y_test, preds),
            "f1":        f1_score(y_test, preds),
            "fn":        int(fn),
            "fp":        int(fp),
            "tp":        int(tp),
            "custo":     fn * cost_fn + fp * cost_fp,
        })

    df = pd.DataFrame(rows)

    best = df.loc[df["custo"].idxmin()]
    print(f"✅ Threshold ótimo: {best['threshold']:.2f}")
    print(f"   Custo total:     R${best['custo']:,.0f}")
    print(f"   Recall:          {best['recall']:.4f}")
    print(f"   Precision:       {best['precision']:.4f}")
    print(f"   FN:              {int(best['fn'])} | FP: {int(best['fp'])}")

    return df


def summary_table(results: list[dict]) -> pd.DataFrame:
    """
    Gera uma tabela comparativa de todos os modelos avaliados.

    Parâmetros
    ----------
    results : lista de dicionários retornados por evaluate_model()

    Retorna
    -------
    pd.DataFrame formatado para exibição
    """
    rows = [
        {
            "Modelo":   r["nome"],
            "AUC-ROC":  round(r["auc"],   4),
            "AUPRC":    round(r["auprc"], 4),
            "F1 Fraude": round(r["f1"],   4),
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("AUPRC", ascending=False).reset_index(drop=True)
    df.index += 1  # começa do 1 em vez de 0
    return df
