"""
src/visualization/plots.py
───────────────────────────
Funções de visualização reutilizáveis entre notebooks.
Centralizar aqui evita copiar e colar o mesmo código de plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

# Paleta de cores consistente em todos os gráficos do projeto
CORES = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]


def plot_class_distribution(y: pd.Series, figsize=(12, 4)) -> None:
    """
    Plota a distribuição das classes e o desbalanceamento.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Gráfico de barras com contagem
    contagem = y.value_counts()
    contagem.plot(kind="bar", ax=axes[0], color=["#2196F3", "#F44336"], edgecolor="black")
    axes[0].set_title("Distribuição das Classes", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Classe (0=Legítima, 1=Fraude)")
    axes[0].set_ylabel("Quantidade")
    axes[0].tick_params(axis="x", rotation=0)
    for i, v in enumerate(contagem):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")

    # Gráfico de pizza com proporção
    axes[1].pie(
        contagem,
        labels=["Legítima", "Fraude"],
        autopct="%1.3f%%",
        colors=["#2196F3", "#F44336"],
        startangle=90,
    )
    axes[1].set_title("Proporção das Classes", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_roc_pr_curves(results: list[dict], figsize=(14, 5)) -> None:
    """
    Plota curva ROC e curva Precision-Recall lado a lado para todos os modelos.

    Parâmetros
    ----------
    results : lista de dicts com chaves 'nome', 'probs', 'preds'
              (saída de evaluate_model / evaluate_isolation_forest)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for (res, cor) in zip(results, CORES):
        y_test = res["y_test"]
        probs  = res["probs"]
        nome   = res["nome"]

        # ── Curva ROC ──────────────────────────────────
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, color=cor, lw=2, label=f"{nome} ({auc:.3f})")

        # ── Curva PR ───────────────────────────────────
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        axes[1].plot(rec, prec, color=cor, lw=2, label=f"{nome} ({ap:.3f})")

    # Linha de baseline ROC
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Baseline (0.500)")
    axes[0].set_title("Curva ROC", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Taxa de Falso Positivo")
    axes[0].set_ylabel("Taxa de Verdadeiro Positivo")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Linha de baseline PR (proporção real de fraudes)
    baseline_pr = results[0]["y_test"].mean()
    axes[1].axhline(baseline_pr, color="k", linestyle="--", lw=1, label=f"Baseline ({baseline_pr:.3f})")
    axes[1].set_title("Curva Precision-Recall", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results: list[dict], ncols: int = 3, figsize=(18, 10)) -> None:
    """
    Plota matrizes de confusão para todos os modelos em um grid.
    """
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, res in zip(axes_flat, results):
        cm = confusion_matrix(res["y_test"], res["preds"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legítima", "Fraude"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(
            f"{res['nome']}\nTP={tp} | FP={fp} | FN={fn} | TN={tn}",
            fontsize=10,
            fontweight="bold",
        )

    # Esconde subplots extras se o número de modelos não preencher o grid
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle("Matrizes de Confusão", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_cost_analysis(df_thresh: pd.DataFrame, best_threshold: float, figsize=(14, 5)) -> None:
    """
    Plota o custo financeiro e as métricas de classificação por threshold.

    Parâmetros
    ----------
    df_thresh       : DataFrame retornado por cost_analysis()
    best_threshold  : threshold com o menor custo (linha vertical verde)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Custo total ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(df_thresh["threshold"], df_thresh["custo"] / 1000, "o-", color="#E74C3C", lw=2)
    ax.axvline(best_threshold, color="green", linestyle="--", lw=2,
               label=f"Threshold ótimo = {best_threshold:.2f}")
    ax.set_title("Custo Financeiro Total por Threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("Threshold de Decisão")
    ax.set_ylabel("Custo Total (R$ mil)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Precision / Recall / F1 ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(df_thresh["threshold"], df_thresh["precision"], "b-o", lw=2, label="Precision")
    ax.plot(df_thresh["threshold"], df_thresh["recall"],    "r-o", lw=2, label="Recall")
    ax.plot(df_thresh["threshold"], df_thresh["f1"],        "g-o", lw=2, label="F1")
    ax.axvline(best_threshold, color="purple", linestyle="--", lw=2,
               label=f"Threshold ótimo = {best_threshold:.2f}")
    ax.set_title("Precision, Recall e F1 por Threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("Threshold de Decisão")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list[str], top_n: int = 20, figsize=(10, 7)) -> None:
    """
    Plota a importância das features de um modelo tree-based (XGBoost, RF, etc.).
    """
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="black")
    ax.set_title(f"Top {top_n} Features mais Importantes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importância")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
