"""
Model selection utilities using the validation split.

Without ground truth, we use stability as the selection criterion:
the configuration with smaller anomaly-rate shift between training
reference and validation is preferred.
"""

import numpy as np
import pandas as pd

from src.utils.logger_utils import logger


def compute_val_stability_metrics(df_full, df_val, score_cols, percentile=95):
    """
    Compute validation-set stability metrics for configuration selection.

    For each score column, computes:
    - anomaly rate in training-reference vs validation
    - absolute rate delta (lower is more stable)
    - validation score dispersion (IQR)

    Args:
        df_full: Full DataFrame with all score columns.
        df_val: Validation DataFrame or validation index.
        score_cols: Score columns to compare.
        percentile: Percentile used as reference threshold.
    Returns:
        pd.DataFrame with ranked stability results.
    """
    val_idx = df_val.index if hasattr(df_val, "index") else df_val
    train_mask = ~df_full.index.isin(val_idx)

    results = []
    for score_col in score_cols:
        if score_col not in df_full.columns:
            continue

        train_scores = df_full.loc[train_mask, score_col].dropna().values
        val_scores = df_full.loc[val_idx, score_col].dropna().values

        if len(train_scores) == 0 or len(val_scores) == 0:
            continue

        thresh = np.percentile(train_scores, percentile)

        train_anomaly_rate = (train_scores >= thresh).mean()
        val_anomaly_rate = (val_scores >= thresh).mean()
        stability_delta = abs(val_anomaly_rate - train_anomaly_rate)

        val_p25 = np.percentile(val_scores, 25)
        val_p75 = np.percentile(val_scores, 75)
        val_iqr = val_p75 - val_p25

        results.append(
            {
                "config": score_col.replace("_score", ""),
                "score_col": score_col,
                "threshold_p95": round(float(thresh), 6),
                "train_anomaly_rate": round(float(train_anomaly_rate * 100), 2),
                "val_anomaly_rate": round(float(val_anomaly_rate * 100), 2),
                "stability_delta_pct": round(float(stability_delta * 100), 2),
                "val_iqr": round(float(val_iqr), 6),
                "n_val_evaluated": int(len(val_scores)),
            }
        )

    if not results:
        logger.warning("Nenhum score disponivel para selecao de configuracao")
        return pd.DataFrame()

    df_selection = pd.DataFrame(results).sort_values("stability_delta_pct")
    df_selection["rank_stability"] = range(1, len(df_selection) + 1)

    logger.info("=" * 80)
    logger.info("SELECAO DE CONFIGURACAO (validation set - estabilidade)")
    logger.info(
        "NOTA: stability_delta_pct = |taxa_anomalia_treino - taxa_anomalia_val|. "
        "Menor = mais estavel. NAO e Precision/Recall."
    )
    for _, row in df_selection.iterrows():
        logger.info(
            f"   #{int(row['rank_stability'])} {row['config']}: "
            f"delta={row['stability_delta_pct']:.2f}% "
            f"(treino={row['train_anomaly_rate']:.2f}%, val={row['val_anomaly_rate']:.2f}%)"
        )

    iso_candidates = df_selection[df_selection["config"].str.startswith("ISO")]
    hbos_candidates = df_selection[df_selection["config"].str.startswith("HBOS")]
    best_iso = iso_candidates.iloc[0] if not iso_candidates.empty else None
    best_hbos = hbos_candidates.iloc[0] if not hbos_candidates.empty else None

    if best_iso is not None:
        logger.info(
            f"   + Melhor config ISO:  {best_iso['config']} "
            f"(delta={best_iso['stability_delta_pct']:.2f}%)"
        )
    if best_hbos is not None:
        logger.info(
            f"   + Melhor config HBOS: {best_hbos['config']} "
            f"(delta={best_hbos['stability_delta_pct']:.2f}%)"
        )
    logger.info("=" * 80)

    return df_selection
