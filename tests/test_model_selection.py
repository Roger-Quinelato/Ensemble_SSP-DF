import pandas as pd
import pytest

from src.utils.model_selection import (
    compute_temporal_strategy_validation,
    compute_val_stability_metrics,
    normalize_temporal_strategy,
)


def test_compute_val_stability_metrics_ranks_configs():
    df_train = pd.DataFrame(
        {
            "ISO_n100_score": [0.1, 0.2, 0.3, 0.4],
            "ISO_n200_score": [0.1, 0.2, 0.31, 0.39],
            "HBOS_bins10_score": [1.0, 1.1, 1.2, 1.3],
            "HBOS_bins20_score": [1.0, 1.15, 1.18, 1.32],
        }
    )
    df_val = pd.DataFrame(
        {
            "ISO_n100_score": [0.8, 0.9],
            "ISO_n200_score": [0.81, 0.89],
            "HBOS_bins10_score": [2.0, 2.1],
            "HBOS_bins20_score": [1.95, 2.2],
        }
    )
    score_cols = [c for c in df_train.columns if c.endswith("_score")]

    out = compute_val_stability_metrics(
        df_train=df_train, df_val=df_val, score_cols=score_cols, percentile=95
    )

    assert not out.empty
    assert {"config", "stability_delta_pct", "train_anomaly_rate", "val_anomaly_rate", "rank_stability"}.issubset(
        out.columns
    )
    assert out["rank_stability"].iloc[0] == 1
    assert out["stability_delta_pct"].between(0, 100).all()


def test_compute_val_stability_metrics_uses_requested_percentile_column():
    df_train = pd.DataFrame({"ISO_n100_score": [0.1, 0.2, 0.3, 0.4]})
    df_val = pd.DataFrame({"ISO_n100_score": [0.35, 0.45]})

    out = compute_val_stability_metrics(
        df_train=df_train,
        df_val=df_val,
        score_cols=["ISO_n100_score"],
        percentile=90,
    )

    assert "threshold_p90" in out.columns


def test_compute_temporal_strategy_validation_ranks_strategies():
    df_train = pd.DataFrame(
        {
            "Temporal_Union_A_score": [0.1, 0.2, 0.3, 0.4],
            "Temporal_Inter_A_score": [0.1, 0.1, 0.1, 0.1],
            "Temporal_Baseline_score": [0.2, 0.25, 0.3, 0.35],
        }
    )
    df_val = pd.DataFrame(
        {
            "Temporal_Union_A_score": [0.45, 0.5],
            "Temporal_Inter_A_score": [0.09, 0.11],
            "Temporal_Baseline_score": [0.33, 0.37],
        }
    )
    out = compute_temporal_strategy_validation(
        df_train=df_train,
        df_val=df_val,
        score_cols=list(df_train.columns),
        percentile=95,
    )
    assert not out.empty
    assert {"temporal_strategy", "rank_temporal_strategy", "mean_stability_delta_pct"}.issubset(
        out.columns
    )
    assert out["rank_temporal_strategy"].iloc[0] == 1


def test_normalize_temporal_strategy_default_and_invalid():
    assert normalize_temporal_strategy(None) == "all"
    assert normalize_temporal_strategy("UNION") == "union"
    with pytest.raises(ValueError, match="temporal_strategy invalida"):
        normalize_temporal_strategy("foo")
