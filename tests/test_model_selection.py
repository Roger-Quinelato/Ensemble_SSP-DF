import pandas as pd

from src.utils.model_selection import compute_val_stability_metrics


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
