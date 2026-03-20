import numpy as np
import pandas as pd

from src.utils.ensemble_decision import (
    compute_ensemble_decision,
    compute_vehicle_risk_summary,
)


def test_compute_ensemble_decision_handles_nan_coverage():
    df = pd.DataFrame(
        {
            "ISO_n100_p95_label": [1.0, 0.0, np.nan, np.nan],
            "ISO_n200_p95_label": [1.0, 1.0, np.nan, np.nan],
            "HBOS_bins10_p95_label": [0.0, np.nan, np.nan, np.nan],
            "Temporal_Baseline_p95_label": [0.0, 1.0, np.nan, np.nan],
        }
    )

    out = compute_ensemble_decision(df.copy(), percentile=95)

    assert {
        "vote_iso",
        "vote_hbos",
        "vote_temp",
        "n_families_scored",
        "n_models_scored",
        "ensemble_vote_pct",
        "ensemble_alert",
        "iso_alert",
        "hbos_alert",
        "temp_alert",
    }.issubset(set(out.columns))
    assert out.loc[0, "n_models_scored"] == 4
    assert out.loc[0, "n_families_scored"] == 3
    assert out.loc[0, "ensemble_vote_pct"] == 1 / 3
    assert out.loc[0, "ensemble_alert"] == 0.0

    assert out.loc[3, "n_models_scored"] == 0
    assert out.loc[3, "n_families_scored"] == 0
    assert np.isnan(out.loc[3, "ensemble_vote_pct"])
    assert np.isnan(out.loc[3, "ensemble_alert"])


def test_compute_vehicle_risk_summary_generates_ranking():
    df = pd.DataFrame(
        {
            "placa": ["A", "A", "B", "C"],
            "ISO_n100_p95_label": [1.0, 0.0, 0.0, np.nan],
            "HBOS_bins10_p95_label": [1.0, 1.0, 0.0, np.nan],
        }
    )
    out = compute_ensemble_decision(df.copy(), percentile=95)
    summary = compute_vehicle_risk_summary(out, placa_col="placa")

    assert not summary.empty
    assert "ranking_risco" in summary.columns
    assert summary["ranking_risco"].min() == 1
