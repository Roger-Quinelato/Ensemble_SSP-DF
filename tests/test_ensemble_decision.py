import numpy as np
import pandas as pd

from src.utils.ensemble_decision import (
    compute_ensemble_decision,
    compute_vehicle_risk_summary,
)


def test_compute_ensemble_decision_handles_nan_coverage():
    df = pd.DataFrame(
        {
            "model_a_p95_label": [1.0, 0.0, np.nan, np.nan],
            "model_b_p95_label": [1.0, np.nan, np.nan, np.nan],
            "model_c_p95_label": [0.0, 1.0, np.nan, np.nan],
        }
    )

    out = compute_ensemble_decision(df.copy(), percentile=95)

    assert {"n_models_scored", "ensemble_vote_count", "ensemble_vote_pct", "ensemble_alert"}.issubset(
        set(out.columns)
    )
    assert out.loc[0, "n_models_scored"] == 3
    assert out.loc[0, "ensemble_vote_count"] == 2
    assert out.loc[0, "ensemble_vote_pct"] == 2 / 3
    assert out.loc[0, "ensemble_alert"] == 1.0

    assert out.loc[3, "n_models_scored"] == 0
    assert np.isnan(out.loc[3, "ensemble_vote_pct"])
    assert np.isnan(out.loc[3, "ensemble_alert"])


def test_compute_vehicle_risk_summary_generates_ranking():
    df = pd.DataFrame(
        {
            "placa": ["A", "A", "B", "C"],
            "model_a_p95_label": [1.0, 0.0, 0.0, np.nan],
            "model_b_p95_label": [1.0, 1.0, 0.0, np.nan],
        }
    )
    out = compute_ensemble_decision(df.copy(), percentile=95)
    summary = compute_vehicle_risk_summary(out, placa_col="placa")

    assert not summary.empty
    assert "ranking_risco" in summary.columns
    assert summary["ranking_risco"].min() == 1
