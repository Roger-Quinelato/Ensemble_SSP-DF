import csv
import json

import numpy as np
import pandas as pd
import yaml

from src.pipeline import experiment_runner as er
from src.utils import tracking
from src.pipeline.experiment_runner import _resolve_primary_percentile


def test_resolve_primary_percentile_prefers_95_when_available():
    cfg = {"parametros": {"percentis_teste": [90, 95, 99]}}
    assert _resolve_primary_percentile(cfg) == 95


def test_resolve_primary_percentile_uses_first_when_95_absent():
    cfg = {"parametros": {"percentis_teste": [90, 99]}}
    assert _resolve_primary_percentile(cfg) == 90


def test_resolve_primary_percentile_falls_back_to_95():
    cfg = {"parametros": {"percentis_teste": ["x", None]}}
    assert _resolve_primary_percentile(cfg) == 95


def test_resolve_primary_percentile_reuses_operational_resolution_normalization():
    cfg = {"parametros": {"percentis_teste": ["99", "x", 90, 90]}}

    resolved = _resolve_primary_percentile(cfg)

    assert resolved == 99
    assert cfg["parametros"]["percentis_teste"] == [99, 90]


def test_resolve_main_scaler_features_prefers_fitted_scaler_schema():
    class _Scaler:
        feature_names_in_ = np.array(["hora_sin", "hora_cos", "RA_Plano"])

    class _Proc:
        scaler = _Scaler()
        features_to_use = ["legacy_feature"]

    resolved = er._resolve_main_scaler_features(_Proc())
    assert resolved == ["hora_sin", "hora_cos", "RA_Plano"]


def test_resolve_main_scaler_features_falls_back_to_proc_features():
    class _Proc:
        scaler = object()
        features_to_use = ["f1", "f2"]

    resolved = er._resolve_main_scaler_features(_Proc())
    assert resolved == ["f1", "f2"]


def test_train_base_models_uses_operational_percentile_for_masks(monkeypatch, tmp_path):
    class _DummyISOModel:
        def score_samples(self, x):
            return -np.asarray([0.1] * len(x))

    class _DummyHBOSModel:
        def decision_function(self, x):
            return np.asarray([0.2] * len(x))

    class _DummyBaselineModels:
        def __init__(self, *args, **kwargs):
            pass

        def train_iso(self, **kwargs):
            return _DummyISOModel()

        def train_hbos(self, **kwargs):
            return _DummyHBOSModel()

    class _DummyThresholdOptimizer:
        def __init__(self, percentiles):
            self.percentiles = list(percentiles)

        def apply_dynamic_thresholds(self, df, score_col, model_name, calibration_scores=None):
            for p in self.percentiles:
                df[f"{model_name}_p{p}_label"] = 0.0
            metrics = [
                {
                    "Model": model_name,
                    "Percentile": p,
                    "Threshold_Value": 0.5,
                }
                for p in self.percentiles
            ]
            return df, metrics

    monkeypatch.setattr(er, "BaselineModels", _DummyBaselineModels)
    monkeypatch.setattr(er, "ThresholdOptimizer", _DummyThresholdOptimizer)
    monkeypatch.setattr(er.joblib, "dump", lambda *a, **k: None)

    df = pd.DataFrame({"registro": [1, 2, 3]})
    features = {
        "x_iso_train": np.ones((3, 2)),
        "x_iso_all": np.ones((3, 2)),
        "x_hbos_train": np.ones((3, 2)),
        "x_hbos_all": np.ones((3, 2)),
    }
    config = {
        "random_state": 42,
        "parametros": {
            "percentis_teste": [90, 99],
            "isolation_forest": {"n_estimators": [100], "contamination": "auto"},
            "hbos": {"n_bins": [10], "contamination": 0.1},
        }
    }

    df_out, iso_masks, hbos_masks, _, _ = er.train_base_models(
        df,
        features,
        config,
        str(tmp_path),
        operational_percentile=90,
    )

    assert "ISO_n100_p90_label" in df_out.columns
    assert "HBOS_bins10_p90_label" in df_out.columns
    assert "ISO_n100_p95_label" not in df_out.columns
    assert "HBOS_bins10_p95_label" not in df_out.columns
    assert iso_masks["ISO_n100"].equals(df_out["ISO_n100_p90_label"] == 0)
    assert hbos_masks["HBOS_bins10"].equals(df_out["HBOS_bins10_p90_label"] == 0)


def test_run_experiment_records_operational_percentile_semantics(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.yaml"
    cfg = {
        "random_state": 42,
        "mapeamento_colunas": {
            "placa": "placa",
            "timestamp": "timestamp",
            "latitude": "latitude",
            "longitude": "longitude",
            "RA": "regiao_adm",
        },
        "parametros": {
            "split_ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "percentis_teste": [90, 99],
            "temporal": {"epochs": 1},
        },
        "configuracoes_gerais": {"report_required": False},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    class _DummyProc:
        def __init__(self, _config):
            self.models_dir = None
            self.features_to_use = []

    base_df = pd.DataFrame(
        {
            "placa": ["A", "B", "C"],
            "timestamp": pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:10:00"]
            ),
            "latitude": [-15.8, -15.81, -15.82],
            "longitude": [-47.9, -47.91, -47.92],
            "ensemble_alert": [1.0, 0.0, 1.0],
            "n_models_scored": [1, 1, 1],
        }
    )

    metrics_logged = []
    monkeypatch.setattr(er, "DataProcessor", _DummyProc)
    monkeypatch.setattr(
        er,
        "setup_deterministic_runtime",
        lambda seed, tf_device="auto": (
            None,
            {"requested": tf_device, "active": "cpu", "gpu_count": 0, "seed": seed},
        ),
    )
    monkeypatch.setattr(
        er,
        "get_git_info",
        lambda: {
            "commit_hash": "abc123",
            "commit_short": "abc123",
            "branch": "dev",
            "is_dirty": False,
            "dirty_warning": None,
            "commit_message": "test",
            "commit_timestamp": "2026-03-29T00:00:00",
        },
    )
    monkeypatch.setattr(er, "format_model_version", lambda *a, **k: "v-test")
    monkeypatch.setattr(
        er,
        "load_data",
        lambda proc, config, input_path: (
            base_df.copy(),
            base_df.copy(),
            base_df.copy(),
            base_df.copy(),
            proc,
            {"split_temporal": {"train_end_index": 1}, "total_veiculos": 3, "periodo": "ok"},
        ),
    )
    monkeypatch.setattr(
        er,
        "prepare_model_features",
        lambda *a, **k: {"iso_features": [], "hbos_features": [], "gru_features": []},
    )
    monkeypatch.setattr(
        er,
        "train_base_models",
        lambda df, features_dict, config, models_dir, operational_percentile: (
            df,
            {"ISO_n100": pd.Series([True, True, True])},
            {"HBOS_bins10": pd.Series([True, True, True])},
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        er,
        "train_temporal_models",
        lambda df, *a, **k: (df, [], []),
    )
    monkeypatch.setattr(
        er,
        "export_results",
        lambda *a, **k: {
            "report_required": False,
            "report_status": "failed",
            "report_error": "optional",
            "report_path": "dummy",
        },
    )

    monkeypatch.setattr(tracking, "init_experiment", lambda *a, **k: object())
    monkeypatch.setattr(tracking, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(tracking, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(tracking, "end_run", lambda *a, **k: None)
    monkeypatch.setattr(tracking, "log_metrics", lambda payload, step=None: metrics_logged.append(payload))

    output_base = tmp_path / "outputs"
    run_id = "20260329_150000"
    er.run_experiment(
        config_path=str(config_path),
        input_path="dummy.csv",
        output_dir=str(output_base),
        run_id=run_id,
    )

    summary_path = output_base / run_id / "metrics" / "run_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["parameters"]["operational_percentile"] == 90
    assert summary["results_summary"]["operational_percentile"] == 90
    assert summary["results_summary"]["n_alerts_operational"] == 2
    assert summary["results_summary"]["n_alerts_p95"] == "N/A"

    runs_index = output_base / "runs_index.csv"
    with open(runs_index, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[-1]["operational_percentile"] == "90"
    assert rows[-1]["n_alerts_operational"] == "2"
    assert rows[-1]["n_alerts_p95"] == "N/A"

    assert metrics_logged
    assert "n_alerts_operational" in metrics_logged[-1]
    assert "n_alerts_p95" not in metrics_logged[-1]
