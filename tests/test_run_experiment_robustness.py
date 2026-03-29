import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pipeline import experiment_runner as er
from src.utils import tracking


def _write_min_config(path, report_required=True):
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
            "percentis_teste": [90, 95, 99],
            "isolation_forest": {"n_estimators": [100], "contamination": "auto"},
            "hbos": {"n_bins": [10], "contamination": 0.1},
            "temporal": {"arch_type": "gru", "window_size": 3, "epochs": 1},
        },
        "configuracoes_gerais": {
            "gap_segmentation_seconds": 1800,
            "report_required": report_required,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _patch_tracking(monkeypatch, status_sink):
    monkeypatch.setattr(tracking, "init_experiment", lambda *a, **k: object())
    monkeypatch.setattr(tracking, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(tracking, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(tracking, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(
        tracking,
        "end_run",
        lambda status="FINISHED": status_sink.append(status),
    )


def _patch_git(monkeypatch):
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


def _patch_runtime(monkeypatch, expected_seed=42, expected_device="auto"):
    def _fake_setup(seed, tf_device="auto"):
        assert int(seed) == int(expected_seed)
        assert tf_device == expected_device
        return None, {
            "requested": tf_device,
            "active": "cpu",
            "gpu_count": 0,
            "gpu_names": [],
            "seed": seed,
        }

    monkeypatch.setattr(er, "setup_deterministic_runtime", _fake_setup)


def _patch_pipeline_happy_path(monkeypatch):
    df_stub = pd.DataFrame(
        {
            "placa": ["ABC1234"],
            "timestamp": [pd.Timestamp("2026-01-01 00:00:00")],
            "latitude": [-15.8],
            "longitude": [-47.9],
            "ensemble_alert": [0.0],
            "n_models_scored": [1],
        }
    )

    def fake_load_data(proc, config, input_path):
        stats = {
            "split_temporal": {"train_end_index": 0},
            "total_veiculos": 1,
            "periodo": "2026-01-01 a 2026-01-01",
        }
        return df_stub.copy(), df_stub.copy(), df_stub.copy(), df_stub.copy(), proc, stats

    monkeypatch.setattr(er, "load_data", fake_load_data)
    monkeypatch.setattr(
        er,
        "prepare_model_features",
        lambda *a, **k: {
            "iso_features": [],
            "hbos_features": [],
            "gru_features": [],
        },
    )
    monkeypatch.setattr(
        er,
        "train_base_models",
        lambda df, *a, **k: (df, {"ISO_n100": pd.Series([True])}, {"HBOS_bins10": pd.Series([True])}, [], []),
    )
    monkeypatch.setattr(
        er,
        "train_temporal_models",
        lambda df, *a, **k: (df, [], []),
    )


def test_run_experiment_persists_failure_summary_and_failed_tracking(monkeypatch, tmp_path):
    statuses = []
    _patch_tracking(monkeypatch, statuses)
    _patch_git(monkeypatch)
    _patch_runtime(monkeypatch, expected_seed=42, expected_device="auto")

    config_path = tmp_path / "cfg.yaml"
    _write_min_config(config_path, report_required=True)

    monkeypatch.setattr(
        er,
        "load_data",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("falha_stage1")),
    )

    output_base = tmp_path / "outputs"
    run_id = "20260329_120000"
    with pytest.raises(RuntimeError, match="falha_stage1"):
        er.run_experiment(
            config_path=str(config_path),
            input_path="dummy.csv",
            output_dir=str(output_base),
            run_id=run_id,
        )

    run_dir = output_base / run_id
    summary_path = run_dir / "metrics" / "run_summary.json"
    assert summary_path.exists()

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "FAILED"
    assert summary["failed_stage"] == "ETAPA 1: CARGA E PROCESSAMENTO DE DADOS"
    assert "falha_stage1" in (summary.get("error_message") or "")
    assert summary["report_status"] == "not_attempted"
    assert "FAILED" in statuses


def test_run_experiment_fails_when_report_is_required(monkeypatch, tmp_path):
    statuses = []
    _patch_tracking(monkeypatch, statuses)
    _patch_git(monkeypatch)
    _patch_runtime(monkeypatch, expected_seed=42, expected_device="auto")
    _patch_pipeline_happy_path(monkeypatch)

    config_path = tmp_path / "cfg_required.yaml"
    _write_min_config(config_path, report_required=True)

    monkeypatch.setattr(
        er,
        "export_results",
        lambda *a, **k: {
            "report_required": True,
            "report_status": "failed",
            "report_error": "report_boom",
            "report_path": "dummy",
        },
    )

    output_base = tmp_path / "outputs_req"
    run_id = "20260329_120001"
    with pytest.raises(RuntimeError, match="Relatorio HTML obrigatorio"):
        er.run_experiment(
            config_path=str(config_path),
            input_path="dummy.csv",
            output_dir=str(output_base),
            run_id=run_id,
        )

    summary_path = output_base / run_id / "metrics" / "run_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "FAILED"
    assert summary["failed_stage"] == "ETAPA 5: EXPORTACAO DE RESULTADOS"
    assert summary["report_required"] is True
    assert summary["report_status"] == "failed"
    assert summary["report_error"] == "report_boom"
    assert "FAILED" in statuses


def test_run_experiment_allows_report_failure_when_not_required(monkeypatch, tmp_path):
    statuses = []
    _patch_tracking(monkeypatch, statuses)
    _patch_git(monkeypatch)
    _patch_runtime(monkeypatch, expected_seed=42, expected_device="auto")
    _patch_pipeline_happy_path(monkeypatch)

    config_path = tmp_path / "cfg_optional.yaml"
    _write_min_config(config_path, report_required=False)

    monkeypatch.setattr(
        er,
        "export_results",
        lambda *a, **k: {
            "report_required": False,
            "report_status": "failed",
            "report_error": "report_optional_error",
            "report_path": "dummy",
        },
    )

    output_base = tmp_path / "outputs_opt"
    run_id = "20260329_120002"
    returned_dir = er.run_experiment(
        config_path=str(config_path),
        input_path="dummy.csv",
        output_dir=str(output_base),
        run_id=run_id,
    )

    assert returned_dir.endswith(run_id)

    summary_path = output_base / run_id / "metrics" / "run_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "SUCCESS"
    assert summary["report_required"] is False
    assert summary["report_status"] == "failed"
    assert summary["report_error"] == "report_optional_error"
    assert "FINISHED" in statuses
