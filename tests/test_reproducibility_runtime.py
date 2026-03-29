import json
import random

import numpy as np
import pytest
import yaml

from src.pipeline import experiment_runner as er
from src.utils import tf_runtime
from src.utils import tracking


class _DummyTF:
    class random:
        called_with = None

        @staticmethod
        def set_seed(seed):
            _DummyTF.random.called_with = seed


def test_setup_deterministic_runtime_reseeds_python_numpy_and_tf(monkeypatch):
    monkeypatch.setattr(
        tf_runtime,
        "configure_tensorflow_runtime",
        lambda tf_device="auto": (
            _DummyTF,
            {
                "requested": tf_device,
                "active": "cpu",
                "gpu_count": 0,
                "gpu_names": [],
            },
        ),
    )

    _, runtime1 = tf_runtime.setup_deterministic_runtime(seed=123, tf_device="cpu")
    py_1 = random.random()
    np_1 = float(np.random.rand())

    _, runtime2 = tf_runtime.setup_deterministic_runtime(seed=123, tf_device="cpu")
    py_2 = random.random()
    np_2 = float(np.random.rand())

    assert runtime1["seed"] == 123
    assert runtime2["seed"] == 123
    assert runtime1["requested"] == "cpu"
    assert _DummyTF.random.called_with == 123
    assert py_1 == py_2
    assert np_1 == np_2


def test_run_experiment_uses_shared_runtime_setup(monkeypatch, tmp_path):
    captured = []
    statuses = []

    cfg = {
        "random_state": 777,
        "mapeamento_colunas": {
            "placa": "placa",
            "timestamp": "timestamp",
            "latitude": "latitude",
            "longitude": "longitude",
            "RA": "regiao_adm",
        },
        "parametros": {"temporal": {"epochs": 1}, "split_ratios": {"train": 0.6}},
        "configuracoes_gerais": {"report_required": False},
    }
    config_path = tmp_path / "cfg.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    monkeypatch.setattr(
        er,
        "setup_deterministic_runtime",
        lambda seed, tf_device="auto": (
            None,
            captured.append((seed, tf_device))
            or {
                "requested": tf_device,
                "active": "cpu",
                "gpu_count": 0,
                "gpu_names": [],
                "seed": seed,
            },
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
    monkeypatch.setattr(tracking, "init_experiment", lambda *a, **k: object())
    monkeypatch.setattr(tracking, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(tracking, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(tracking, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(
        tracking,
        "end_run",
        lambda status="FINISHED": statuses.append(status),
    )

    monkeypatch.setattr(
        er,
        "load_data",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop_after_runtime_setup")),
    )

    output_base = tmp_path / "outputs"
    run_id = "20260329_140000"
    with pytest.raises(RuntimeError, match="stop_after_runtime_setup"):
        er.run_experiment(
            config_path=str(config_path),
            input_path="dummy.csv",
            output_dir=str(output_base),
            run_id=run_id,
            tf_device="gpu",
        )

    assert captured == [(777, "gpu")]
    assert "FAILED" in statuses

    summary_path = output_base / run_id / "metrics" / "run_summary.json"
    assert summary_path.exists()
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["status"] == "FAILED"
    assert summary["failed_stage"] == "ETAPA 1: CARGA E PROCESSAMENTO DE DADOS"
