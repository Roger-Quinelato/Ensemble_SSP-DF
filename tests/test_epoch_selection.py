"""
Teste opcional para varrer multiplos valores de epoch e sugerir o melhor.

Por padrao fica SKIPPED para nao pesar no CI.
Para executar manualmente:

    $env:RUN_EPOCH_SWEEP="1"
    $env:EPOCH_SWEEP_VALUES="10,20,50"
    pytest tests/test_epoch_selection.py -v -s
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _parse_epochs(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Epoch invalida: {value}")
        values.append(value)
    if not values:
        raise ValueError("Lista de epochs vazia.")
    return values


def _latest_run_dir(output_base: Path) -> Path:
    run_pattern = re.compile(r"^\d{8}_\d{6}$")
    runs = sorted(
        [d for d in output_base.iterdir() if d.is_dir() and run_pattern.match(d.name)],
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"Nenhuma run encontrada em {output_base}")
    return runs[0]


def _score_epoch(temporal_metrics_path: Path) -> dict:
    df = pd.read_csv(temporal_metrics_path)
    if df.empty:
        raise ValueError(f"{temporal_metrics_path} esta vazio")

    df_p95 = df[df["Percentile"] == 95].copy()
    if df_p95.empty:
        raise ValueError("temporal_metrics.csv nao possui linha de Percentile=95")

    # Heuristica operacional sem ground truth:
    # 1) queremos taxa de anomalia p95 perto de 5% entre avaliados;
    # 2) queremos baixa dispersao entre modelos temporais.
    target = 5.0
    mean_pct = float(df_p95["Pct_Anomalies_Of_Evaluated"].mean())
    std_pct = float(df_p95["Pct_Anomalies_Of_Evaluated"].std(ddof=0))
    objective = abs(mean_pct - target) + (0.25 * std_pct)
    return {
        "mean_pct_anomaly_eval_p95": round(mean_pct, 4),
        "std_pct_anomaly_eval_p95": round(std_pct, 4),
        "objective_score": round(objective, 6),
    }


@pytest.mark.slow
def test_epoch_sweep_selects_best_epoch(tmp_path):
    if os.environ.get("RUN_EPOCH_SWEEP", "0") != "1":
        pytest.skip("Defina RUN_EPOCH_SWEEP=1 para executar a varredura de epochs.")

    input_path = Path(os.environ.get("EPOCH_SWEEP_INPUT", "data/input/amostra_ssp.csv"))
    if not input_path.exists():
        pytest.skip(f"Arquivo de entrada nao encontrado: {input_path}")

    n_rows = len(pd.read_csv(input_path))
    if n_rows < 1000:
        pytest.skip(
            f"Dataset com {n_rows} registros. Recomenda-se >=1000 para sweep operacional."
        )

    epochs_raw = os.environ.get("EPOCH_SWEEP_VALUES", "10,20,50")
    epochs = _parse_epochs(epochs_raw)
    output_base = tmp_path / "outputs_epoch_sweep"
    output_base.mkdir(parents=True, exist_ok=True)

    results = []
    for epoch in epochs:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.main",
                "--input",
                str(input_path),
                "--epochs",
                str(epoch),
                "--seed",
                "42",
                "--output-dir",
                str(output_base),
            ],
            cwd=".",
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, (
            f"Falha ao treinar com epoch={epoch}\n"
            f"STDOUT:\n{run.stdout[-2000:]}\n"
            f"STDERR:\n{run.stderr[-2000:]}"
        )

        run_dir = _latest_run_dir(output_base)
        temporal_metrics = run_dir / "metrics" / "temporal_metrics.csv"
        run_summary = run_dir / "metrics" / "run_summary.json"

        assert temporal_metrics.exists(), f"Nao encontrou {temporal_metrics}"
        assert run_summary.exists(), f"Nao encontrou {run_summary}"

        with run_summary.open(encoding="utf-8") as f:
            summary = json.load(f)

        score = _score_epoch(temporal_metrics)
        score["epoch"] = epoch
        score["run_dir"] = str(run_dir)
        score["n_alerts_p95"] = summary.get("results_summary", {}).get("n_alerts_p95")
        results.append(score)

    ranking = pd.DataFrame(results).sort_values("objective_score", ascending=True)
    best = ranking.iloc[0]

    summary_csv = output_base / "epoch_sweep_summary.csv"
    summary_json = output_base / "epoch_sweep_summary.json"
    ranking.to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": int(best["epoch"]),
                "objective_score": float(best["objective_score"]),
                "criterion": "abs(mean_pct_anomaly_eval_p95-5.0) + 0.25*std",
                "ranking": ranking.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    assert int(best["epoch"]) in epochs
    assert summary_csv.exists()
    assert summary_json.exists()
