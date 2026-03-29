import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.inference import predict


def _prepare_input_for_inference(tmp_path):
    input_path = Path("data/input/amostra_ssp.csv")
    if not input_path.exists():
        pytest.skip("Arquivo data/input/amostra_ssp.csv nao encontrado")

    df_input = pd.read_csv(input_path)
    if df_input.empty:
        pytest.skip("Arquivo de amostra vazio para inferencia")

    if len(df_input) >= 100:
        return str(input_path)

    repeats = (120 // len(df_input)) + 1
    df_expanded = pd.concat([df_input] * repeats, ignore_index=True).head(120)
    if "timestamp" in df_expanded.columns:
        df_expanded["timestamp"] = pd.date_range(
            "2024-01-01 00:00:00",
            periods=len(df_expanded),
            freq="5min",
        )

    expanded_path = tmp_path / "amostra_ssp_expanded.csv"
    df_expanded.to_csv(expanded_path, index=False)
    return str(expanded_path)


def test_predict_smoke(models_dir, tmp_path):
    """Inference deve completar sem erros com os modelos do treino."""
    input_path = _prepare_input_for_inference(tmp_path)
    output_dir = tmp_path / "outputs_inference_test"

    result = predict(
        input_path=input_path,
        models_dir=models_dir,
        output_dir=str(output_dir),
    )

    assert "ensemble_alert" in result.columns
    assert "ensemble_vote_pct" in result.columns
    assert (output_dir / "inference_result.parquet").exists()


def test_inference_cli_smoke(models_dir, tmp_path):
    """Smoke test de CLI para python -m src.pipeline.inference."""
    input_path = _prepare_input_for_inference(tmp_path)
    output_dir = tmp_path / "inference_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.pipeline.inference",
            "--models-dir",
            models_dir,
            "--input",
            input_path,
            "--output-dir",
            str(output_dir),
            "--percentile",
            "95",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"inference.py falhou com codigo {result.returncode}\n"
        f"STDOUT: {result.stdout[-2000:]}\n"
        f"STDERR: {result.stderr[-2000:]}"
    )

    combined = (result.stdout + "\n" + result.stderr).lower()
    assert "modo degradado" not in combined, (
        "ALERTA: inference.py rodou em modo degradado - "
        "thresholds nao foram serializados (NC1 nao aplicado)"
    )
    assert "thresholds_p95.json nao encontrado" not in combined, (
        "ALERTA: thresholds de treino nao foram carregados"
    )
    assert (output_dir / "inference_result.parquet").exists(), (
        "inference_result.parquet nao foi gerado"
    )
