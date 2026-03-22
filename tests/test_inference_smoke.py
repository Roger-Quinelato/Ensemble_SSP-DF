def test_inference_smoke(tmp_path):
    """Smoke test: inference.py deve completar sem erros com dados minimos."""
    import glob
    import os
    import subprocess
    import sys

    import pandas as pd
    import pytest

    manifests = sorted(glob.glob("outputs/*/models_saved/models_manifest.json"))
    if not manifests:
        pytest.skip(
            "Nenhum modelo treinado encontrado - execute run_experiment() primeiro"
        )

    models_dir = os.path.dirname(manifests[-1])

    input_path = "data/input/amostra_ssp.csv"
    if not os.path.exists(input_path):
        pytest.skip("Arquivo data/input/amostra_ssp.csv nao encontrado")

    df_input = pd.read_csv(input_path)
    if df_input.empty:
        pytest.skip("Arquivo de amostra vazio para inferencia")

    if len(df_input) < 100:
        repeats = (120 // len(df_input)) + 1
        df_expanded = pd.concat([df_input] * repeats, ignore_index=True).head(120)
        if "timestamp" in df_expanded.columns:
            df_expanded["timestamp"] = pd.date_range(
                "2024-01-01 00:00:00",
                periods=len(df_expanded),
                freq="5min",
            )
        input_path = str(tmp_path / "amostra_ssp_expanded.csv")
        df_expanded.to_csv(input_path, index=False)

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

    output_parquet = output_dir / "inference_result.parquet"
    assert output_parquet.exists(), "inference_result.parquet nao foi gerado"
