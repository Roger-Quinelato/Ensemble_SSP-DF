def test_inference_smoke(tmp_path):
    """Inference deve completar sem erros com os modelos do treino."""
    import os
    import re
    import shutil

    import pandas as pd
    import pytest

    from src.pipeline.inference import predict

    models_dir = None
    run_pattern = re.compile(r"^\d{8}_\d{6}$")
    if os.path.exists("outputs"):
        runs = sorted(
            [
                d
                for d in os.listdir("outputs")
                if os.path.isdir(os.path.join("outputs", d)) and run_pattern.match(d)
            ],
            reverse=True,
        )
        if runs:
            candidate = os.path.join("outputs", runs[0], "models_saved")
            if os.path.exists(candidate):
                models_dir = candidate
    if not models_dir:
        pytest.skip("Modelos versionados nao encontrados - rode run_experiment() primeiro")

    input_path = "data/input/amostra_ssp.csv"
    if os.path.exists(input_path):
        df_input = pd.read_csv(input_path)
        if df_input.empty:
            pytest.skip("Arquivo de amostra vazio para inferencia")
        if len(df_input) < 100:
            repeats = (120 // max(len(df_input), 1)) + 1
            df_expanded = pd.concat([df_input] * repeats, ignore_index=True).head(120)
            if "timestamp" in df_expanded.columns:
                df_expanded["timestamp"] = pd.date_range(
                    "2024-01-01 00:00:00",
                    periods=len(df_expanded),
                    freq="5min",
                )
            input_path = str(tmp_path / "amostra_ssp_expanded.csv")
            df_expanded.to_csv(input_path, index=False)

    result = predict(
        input_path=input_path,
        models_dir=models_dir,
        output_dir="outputs_inference_test",
    )
    assert "ensemble_alert" in result.columns
    assert "ensemble_vote_pct" in result.columns
    assert os.path.exists("outputs_inference_test/inference_result.parquet")
    shutil.rmtree("outputs_inference_test", ignore_errors=True)
