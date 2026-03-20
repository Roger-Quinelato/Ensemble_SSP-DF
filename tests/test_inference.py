def test_inference_smoke():
    """Inference deve completar sem erros com os modelos do treino."""
    import os
    import shutil

    import pytest

    from src.pipeline.inference import predict

    models_dir = "outputs/models_saved"
    if not os.path.exists(models_dir):
        pytest.skip("Modelos nao treinados - rode run_experiment() primeiro")

    result = predict(
        input_path="data/input/amostra_ssp.csv",
        models_dir=models_dir,
        output_dir="outputs_inference_test",
    )
    assert "ensemble_alert" in result.columns
    assert "ensemble_vote_pct" in result.columns
    assert os.path.exists("outputs_inference_test/inference_result.parquet")
    shutil.rmtree("outputs_inference_test", ignore_errors=True)
