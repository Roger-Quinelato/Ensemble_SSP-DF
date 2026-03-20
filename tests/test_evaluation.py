"""
Testes para o módulo de avaliação do projeto Ensemble_SSP-DF.

Valida funções de cálculo de métricas, avaliação de modelos e geração de resultados.
Garante que as métricas e avaliações estão corretas e robustas para diferentes cenários de entrada.
"""
import pytest
import numpy as np
import pandas as pd
from src.utils.evaluation import ThresholdOptimizer, ModelConcordanceAnalyzer

def test_apply_dynamic_thresholds():
    df = pd.DataFrame({'score': np.random.rand(100)})
    optimizer = ThresholdOptimizer([90, 95, 99])
    df, results = optimizer.apply_dynamic_thresholds(df, 'score', 'TestModel')
    assert 'TestModel_p90_label' in df.columns
    assert isinstance(results, list)

def test_analyze_concordance():
    df = pd.DataFrame(
        {
            "pred1": np.random.randint(0, 2, 100),
            "pred2": np.random.randint(0, 2, 100),
            "pred3": np.random.randint(0, 2, 100),
        }
    )
    metrics_df = ModelConcordanceAnalyzer.analyze_concordance(
        df, ["pred1", "pred2", "pred3"]
    )
    assert "Agreement" in metrics_df.columns
    assert "Jaccard_Anomalies" in metrics_df.columns


def test_nan_score_produces_nan_label():
    """Registros sem score NÃO devem receber label 0 (normal)."""
    df = pd.DataFrame({"model_score": [0.1, 0.9, np.nan, 0.5, np.nan]})
    optimizer = ThresholdOptimizer(percentiles=[50])
    df_result, _ = optimizer.apply_dynamic_thresholds(
        df, "model_score", "test_model", calibration_scores=np.array([0.1, 0.9, 0.5])
    )
    # NaN scores devem permanecer NaN no label
    assert df_result["test_model_p50_label"].isna().sum() == 2, (
        "Registros com score NaN devem ter label NaN, NÃO 0"
    )
    # Scores válidos devem ter label 0 ou 1
    assert df_result["test_model_p50_label"].dropna().isin([0.0, 1.0]).all(), (
        "Labels de registros avaliados devem ser 0.0 ou 1.0"
    )
