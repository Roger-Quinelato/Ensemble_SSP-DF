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
