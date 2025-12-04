import pytest
import numpy as np
import pandas as pd
from src.evaluation import ThresholdOptimizer, GroundTruthComparator

def test_apply_dynamic_thresholds():
    df = pd.DataFrame({'score': np.random.rand(100)})
    optimizer = ThresholdOptimizer([90, 95, 99])
    df, results = optimizer.apply_dynamic_thresholds(df, 'score', 'TestModel')
    assert 'TestModel_p90_label' in df.columns
    assert isinstance(results, list)

def test_compare_all_models():
    df = pd.DataFrame({'gt': np.random.randint(0, 2, 100), 'pred1': np.random.randint(0, 2, 100), 'pred2': np.random.randint(0, 2, 100)})
    metrics_df = GroundTruthComparator.compare_all_models(df, 'gt', ['pred1', 'pred2'])
    assert 'Precision' in metrics_df.columns
