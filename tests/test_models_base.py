"""
Testes para o módulo de modelos base do projeto Ensemble_SSP-DF.
"""
import numpy as np
from src.models.models_base import BaselineModels

def test_train_iso():
    X = np.random.rand(100, 5)
    model = BaselineModels(X)
    labels, scores, iso_model = model.train_iso(n_estimators=10)
    assert len(labels) == 100
    assert len(scores) == 100
    assert hasattr(iso_model, 'fit')

def test_train_hbos():
    X = np.random.rand(100, 5)
    model = BaselineModels(X)
    labels, scores, hbos_model = model.train_hbos(n_bins=10, contamination=0.1)
    assert len(labels) == 100
    assert len(scores) == 100
    assert hasattr(hbos_model, 'fit')
