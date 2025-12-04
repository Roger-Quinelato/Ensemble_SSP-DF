import pytest
import numpy as np
from src.models_base import BaselineModels

def test_train_iso():
    X = np.random.rand(100, 5)
    model = BaselineModels(X)
    labels, scores, iso_model = model.train_iso(n_estimators=10)
    assert len(labels) == 100
    assert len(scores) == 100
    assert hasattr(iso_model, 'fit')

def test_train_lof():
    X = np.random.rand(100, 5)
    model = BaselineModels(X)
    labels, scores = model.train_lof(k_neighbors=5, strategy='standard')
    assert len(labels) == 100
    assert len(scores) == 100
