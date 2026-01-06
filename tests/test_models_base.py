"""
Testes para o módulo de modelos base do projeto Ensemble_SSP-DF.

Valida a implementação dos modelos tradicionais, funções de treinamento, predição e integração com o pipeline.
Garante que os modelos base funcionam corretamente e retornam resultados esperados.
"""
import pytest
import numpy as np
from src.models.models_base import BaselineModels

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
    labels, scores = model.train_lof(k_neighbors=5)
    assert len(labels) == 100
    assert len(scores) == 100
