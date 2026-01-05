"""
Testes para o módulo de modelos deep learning do projeto Ensemble_SSP-DF.

Valida a implementação dos modelos de aprendizado profundo, funções de treinamento, predição e integração com o pipeline.
Garante que os modelos deep learning funcionam corretamente e retornam resultados esperados.
"""
import pytest
import numpy as np
from src.models_deep import LSTMPipeline

def test_create_sequences_with_index():
    X = np.random.rand(20, 3)
    vehicle_ids = ['A']*10 + ['B']*10
    timestamps = np.arange(20)
    original_indices = list(range(20))
    pipeline = LSTMPipeline(X, vehicle_ids, timestamps, original_indices, window_size=3, max_gap_seconds=100)
    X_seq, idx = pipeline.create_sequences_with_index()
    assert X_seq.shape[1:] == (3, 3)
    assert len(idx) == X_seq.shape[0]

def test_train_evaluate():
    X = np.random.rand(20, 3)
    vehicle_ids = ['A']*10 + ['B']*10
    timestamps = np.arange(20)
    original_indices = list(range(20))
    pipeline = LSTMPipeline(X, vehicle_ids, timestamps, original_indices, window_size=3, max_gap_seconds=100)
    mse, idx, model = pipeline.train_evaluate('test', epochs=1)
    assert mse is not None
    assert idx is not None
