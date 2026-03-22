"""
Testes para validacao de schema de entrada.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.schema import validate_input


def _make_valid_df(n=50):
    """Cria DataFrame valido minimo para testes."""
    return pd.DataFrame(
        {
            "placa": [f"ABC{i:04d}" for i in range(n)],
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "latitude": np.random.uniform(-16.0, -15.5, n),
            "longitude": np.random.uniform(-48.0, -47.5, n),
        }
    )


def test_valid_data():
    df = _make_valid_df(200)
    result = validate_input(df)
    assert len(result) == 200


def test_missing_placa():
    df = _make_valid_df(200).drop(columns=["placa"])
    with pytest.raises(Exception):
        validate_input(df)


def test_invalid_latitude():
    df = _make_valid_df(200)
    df.loc[0, "latitude"] = -5.0  # Fora do DF
    with pytest.raises(Exception):
        validate_input(df)


def test_small_sample_is_allowed():
    """
    Schema atual permite micro-batches (>=1 registro).
    Isso garante que amostra_ssp.csv pequena funcione como padrao.
    """
    df = _make_valid_df(10)
    result = validate_input(df)
    assert len(result) == 10


def test_future_timestamp():
    df = _make_valid_df(200)
    df.loc[0, "timestamp"] = pd.Timestamp("2035-01-01")
    with pytest.raises(Exception):
        validate_input(df)
