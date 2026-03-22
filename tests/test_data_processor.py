"""
Testes para o módulo DataProcessor do projeto Ensemble_SSP-DF.

Valida funcionalidades de processamento de dados, incluindo limpeza, padronização, engenharia de features e tratamento de datas.
Garante que o DataProcessor está funcionando conforme esperado e que as principais funções retornam resultados corretos.
"""
import pytest
import pandas as pd
from src.data.data_processor import DataProcessor
import yaml


def _ensure_min_records(input_path, tmp_path, min_records=100):
    """
    Garante dataset de teste com no minimo min_records linhas para satisfazer schema.
    """
    df = pd.read_csv(input_path)
    if df.empty:
        pytest.skip("Arquivo de amostra vazio para testes de DataProcessor")
    if len(df) >= min_records:
        return input_path

    repeats = (min_records // len(df)) + 1
    df_expanded = pd.concat([df] * repeats, ignore_index=True).head(min_records)
    if "timestamp" in df_expanded.columns:
        df_expanded["timestamp"] = pd.date_range(
            "2024-01-01 00:00:00",
            periods=len(df_expanded),
            freq="5min",
        )

    expanded_path = tmp_path / "amostra_ssp_min_records.csv"
    df_expanded.to_csv(expanded_path, index=False)
    return str(expanded_path)


def test_load_and_standardize(tmp_path):
    with open('config_mapeamento.yaml', 'r') as f:
        config = yaml.safe_load(f)
    proc = DataProcessor(config)
    input_path = _ensure_min_records("data/input/amostra_ssp.csv", tmp_path)
    df = proc.load_and_standardize(input_path)
    if hasattr(df, "compute"):
        df = df.compute()
    assert not df.empty
    assert 'placa' in df.columns
    assert 'timestamp' in df.columns

def test_feature_engineering(tmp_path):
    with open('config_mapeamento.yaml', 'r') as f:
        config = yaml.safe_load(f)
    proc = DataProcessor(config)
    input_path = _ensure_min_records("data/input/amostra_ssp.csv", tmp_path)
    df = proc.load_and_standardize(input_path)
    df_feat, features = proc.feature_engineering(df)
    assert 'hora_sin' in df_feat.columns
    assert 'velocidade_kmh' in df_feat.columns
    assert 'hora_sin' in features
