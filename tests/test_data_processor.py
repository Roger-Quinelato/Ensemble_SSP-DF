"""
Testes para o módulo DataProcessor do projeto Ensemble_SSP-DF.

Valida funcionalidades de processamento de dados, incluindo limpeza, padronização, engenharia de features e tratamento de datas.
Garante que o DataProcessor está funcionando conforme esperado e que as principais funções retornam resultados corretos.
"""
import pytest
import pandas as pd
from src.data_processor import DataProcessor
import yaml

def test_load_and_standardize():
    with open('config_mapeamento.yaml', 'r') as f:
        config = yaml.safe_load(f)
    proc = DataProcessor(config)
    df = proc.load_and_standardize('data/input/amostra_ssp.csv')
    assert not df.empty
    assert 'placa' in df.columns
    assert 'timestamp' in df.columns

def test_feature_engineering():
    with open('config_mapeamento.yaml', 'r') as f:
        config = yaml.safe_load(f)
    proc = DataProcessor(config)
    df = proc.load_and_standardize('data/input/amostra_ssp.csv')
    df_feat = proc.feature_engineering(df)
    assert 'hora_sin' in df_feat.columns
    assert 'velocidade_ms' in df_feat.columns
