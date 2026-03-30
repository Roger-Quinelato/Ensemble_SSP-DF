"""
Testes para o módulo DataProcessor do projeto Ensemble_SSP-DF.

Valida funcionalidades de processamento de dados, incluindo limpeza, padronização, engenharia de features e tratamento de datas.
Garante que o DataProcessor está funcionando conforme esperado e que as principais funções retornam resultados corretos.
"""
import pytest
import pandas as pd
from src.data.data_processor import DataProcessor
import yaml
from pathlib import Path


def _ensure_min_records(input_path, tmp_path, min_records=100):
    """
    Garante dataset de teste com no minimo min_records linhas para satisfazer schema.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        pytest.skip(f"Arquivo de amostra nao encontrado: {input_path}")

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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.hermetic
def test_load_and_standardize_rejects_unsupported_extension(tmp_path):
    with open('config_mapeamento.yaml', 'r') as f:
        config = yaml.safe_load(f)
    proc = DataProcessor(config)

    bad_input = tmp_path / "entrada.txt"
    bad_input.write_text("conteudo invalido", encoding="utf-8")

    with pytest.raises(ValueError, match="Formato de input nao suportado"):
        proc.load_and_standardize(str(bad_input))


@pytest.mark.hermetic
def test_load_and_standardize_rejects_parent_ref_path():
    with open('config_mapeamento.yaml', 'r') as f:
        config = yaml.safe_load(f)
    proc = DataProcessor(config)

    with pytest.raises(ValueError, match="path traversal relativo"):
        proc.load_and_standardize("..\\fora\\dados.csv")


@pytest.mark.hermetic
def test_load_and_standardize_csv_parquet_equivalence(tmp_path):
    config = {
        "mapeamento_colunas": {
            "placa": "placa",
            "timestamp": "timestamp",
            "latitude": "latitude",
            "longitude": "longitude",
            "RA": "regiao_adm",
        }
    }

    df_source = pd.DataFrame(
        {
            "placa": ["ABC1234", "DEF5678", "GHI9012"],
            "timestamp": pd.to_datetime(
                ["2024-01-01 08:00:00", "2024-01-01 08:05:00", "2024-01-01 08:10:00"]
            ),
            "latitude": [-15.80, -15.81, -15.82],
            "longitude": [-47.90, -47.91, -47.92],
            "regiao_adm": ["Plano Piloto", "Ceilandia", "Taguatinga"],
        }
    )

    csv_path = tmp_path / "input.csv"
    parquet_path = tmp_path / "input.parquet"
    df_source.to_csv(csv_path, index=False)
    df_source.to_parquet(parquet_path, index=False)

    proc_csv = DataProcessor(config)
    proc_parquet = DataProcessor(config)
    df_csv = proc_csv.load_and_standardize(str(csv_path))
    df_parquet = proc_parquet.load_and_standardize(str(parquet_path))

    critical_cols = {"placa", "timestamp", "latitude", "longitude"}
    assert critical_cols.issubset(df_csv.columns)
    assert critical_cols.issubset(df_parquet.columns)
    assert len(df_csv) == len(df_parquet) == len(df_source)

    assert pd.api.types.is_datetime64_any_dtype(df_csv["timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(df_parquet["timestamp"])
    assert pd.api.types.is_float_dtype(df_csv["latitude"])
    assert pd.api.types.is_float_dtype(df_parquet["latitude"])
    assert pd.api.types.is_float_dtype(df_csv["longitude"])
    assert pd.api.types.is_float_dtype(df_parquet["longitude"])

    cols = ["placa", "timestamp", "latitude", "longitude"]
    pd.testing.assert_frame_equal(
        df_csv[cols].reset_index(drop=True),
        df_parquet[cols].reset_index(drop=True),
        check_dtype=False,
    )
