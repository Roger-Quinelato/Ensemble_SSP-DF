import pytest

from src.utils.config_schema import (
    validate_inference_config,
    validate_pipeline_config,
    validate_training_config,
)


def _base_config():
    return {
        "random_state": 42,
        "mapeamento_colunas": {
            "placa": "placa",
            "timestamp": "timestamp",
            "latitude": "latitude",
            "longitude": "longitude",
            "RA": "regiao_adm",
        },
        "parametros": {
            "split_ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "percentis_teste": [90, 95, 99],
            "isolation_forest": {"n_estimators": [100], "contamination": "auto"},
            "hbos": {"n_bins": [10], "contamination": 0.1},
            "temporal": {"arch_type": "gru", "window_size": 3, "epochs": 1},
        },
        "configuracoes_gerais": {"gap_segmentation_seconds": 1800},
    }


def test_validate_pipeline_config_accepts_valid_config():
    cfg = _base_config()
    out = validate_pipeline_config(cfg, context="teste")
    assert out["mapeamento_colunas"]["placa"] == "placa"
    assert out["parametros"]["split_ratios"]["validation"] == pytest.approx(0.2)
    assert out["parametros"]["percentis_teste"] == [90, 95, 99]


def test_validate_pipeline_config_rejects_missing_required_mapping():
    cfg = _base_config()
    del cfg["mapeamento_colunas"]["timestamp"]
    with pytest.raises(ValueError, match="faltam chaves em mapeamento_colunas"):
        validate_pipeline_config(cfg, context="teste")


def test_validate_pipeline_config_rejects_invalid_block_type():
    cfg = _base_config()
    cfg["parametros"] = "invalido"
    with pytest.raises(ValueError, match="'parametros' deve ser objeto"):
        validate_pipeline_config(cfg, context="teste")


def test_validate_pipeline_config_rejects_invalid_split_ratio():
    cfg = _base_config()
    cfg["parametros"]["split_ratios"] = {"train": 0.8, "validation": 0.2, "test": 0.2}
    with pytest.raises(ValueError, match="soma de split_ratios > 1"):
        validate_pipeline_config(cfg, context="teste")


def test_validate_pipeline_config_rejects_invalid_percentile():
    cfg = _base_config()
    cfg["parametros"]["percentis_teste"] = [90, 101]
    with pytest.raises(ValueError, match="percentile fora de \\[0, 100\\]"):
        validate_pipeline_config(cfg, context="teste")


def test_validate_pipeline_config_rejects_non_dict_root():
    with pytest.raises(ValueError, match="esperado objeto/dict na raiz"):
        validate_pipeline_config(["nao", "dict"], context="teste")


def test_validate_pipeline_config_supports_split_val_alias():
    cfg = _base_config()
    cfg["parametros"]["split_ratios"] = {"train": 0.7, "val": 0.2, "test": 0.1}
    out = validate_pipeline_config(cfg, context="teste")
    assert out["parametros"]["split_ratios"]["validation"] == pytest.approx(0.2)


def test_validate_pipeline_config_sets_default_percentiles_for_compatibility():
    cfg = _base_config()
    del cfg["parametros"]["percentis_teste"]
    out = validate_training_config(cfg)
    assert out["parametros"]["percentis_teste"] == [95]
    out_infer = validate_inference_config(cfg)
    assert out_infer["parametros"]["percentis_teste"] == [95]


def test_validate_pipeline_config_normalizes_temporal_strategy():
    cfg = _base_config()
    cfg["parametros"]["temporal"]["temporal_strategy"] = "INTER"
    out = validate_pipeline_config(cfg, context="teste")
    assert out["parametros"]["temporal"]["temporal_strategy"] == "inter"


def test_validate_pipeline_config_rejects_invalid_temporal_strategy():
    cfg = _base_config()
    cfg["parametros"]["temporal"]["temporal_strategy"] = "invalida"
    with pytest.raises(ValueError, match="temporal_strategy invalida"):
        validate_pipeline_config(cfg, context="teste")
