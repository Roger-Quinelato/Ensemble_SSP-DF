import pytest

from config.feature_config import (
    get_feature_importance_order,
    get_features_for_model,
)


def test_get_features_for_model_isolation_includes_ra_columns():
    available = [
        "hora_sin",
        "hora_cos",
        "dia_sem",
        "eh_feriado",
        "velocidade_kmh",
        "aceleracao",
        "dist_m",
        "RA_Plano",
        "RA_Ceilandia",
    ]
    feats = get_features_for_model("isolation_forest", available)
    assert "RA_Plano" in feats
    assert "RA_Ceilandia" in feats


def test_get_features_for_model_hbos_excludes_ra_columns():
    available = [
        "velocidade_kmh",
        "hora_sin",
        "hora_cos",
        "dia_sem",
        "eh_feriado",
        "RA_Plano",
    ]
    feats = get_features_for_model("hbos", available)
    assert "RA_Plano" not in feats
    assert feats == ["velocidade_kmh", "hora_sin", "hora_cos", "dia_sem", "eh_feriado"]


def test_get_features_for_model_unknown_raises():
    with pytest.raises(ValueError, match="configurado"):
        get_features_for_model("modelo_inexistente", [])


def test_get_feature_importance_order_returns_empty_for_unknown():
    assert get_feature_importance_order("unknown_model") == []
