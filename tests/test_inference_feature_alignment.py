import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.utils.artifact_utils import sha256_file
from src.pipeline.inference import (
    _align_df_with_expected_ra,
    _build_feature_matrix,
    _ensure_expected_features_present,
    _load_feature_schema,
    _load_models_from_manifest,
    _normalize_and_validate_inference_paths,
    _resolve_expected_main_features,
    _resolve_feature_list,
    load_thresholds,
)
from src.utils.path_security import normalize_cli_path


def test_align_df_with_expected_ra_adds_missing_and_drops_unseen():
    df = pd.DataFrame(
        {
            "hora_sin": [0.1, 0.2],
            "hora_cos": [0.9, 0.8],
            "RA_Plano": [1, 0],
            "RA_Nova": [0, 1],
        }
    )
    expected = ["hora_sin", "hora_cos", "RA_Plano", "RA_Ceilandia"]

    aligned = _align_df_with_expected_ra(df, expected, "teste")

    assert "RA_Ceilandia" in aligned.columns
    assert aligned["RA_Ceilandia"].tolist() == [0, 0]
    assert "RA_Nova" not in aligned.columns

    # Nao muta DataFrame original.
    assert "RA_Nova" in df.columns


def test_align_df_with_expected_ra_raises_for_missing_non_ra():
    df = pd.DataFrame({"hora_sin": [0.1], "RA_Plano": [1]})
    expected = ["hora_sin", "hora_cos", "RA_Plano"]

    with pytest.raises(ValueError, match="Features obrigatorias ausentes"):
        _align_df_with_expected_ra(df, expected, "teste")


def test_build_feature_matrix_respects_expected_order():
    df = pd.DataFrame(
        {
            "f_a": [1.0, 2.0],
            "f_b": [3.0, 4.0],
            "f_c": [5.0, 6.0],
        }
    )

    matrix = _build_feature_matrix(df, ["f_c", "f_a"], "matriz")
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix[0], np.array([5.0, 1.0]))


def test_resolve_expected_main_features_falls_back_to_scaler_feature_names():
    scaler = StandardScaler()
    train_df = pd.DataFrame(
        {
            "hora_sin": [0.0, 0.1, 0.2],
            "hora_cos": [1.0, 0.9, 0.8],
            "RA_Plano": [1, 0, 1],
        }
    )
    scaler.fit(train_df)

    resolved = _resolve_expected_main_features({}, scaler)
    assert resolved == ["hora_sin", "hora_cos", "RA_Plano"]


def test_resolve_feature_list_prefers_schema_over_fallback():
    schema = {"iso_features": ["f2", "f1"]}
    fallback = ["f1", "f2", "f3"]

    resolved = _resolve_feature_list(schema, "iso_features", fallback)
    assert resolved == ["f2", "f1"]


def test_resolve_feature_list_warns_explicitly_on_legacy_fallback(caplog):
    fallback = ["f1", "f2"]

    caplog.set_level("WARNING")
    resolved = _resolve_feature_list(
        {},
        "iso_features",
        fallback,
        context_label="ISO",
        warn_on_fallback=True,
    )

    assert resolved == fallback
    assert "ISO: usando fallback legado de features por ausencia de feature_schema.json." in caplog.text


def test_ensure_expected_features_present_raises_when_missing():
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})

    with pytest.raises(ValueError, match="Contrato treino->inferencia violado"):
        _ensure_expected_features_present(
            df,
            ["f1", "f2", "f3"],
            context_label="transform_scaler",
        )


def test_ensure_expected_features_present_accepts_complete_schema():
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})

    _ensure_expected_features_present(
        df,
        ["f1", "f2"],
        context_label="transform_scaler",
    )


def test_load_thresholds_requires_hash_in_strict_mode(tmp_path):
    thresholds_path = tmp_path / "thresholds_p95.json"
    thresholds_path.write_text('{"ISO_N100": 0.123}', encoding="utf-8")

    manifest = {
        "thresholds": {
            "95": {
                "path": "thresholds_p95.json",
            }
        }
    }

    with pytest.raises(ValueError, match="sha256 ausente"):
        load_thresholds(
            str(tmp_path),
            percentile=95,
            manifest=manifest,
            strict_integrity=True,
        )


def test_load_thresholds_keeps_degraded_mode_when_missing_only_in_permissive(tmp_path):
    loaded = load_thresholds(
        str(tmp_path),
        percentile=95,
        manifest=None,
        strict_integrity=False,
        require_hash=False,
    )
    assert loaded is None


def test_load_models_manifest_requires_hash_for_legacy_scaler_paths(tmp_path):
    scaler_path = tmp_path / "scaler.joblib"
    scaler_path.write_bytes(b"dummy")

    manifest = {
        "iso": [],
        "hbos": [],
        "temporal": [],
        "scalers": {"main": str(scaler_path)},
    }

    with pytest.raises(ValueError, match="scaler.joblib"):
        _load_models_from_manifest(
            manifest,
            strict_integrity=True,
            require_hash=True,
        )


def test_load_models_manifest_accepts_modern_scaler_dict_with_sha256(tmp_path):
    scaler = StandardScaler()
    train_df = pd.DataFrame({"f1": [0.0, 1.0], "f2": [2.0, 3.0]})
    scaler.fit(train_df)

    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    scaler_hash = sha256_file(str(scaler_path))

    manifest = {
        "iso": [],
        "hbos": [],
        "temporal": [],
        "scaler": {"path": str(scaler_path), "sha256": scaler_hash},
    }

    _, _, _, loaded_scaler, loaded_gru_scaler = _load_models_from_manifest(
        manifest,
        strict_integrity=True,
        require_hash=True,
    )

    assert loaded_gru_scaler is None
    assert loaded_scaler is not None
    assert list(loaded_scaler.feature_names_in_) == ["f1", "f2"]


def test_load_thresholds_strict_requires_percentile_entry(tmp_path):
    thresholds_path = tmp_path / "thresholds_p95.json"
    thresholds_path.write_text('{"ISO_N100": 0.5}', encoding="utf-8")
    manifest = {"thresholds": {"90": {"path": "thresholds_p95.json", "sha256": "abc"}}}

    with pytest.raises(ValueError, match="percentil 95 ausente"):
        load_thresholds(
            str(tmp_path),
            percentile=95,
            manifest=manifest,
            strict_integrity=True,
        )


def test_load_feature_schema_strict_requires_sha256(tmp_path):
    schema_path = tmp_path / "feature_schema.json"
    schema_path.write_text(
        json.dumps({"main_scaler_features": ["f1", "f2"]}),
        encoding="utf-8",
    )
    manifest = {"feature_schema": {"path": "feature_schema.json"}}

    with pytest.raises(ValueError, match="campo sha256 ausente"):
        _load_feature_schema(
            str(tmp_path),
            manifest=manifest,
            strict_integrity=True,
        )


def test_load_feature_schema_permissive_warns_without_sha256(tmp_path, caplog):
    schema_payload = {"main_scaler_features": ["f1", "f2"]}
    schema_path = tmp_path / "feature_schema.json"
    schema_path.write_text(json.dumps(schema_payload), encoding="utf-8")

    caplog.set_level("WARNING")
    loaded = _load_feature_schema(
        str(tmp_path),
        manifest={},
        strict_integrity=False,
    )

    assert loaded == schema_payload
    assert "Integridade nao verificavel para feature_schema.json em modo permissivo" in caplog.text


def test_load_thresholds_permissive_warns_and_degrades(tmp_path, caplog):
    caplog.set_level("WARNING")
    manifest = {"thresholds": {}}

    loaded = load_thresholds(
        str(tmp_path),
        percentile=95,
        manifest=manifest,
        strict_integrity=False,
    )

    assert loaded is None
    assert "Manifesto sem entrada para thresholds p95" in caplog.text


def test_load_models_manifest_legacy_scaler_warns_in_permissive_mode(tmp_path, caplog):
    scaler = StandardScaler()
    train_df = pd.DataFrame({"f1": [0.0, 1.0], "f2": [2.0, 3.0]})
    scaler.fit(train_df)

    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    manifest = {
        "iso": [],
        "hbos": [],
        "temporal": [],
        "scalers": {"main": str(scaler_path)},
    }

    caplog.set_level("WARNING")
    _, _, _, loaded_scaler, _ = _load_models_from_manifest(
        manifest,
        strict_integrity=False,
        require_hash=False,
    )

    assert loaded_scaler is not None
    assert "Modo permissivo: carregando scaler.joblib via caminho legado sem sha256." in caplog.text


def test_normalize_cli_path_blocks_relative_parent_traversal():
    with pytest.raises(ValueError, match="path traversal relativo"):
        normalize_cli_path(
            "..\\segredo\\dados.csv",
            "--input",
            block_relative_parent=True,
        )


def test_normalize_cli_path_returns_absolute_normalized_path(tmp_path):
    fpath = tmp_path / "arquivo.csv"
    fpath.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = normalize_cli_path(
        str(fpath),
        "--input",
        must_exist=True,
        expect_dir=False,
    )

    assert os.path.isabs(resolved)
    assert resolved.endswith("arquivo.csv")


def test_inference_path_validation_rejects_models_dir_parent_ref(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("random_state: 42\nmapeamento_colunas: {}\n", encoding="utf-8")
    inp = tmp_path / "in.csv"
    inp.write_text("placa,timestamp,latitude,longitude\nA,2024-01-01,-15.8,-47.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--models-dir"):
        _normalize_and_validate_inference_paths(
            input_path=str(inp),
            models_dir="..\\models_saved",
            config_path=str(cfg),
            output_dir=str(tmp_path / "out"),
        )
