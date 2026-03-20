import hashlib

import pytest

from src.utils.artifact_utils import (
    sha256_file,
    verify_artifact,
    verify_artifact_strict,
)


def test_sha256_file_matches_hashlib(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"abc123")
    expected = hashlib.sha256(b"abc123").hexdigest()

    assert sha256_file(str(artifact)) == expected


def test_verify_artifact_detects_mismatch(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"abc123")
    wrong_hash = hashlib.sha256(b"different").hexdigest()

    assert verify_artifact(str(artifact), wrong_hash) is False


def test_verify_artifact_strict_raises_on_mismatch(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"abc123")
    wrong_hash = hashlib.sha256(b"different").hexdigest()

    with pytest.raises(ValueError, match="FALHA DE INTEGRIDADE"):
        verify_artifact_strict(str(artifact), wrong_hash, model_name="ISO_N100")
