"""
Testes de integracao treino -> inferencia.

Verificam que a cadeia completa de artefatos (thresholds, scalers, modelos)
e consistente entre o treinamento e a inferencia.

DEPENDENCIA: Requer NC1 (thresholds serializados) aplicado para funcionar.
"""
import json
import os
import warnings

import pytest


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def latest_run_dir():
    """Retorna o diretorio da run mais recente em outputs/."""
    outputs_base = "outputs"
    if not os.path.exists(outputs_base):
        pytest.skip("Pasta outputs/ nao encontrada - execute run_experiment() primeiro")

    runs = sorted(
        [
            f
            for f in os.listdir(outputs_base)
            if os.path.isdir(os.path.join(outputs_base, f))
            and f != "logs"
            and os.path.isdir(os.path.join(outputs_base, f, "models_saved"))
        ],
        reverse=True,
    )

    if not runs:
        pytest.skip("Nenhuma run encontrada em outputs/")

    return os.path.join(outputs_base, runs[0])


@pytest.fixture(scope="module")
def models_dir(latest_run_dir):
    models_path = os.path.join(latest_run_dir, "models_saved")
    if not os.path.exists(models_path):
        pytest.skip(f"models_saved nao encontrado em {latest_run_dir}")
    return models_path


@pytest.fixture(scope="module")
def metrics_dir(latest_run_dir):
    return os.path.join(latest_run_dir, "metrics")


# =============================================================================
# TESTE 1: Thresholds serializados existem e sao validos
# =============================================================================


class TestThresholdSerialization:
    """Verifica que NC1 foi aplicado corretamente."""

    def test_threshold_files_exist(self, models_dir):
        """Arquivos thresholds_p{90,95,99}.json devem existir apos treinamento."""
        for p in [90, 95, 99]:
            path = os.path.join(models_dir, f"thresholds_p{p}.json")
            assert os.path.exists(path), (
                f"REGRESSAO NC1: thresholds_p{p}.json nao encontrado em {models_dir}.\n"
                "inference.py esta operando em MODO DEGRADADO (recalibrando em producao).\n"
                "Aplicar NC1 (serializacao de thresholds) antes de prosseguir."
            )

    def test_thresholds_are_numeric_and_positive(self, models_dir):
        """Todos os thresholds devem ser float positivos."""
        for p in [90, 95, 99]:
            path = os.path.join(models_dir, f"thresholds_p{p}.json")
            if not os.path.exists(path):
                pytest.skip(f"thresholds_p{p}.json nao encontrado")
            with open(path, encoding="utf-8") as f:
                thresholds = json.load(f)
            assert len(thresholds) > 0, f"thresholds_p{p}.json esta vazio"
            for model_name, value in thresholds.items():
                assert isinstance(value, (int, float)), (
                    f"Threshold de {model_name} (p{p}) nao e numerico: {value}"
                )
                assert value >= 0, (
                    f"Threshold de {model_name} (p{p}) e negativo: {value}\n"
                    "Thresholds de anomalia devem ser nao-negativos."
                )

    def test_threshold_p95_covers_expected_models(self, models_dir):
        """thresholds_p95.json deve cobrir ISO, HBOS e Temporal."""
        path = os.path.join(models_dir, "thresholds_p95.json")
        if not os.path.exists(path):
            pytest.skip("thresholds_p95.json nao encontrado")
        with open(path, encoding="utf-8") as f:
            thresholds = json.load(f)

        model_names = list(thresholds.keys())
        iso_models = [m for m in model_names if m.startswith("ISO")]
        hbos_models = [m for m in model_names if m.startswith("HBOS")]
        temp_models = [m for m in model_names if m.startswith("Temporal")]

        assert len(iso_models) > 0, (
            f"Nenhum modelo ISO em thresholds_p95.json. Modelos: {model_names}"
        )
        assert len(hbos_models) > 0, (
            f"Nenhum modelo HBOS em thresholds_p95.json. Modelos: {model_names}"
        )
        # Temporal e opcional (pode nao ter sequencias suficientes em dados minimos)
        if len(temp_models) == 0:
            warnings.warn(
                "Nenhum modelo Temporal em thresholds_p95.json - "
                "normal se dados de treino tem menos de window_size registros por veiculo.",
                stacklevel=1,
            )


# =============================================================================
# TESTE 2: Thresholds de inferencia == thresholds de treinamento
# =============================================================================


class TestThresholdConsistency:
    """Verifica que inference.py usa os thresholds do treino, nao recalibra."""

    def test_inference_loads_training_thresholds(self, models_dir):
        """load_thresholds() deve retornar os mesmos valores do arquivo .json."""
        try:
            from src.pipeline.inference import load_thresholds
        except ImportError:
            pytest.skip("inference.py nao encontrado")

        thresh_path = os.path.join(models_dir, "thresholds_p95.json")
        if not os.path.exists(thresh_path):
            pytest.skip("thresholds_p95.json nao encontrado - aplicar NC1")

        with open(thresh_path, encoding="utf-8") as f:
            expected = json.load(f)

        loaded = load_thresholds(models_dir, percentile=95)

        assert loaded is not None, (
            "load_thresholds() retornou None mesmo com arquivo existente.\n"
            "Verificar implementacao de load_thresholds() em inference.py."
        )

        for model_name, expected_val in expected.items():
            assert model_name in loaded, (
                f"Modelo {model_name} presente no arquivo mas nao retornado por load_thresholds()"
            )
            assert abs(loaded[model_name] - expected_val) < 1e-10, (
                f"Threshold inconsistente para {model_name}:\n"
                f"  Arquivo: {expected_val}\n"
                f"  load_thresholds(): {loaded[model_name]}\n"
                "Os valores devem ser identicos - sem arredondamento."
            )

    def test_inference_does_not_warn_degraded_mode(self, models_dir, tmp_path, caplog):
        """inference.py NAO deve logar warning de 'modo degradado'."""
        try:
            from src.pipeline.inference import load_thresholds
        except ImportError:
            pytest.skip("inference.py nao encontrado")

        import logging

        with caplog.at_level(logging.WARNING, logger="sspdf"):
            load_thresholds(models_dir, percentile=95)

        degraded_warnings = [
            r
            for r in caplog.records
            if "modo degradado" in r.message.lower()
            or "recalculados" in r.message.lower()
        ]

        assert len(degraded_warnings) == 0, (
            "REGRESSAO NC1: inference.py logou warning de modo degradado:\n"
            + "\n".join(r.message for r in degraded_warnings)
            + "\nIsso significa que thresholds nao foram encontrados e "
            + "serao recalibrados nos dados de producao."
        )


# =============================================================================
# TESTE 3: Manifesto de modelos valido e completo
# =============================================================================


class TestManifestIntegrity:
    """Verifica que models_manifest.json e valido e tem hashes SHA256."""

    def test_manifest_exists(self, models_dir):
        path = os.path.join(models_dir, "models_manifest.json")
        assert os.path.exists(path), (
            f"models_manifest.json nao encontrado em {models_dir}.\n"
            "Verificar que export_results() foi executado corretamente."
        )

    def test_manifest_has_sha256(self, models_dir):
        """Cada entrada no manifesto deve ter campo sha256 (Fix HN3)."""
        path = os.path.join(models_dir, "models_manifest.json")
        if not os.path.exists(path):
            pytest.skip("models_manifest.json nao encontrado")

        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)

        entries_without_hash = []
        for section_name, section in manifest.items():
            if isinstance(section, list):
                for entry in section:
                    if isinstance(entry, dict) and "path" in entry:
                        if "sha256" not in entry:
                            entries_without_hash.append(
                                f"{section_name}/{entry.get('tag', '?')}"
                            )
            elif isinstance(section, dict) and "path" in section:
                if "sha256" not in section:
                    entries_without_hash.append(section_name)

        if entries_without_hash:
            pytest.xfail(
                f"Entradas sem SHA256 (Fix HN3 nao aplicado): {entries_without_hash}\n"
                "Aplicar HN3 para adicionar integridade de artefatos ao manifesto."
            )

    def test_manifest_model_files_exist(self, models_dir):
        """Todos os arquivos referenciados no manifesto devem existir."""
        path = os.path.join(models_dir, "models_manifest.json")
        if not os.path.exists(path):
            pytest.skip("models_manifest.json nao encontrado")

        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)

        missing_files = []
        for section_name, section in manifest.items():
            if section_name in ("thresholds", "run_id", "timestamp"):
                continue
            if isinstance(section, list):
                for entry in section:
                    if isinstance(entry, dict) and "path" in entry:
                        if not os.path.exists(entry["path"]):
                            missing_files.append(entry["path"])
            elif isinstance(section, dict) and "path" in section:
                if section.get("path") and not os.path.exists(section["path"]):
                    missing_files.append(section["path"])

        assert len(missing_files) == 0, (
            "Arquivos referenciados no manifesto nao encontrados:\n"
            + "\n".join(f"  {p}" for p in missing_files)
        )
