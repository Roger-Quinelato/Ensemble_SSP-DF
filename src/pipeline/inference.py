"""
Modo de inferencia do pipeline SSP-DF.

Carrega modelos ja treinados e classifica novos dados sem re-treinar.

Uso:
    python -m src.pipeline.inference \
        --models-dir outputs/<run_id>/models_saved \
        --config config_mapeamento.yaml \
        --input novos_dados.csv \
        --output outputs_inference/ \
        --tf-device auto
"""

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from config.feature_config import get_features_for_model
from src.data.data_processor import DataProcessor
from src.utils.ensemble_decision import (
    compute_ensemble_decision,
    compute_vehicle_risk_summary,
)
from src.utils.evaluation import ThresholdOptimizer
from src.utils.artifact_utils import verify_artifact_strict
from src.utils.tf_runtime import configure_tensorflow_runtime
from src.utils.logger_utils import resolve_os_user
from src.utils.path_security import normalize_cli_path

logger = logging.getLogger("sspdf")
FEATURE_SCHEMA_FILENAME = "feature_schema.json"


def _normalize_and_validate_inference_paths(
    input_path,
    models_dir,
    config_path,
    output_dir,
):
    """
    Hardening de paths da CLI de inferencia.
    """
    models_dir = normalize_cli_path(
        models_dir,
        "--models-dir",
        must_exist=True,
        expect_dir=True,
        block_relative_parent=True,
    )
    config_path = normalize_cli_path(
        config_path,
        "--config",
        must_exist=True,
        expect_dir=False,
    )
    input_path = normalize_cli_path(
        input_path,
        "--input",
        must_exist=True,
        expect_dir=False,
    )
    output_dir = normalize_cli_path(
        output_dir,
        "--output-dir",
        must_exist=False,
        expect_dir=True,
        block_relative_parent=True,
    )
    return input_path, models_dir, config_path, output_dir


def _canonical_temporal_name_from_file(fname):
    """
    Converte temporal_*.h5 para nome canonico usado no treinamento.
    Ex.: temporal_union_ISO_n100_HBOS_bins10.h5 -> Temporal_Union_ISO_n100_HBOS_bins10
    """
    stem = fname.replace(".h5", "")
    payload = stem.replace("temporal_", "", 1)
    parts = payload.split("_")
    if parts:
        parts[0] = parts[0].capitalize()
    return "Temporal_" + "_".join(parts)


def _resolve_path(models_dir, model_path):
    if os.path.isabs(model_path):
        return model_path
    if os.path.exists(model_path):
        return model_path
    return os.path.join(models_dir, model_path)


def load_models_manifest(models_dir):
    """
    Carrega o manifesto de modelos treinados.
    Retorna dict com paths dos modelos disponiveis.
    """
    manifest_path = os.path.join(models_dir, "models_manifest.json")
    if not os.path.exists(manifest_path):
        # Fallback: descobrir modelos pelo nome de arquivo
        logger.warning(
            "models_manifest.json nao encontrado. "
            "Descobrindo modelos pelo nome do arquivo (modo compatibilidade)."
        )
        manifest = {
            "iso": [],
            "hbos": [],
            "temporal": [],
            "scalers": {},
            "_integrity_available": False,
        }
        for fname in os.listdir(models_dir):
            fpath = os.path.join(models_dir, fname)
            if fname.startswith("iso_") and fname.endswith(".joblib"):
                manifest["iso"].append(
                    {
                        "tag": fname.replace(".joblib", "").upper(),
                        "name": fname.replace(".joblib", "").upper(),
                        "path": fpath,
                    }
                )
            elif fname.startswith("hbos_") and fname.endswith(".joblib"):
                manifest["hbos"].append(
                    {
                        "tag": fname.replace(".joblib", "").upper(),
                        "name": fname.replace(".joblib", "").upper(),
                        "path": fpath,
                    }
                )
            elif fname.startswith("temporal_") and fname.endswith(".h5"):
                name = _canonical_temporal_name_from_file(fname)
                manifest["temporal"].append({"tag": name, "name": name, "path": fpath})
            elif fname == "scaler.joblib":
                manifest["scalers"]["main"] = fpath
            elif fname == "gru_scaler.joblib":
                manifest["scalers"]["gru"] = fpath
        return manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Resolver paths relativos para robustez.
    for section in ("iso", "hbos", "temporal"):
        for item in manifest.get(section, []):
            if "path" in item:
                item["path"] = _resolve_path(models_dir, item["path"])
            if "tag" not in item and "name" in item:
                item["tag"] = item["name"]
            if section == "temporal" and "name" in item:
                parts = str(item["name"]).split("_")
                if len(parts) >= 2 and parts[0] == "Temporal":
                    parts[1] = parts[1].capitalize()
                    item["name"] = "_".join(parts)
                    item["tag"] = item["name"]

    # Compatibilidade: manifesto pode ter scalers em bloco legado ou em chaves dedicadas.
    for key, value in manifest.get("scalers", {}).items():
        manifest["scalers"][key] = _resolve_path(models_dir, value)
    for scaler_key in ("scaler", "gru_scaler"):
        scaler_info = manifest.get(scaler_key)
        if isinstance(scaler_info, dict) and "path" in scaler_info:
            scaler_info["path"] = _resolve_path(models_dir, scaler_info["path"])

    # Marcar disponibilidade de integridade (sha256 em pelo menos um artefato).
    integrity_fields = []
    for section in ("iso", "hbos", "temporal"):
        integrity_fields.extend([entry.get("sha256") for entry in manifest.get(section, [])])
    for scaler_key in ("scaler", "gru_scaler"):
        scaler_info = manifest.get(scaler_key, {})
        if isinstance(scaler_info, dict):
            integrity_fields.append(scaler_info.get("sha256"))
    for threshold_info in manifest.get("thresholds", {}).values():
        if isinstance(threshold_info, dict):
            integrity_fields.append(threshold_info.get("sha256"))
    feature_schema_info = manifest.get("feature_schema", {})
    if isinstance(feature_schema_info, dict):
        integrity_fields.append(feature_schema_info.get("sha256"))
    manifest["_integrity_available"] = any(integrity_fields)
    return manifest


def _load_feature_schema(models_dir, manifest=None, strict_integrity=True):
    """
    Carrega schema de features do treino quando disponivel.
    """
    schema_info = None
    schema_path = os.path.join(models_dir, FEATURE_SCHEMA_FILENAME)

    if isinstance(manifest, dict):
        raw_schema_info = manifest.get("feature_schema")
        if isinstance(raw_schema_info, dict):
            schema_info = raw_schema_info
            if schema_info.get("path"):
                schema_path = _resolve_path(models_dir, schema_info["path"])
        elif raw_schema_info is not None:
            if strict_integrity:
                raise ValueError(
                    "Manifesto invalido para feature_schema.json: entrada nao e objeto."
                )
            logger.warning(
                "Manifesto invalido para feature_schema.json: entrada nao e objeto. "
                "Usando fallback permissivo para schema local."
            )

    if not os.path.exists(schema_path):
        return {}

    expected_hash = None
    if isinstance(schema_info, dict):
        expected_hash = schema_info.get("sha256")

    if strict_integrity:
        if not isinstance(schema_info, dict):
            raise ValueError(
                "Manifesto invalido para feature_schema.json: metadados ausentes em modo estrito."
            )
        if not expected_hash:
            raise ValueError(
                "Manifesto invalido para feature_schema.json: campo sha256 ausente."
            )
        verify_artifact_strict(schema_path, expected_hash, FEATURE_SCHEMA_FILENAME)
    else:
        if expected_hash:
            verify_artifact_strict(schema_path, expected_hash, FEATURE_SCHEMA_FILENAME)
        else:
            logger.warning(
                "Integridade nao verificavel para feature_schema.json em modo permissivo: "
                "campo sha256 ausente no manifesto."
            )

    with open(schema_path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        if strict_integrity:
            raise ValueError("feature_schema.json invalido (nao e objeto).")
        logger.warning("feature_schema.json invalido (nao e objeto). Ignorando.")
        return {}
    return payload


def _resolve_expected_main_features(feature_schema, scaler):
    """
    Resolve ordem de features do scaler principal.
    Prioridade: feature_schema.json > scaler.feature_names_in_.
    """
    schema_features = feature_schema.get("main_scaler_features", [])
    if isinstance(schema_features, list) and schema_features:
        return list(schema_features)

    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and len(scaler_features) > 0:
        return list(scaler_features)

    return []


def _resolve_feature_list(
    feature_schema,
    key,
    fallback,
    context_label=None,
    warn_on_fallback=False,
):
    expected = feature_schema.get(key, [])
    if isinstance(expected, list) and expected:
        return list(expected)
    resolved = list(fallback)
    if warn_on_fallback:
        label = context_label or key
        logger.warning(
            f"{label}: usando fallback legado de features por ausencia de feature_schema.json."
        )
    return resolved


def _align_df_with_expected_ra(df, expected_features, context_label):
    """
    Garante alinhamento de colunas RA_* com o schema do treino.
    """
    aligned = df.copy()
    expected_set = set(expected_features)

    missing = [col for col in expected_features if col not in aligned.columns]
    missing_ra = [col for col in missing if col.startswith("RA_")]
    missing_non_ra = [col for col in missing if not col.startswith("RA_")]

    for col in missing_ra:
        aligned[col] = 0

    if missing_non_ra:
        raise ValueError(
            f"Features obrigatorias ausentes para {context_label}: {missing_non_ra}. "
            "Nao e seguro inferir sem essas colunas."
        )

    extra_ra = [
        col
        for col in aligned.columns
        if col.startswith("RA_") and col not in expected_set
    ]
    if extra_ra:
        aligned = aligned.drop(columns=extra_ra, errors="ignore")
        logger.warning(
            f"{context_label}: ignorando {len(extra_ra)} RA_* nao vistas no treino: {extra_ra}"
        )

    if missing_ra:
        logger.info(
            f"{context_label}: adicionadas {len(missing_ra)} RA_* ausentes com 0: {missing_ra}"
        )
    return aligned


def _build_feature_matrix(df, features, context_label):
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(
            f"Features ausentes para {context_label}: {missing}. "
            "Verifique aderencia entre treino e inferencia."
        )
    return df[features].values


def _ensure_expected_features_present(df, expected_features, context_label):
    """
    Garante que o DataFrame manteve todas as colunas esperadas apos transformacoes.
    """
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(
            f"Features esperadas ausentes apos {context_label}: {missing}. "
            "Contrato treino->inferencia violado."
        )


def _ensure_artifact_integrity(
    entry, model_name, strict_integrity=True, require_hash=False
):
    """
    Verifica SHA256 quando disponivel no manifesto.
    """
    model_path = entry.get("path")
    expected_hash = entry.get("sha256")

    if not model_path:
        raise ValueError(f"Entrada de manifesto sem path para {model_name}")

    if expected_hash:
        verify_artifact_strict(model_path, expected_hash, model_name)
        return

    if require_hash:
        raise ValueError(
            f"Manifesto invalido para {model_name}: campo sha256 ausente."
        )

    if strict_integrity:
        logger.warning(
            f"Integridade nao verificavel para {model_name}: campo sha256 ausente no manifesto."
        )
    else:
        logger.warning(
            f"Modo permissivo: carregando {model_name} sem verificacao de sha256."
        )


def _load_models_from_manifest(
    manifest, strict_integrity=True, require_hash=False
):
    """
    Carrega modelos listados no manifesto, verificando integridade SHA256.

    Args:
        manifest: Dict do models_manifest.json.
        strict_integrity: Se True, valida hash quando disponivel.
    Returns:
        tuple: (iso_models, hbos_models, temporal_entries, scaler, gru_scaler)
    """
    iso_models = {}
    hbos_models = {}
    temporal_entries = []
    scaler = None
    gru_scaler = None

    for entry in manifest.get("iso", []):
        tag = entry.get("tag") or entry.get("name")
        _ensure_artifact_integrity(
            entry,
            tag,
            strict_integrity=strict_integrity,
            require_hash=require_hash,
        )
        iso_models[tag] = joblib.load(entry["path"])

    for entry in manifest.get("hbos", []):
        tag = entry.get("tag") or entry.get("name")
        _ensure_artifact_integrity(
            entry,
            tag,
            strict_integrity=strict_integrity,
            require_hash=require_hash,
        )
        hbos_models[tag] = joblib.load(entry["path"])

    for entry in manifest.get("temporal", []):
        tag = entry.get("tag") or entry.get("name")
        _ensure_artifact_integrity(
            entry,
            tag,
            strict_integrity=strict_integrity,
            require_hash=require_hash,
        )
        temporal_entries.append({"tag": tag, "path": entry["path"]})

    scaler_info = manifest.get("scaler")
    if isinstance(scaler_info, dict) and scaler_info.get("path"):
        _ensure_artifact_integrity(
            scaler_info,
            "scaler.joblib",
            strict_integrity=strict_integrity,
            require_hash=require_hash,
        )
        scaler = joblib.load(scaler_info["path"])
    else:
        scaler_path = manifest.get("scalers", {}).get("main")
        if scaler_path and os.path.exists(scaler_path):
            if require_hash:
                raise ValueError(
                    "Manifesto invalido para scaler.joblib: campo sha256 ausente."
                )
            if strict_integrity:
                logger.warning(
                    "Integridade nao verificavel para scaler.joblib em manifesto legado."
                )
            else:
                logger.warning(
                    "Modo permissivo: carregando scaler.joblib via caminho legado sem sha256."
                )
            scaler = joblib.load(scaler_path)

    gru_scaler_info = manifest.get("gru_scaler")
    if isinstance(gru_scaler_info, dict) and gru_scaler_info.get("path"):
        _ensure_artifact_integrity(
            gru_scaler_info,
            "gru_scaler.joblib",
            strict_integrity=strict_integrity,
            require_hash=require_hash,
        )
        gru_scaler = joblib.load(gru_scaler_info["path"])
    else:
        gru_scaler_path = manifest.get("scalers", {}).get("gru")
        if gru_scaler_path and os.path.exists(gru_scaler_path):
            if require_hash:
                raise ValueError(
                    "Manifesto invalido para gru_scaler.joblib: campo sha256 ausente."
                )
            if strict_integrity:
                logger.warning(
                    "Integridade nao verificavel para gru_scaler.joblib em manifesto legado."
                )
            else:
                logger.warning(
                    "Modo permissivo: carregando gru_scaler.joblib via caminho legado sem sha256."
                )
            gru_scaler = joblib.load(gru_scaler_path)

    return iso_models, hbos_models, temporal_entries, scaler, gru_scaler


def load_thresholds(
    models_dir,
    percentile=95,
    manifest=None,
    strict_integrity=True,
    require_hash=False,
):
    """
    Carrega thresholds de producao salvos pelo treinamento.
    Se nao existir arquivo de thresholds, usa os percentis padrao dos scores.
    """
    manifest_thresh = None
    effective_require_hash = bool(strict_integrity or require_hash)
    thresh_path = os.path.join(models_dir, f"thresholds_p{percentile}.json")
    thresholds_map = manifest.get("thresholds") if isinstance(manifest, dict) else None
    if strict_integrity:
        if not isinstance(thresholds_map, dict):
            raise ValueError(
                f"Manifesto invalido para thresholds_p{percentile}.json: bloco thresholds ausente."
            )
        manifest_thresh = thresholds_map.get(str(percentile))
        if not isinstance(manifest_thresh, dict):
            raise ValueError(
                f"Manifesto invalido: percentil {percentile} ausente em thresholds."
            )
        if not manifest_thresh.get("path"):
            raise ValueError(
                f"Manifesto invalido para thresholds_p{percentile}.json: campo path ausente."
            )
        if not manifest_thresh.get("sha256"):
            raise ValueError(
                f"Manifesto invalido para thresholds_p{percentile}.json: campo sha256 ausente."
            )
        thresh_path = _resolve_path(models_dir, manifest_thresh["path"])
    else:
        if isinstance(thresholds_map, dict):
            manifest_thresh = thresholds_map.get(str(percentile))
            if isinstance(manifest_thresh, dict):
                if manifest_thresh.get("path"):
                    thresh_path = _resolve_path(models_dir, manifest_thresh["path"])
                else:
                    logger.warning(
                        f"Manifesto sem path para thresholds_p{percentile}.json em modo permissivo."
                    )
            elif manifest_thresh is not None:
                logger.warning(
                    f"Manifesto invalido para thresholds_p{percentile}.json: entrada nao e objeto."
                )
            else:
                logger.warning(
                    f"Manifesto sem entrada para thresholds p{percentile}; tentando arquivo padrao (modo permissivo)."
                )
        elif manifest is not None:
            logger.warning(
                f"Manifesto sem bloco thresholds; tentando thresholds_p{percentile}.json local (modo permissivo)."
            )

    if os.path.exists(thresh_path):
        with open(thresh_path, encoding="utf-8") as f:
            thresholds = json.load(f)

        if isinstance(manifest_thresh, dict):
            expected_hash = manifest_thresh.get("sha256")
            if expected_hash:
                verify_artifact_strict(
                    thresh_path,
                    expected_hash,
                    f"thresholds_p{percentile}.json",
                )
                logger.info(
                    f"Integridade de thresholds verificada: thresholds_p{percentile}.json"
                )
            elif effective_require_hash:
                raise ValueError(
                    f"Manifesto invalido para thresholds_p{percentile}.json: campo sha256 ausente."
                )
            else:
                logger.warning(
                    f"sha256 de thresholds nao disponivel para p{percentile} no manifesto."
                )
        elif effective_require_hash:
            raise ValueError(
                f"Manifesto invalido para thresholds_p{percentile}.json: metadados ausentes em modo estrito."
            )
        else:
            logger.warning(
                f"Thresholds p{percentile} carregados sem metadados de integridade (modo permissivo)."
            )
        return thresholds
    if strict_integrity:
        raise FileNotFoundError(
            f"thresholds_p{percentile}.json nao encontrado em modo estrito: {thresh_path}"
        )
    logger.warning(
        f"thresholds_p{percentile}.json nao encontrado em {models_dir}. "
        "Os thresholds serao recalculados nos novos dados (modo degradado). "
        f"Para usar thresholds do treino, adicione geracao de thresholds_p{percentile}.json "
        "no run_experiment()."
    )
    return None


def predict(
    input_path,
    models_dir,
    config_path="config_mapeamento.yaml",
    output_dir="outputs_inference",
    percentile=95,
    strict_integrity=True,
    tf_device="auto",
):
    """
    Classifica novos dados usando modelos ja treinados.

    Args:
        input_path: Caminho para os novos dados (.csv ou .parquet).
        models_dir: Diretorio com modelos e scalers salvos pelo treinamento.
        config_path: Caminho do YAML de configuracao.
        output_dir: Diretorio para salvar os resultados da inferencia.
        percentile: Percentil a usar para threshold (default: 95).
        tf_device: Runtime TensorFlow ('auto', 'cpu', 'gpu').
    Returns:
        pd.DataFrame: DataFrame com scores e ensemble_alert.
    """
    # CONTRATO DE ARTEFATOS:
    # OBRIGATORIOS: models_manifest.json, scaler.joblib, iso_*.joblib, hbos_*.joblib
    # CONDICIONAIS (temporal): gru_scaler.joblib, temporal_*.h5
    # MODO DEGRADADO: thresholds ausentes -> recalibra (WARNING, nao excecao)
    if not logger.handlers:
        from src.utils.logger_utils import setup_logger

        setup_logger(name="sspdf")

    input_path, models_dir, config_path, output_dir = _normalize_and_validate_inference_paths(
        input_path=input_path,
        models_dir=models_dir,
        config_path=config_path,
        output_dir=output_dir,
    )

    logger.warning(
        "[TRUST-BOUNDARY] --models-dir e tratado como origem ESTRITAMENTE CONFIAVEL. "
        "Nao aponte para diretorios de terceiros ou nao auditados."
    )
    if os.path.basename(os.path.normpath(models_dir)) != "models_saved":
        logger.warning(
            "[TRUST-BOUNDARY] --models-dir nao termina com 'models_saved'. "
            "Fluxo institucional recomendado: outputs/<run_id>/models_saved."
        )

    if not strict_integrity:
        logger.warning(
            "[SEGURANCA] --allow-legacy-manifest ativo: validacao SHA256 DESATIVADA. "
            "Certifique-se de que os artefatos em --models-dir sao de origem estritamente controlada. "
            "Use somente para runs antigas sem manifesto ou em ambiente isolado."
        )
        logger.warning(
            "[SEGURANCA] Artefatos de modelos (joblib/.h5) podem executar desserializacao de alto risco "
            "quando provenientes de origem nao confiavel. "
            "Em producao institucional, mantenha strict_integrity=True e nao use --allow-legacy-manifest."
        )

    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MODO DE INFERENCIA - SSP-DF Pipeline")
    logger.info(f"   Executor SO: {resolve_os_user()}")
    logger.info(f"   Input:      {input_path}")
    logger.info(f"   Models dir: {models_dir}")
    logger.info(f"   Output:     {output_dir}")
    logger.info("=" * 80)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    tf, tf_runtime = configure_tensorflow_runtime(tf_device=tf_device)
    tf_seed = int(config.get("random_state", 42))
    tf.random.set_seed(tf_seed)
    logger.info(
        f"TensorFlow runtime em inferencia | solicitado={tf_runtime['requested']} "
        f"| ativo={tf_runtime['active']} | gpus_detectadas={tf_runtime['gpu_count']}"
    )

    # Importa apos configurar runtime para respeitar modo CPU/GPU.
    from src.models.temporal_autoencoder import TemporalAutoencoder

    # 1. Carregar e validar dados novos
    logger.info("ETAPA 1: Carga e validacao dos novos dados")
    proc = DataProcessor(config)
    proc.models_dir = models_dir
    df = proc.load_and_standardize(input_path)
    map_cols = config["mapeamento_colunas"]
    df = df.sort_values(map_cols["timestamp"]).reset_index(drop=True)
    logger.info(f"   {len(df):,} registros carregados e validados")

    # 2. Feature engineering SEM fit (apenas transform)
    logger.info("ETAPA 2: Feature engineering")
    df, inferred_features = proc.feature_engineering(df)
    proc.features_to_use = inferred_features

    # 3. Carregar manifesto e modelos
    logger.info("ETAPA 3: Carregando modelos")
    manifest = load_models_manifest(models_dir)
    has_integrity_metadata = bool(manifest.get("_integrity_available"))
    require_hash = bool(strict_integrity and has_integrity_metadata)
    (
        iso_models,
        hbos_models,
        temporal_entries,
        scaler,
        gru_scaler,
    ) = _load_models_from_manifest(
        manifest,
        strict_integrity=strict_integrity,
        require_hash=require_hash,
    )

    if strict_integrity and manifest.get("_integrity_available"):
        logger.info("   Integridade SHA256 validada para artefatos do manifesto.")
    elif strict_integrity:
        logger.warning(
            "   Manifesto sem hashes SHA256 (modo legado). Integridade nao foi validada."
        )

    feature_schema = _load_feature_schema(
        models_dir,
        manifest=manifest,
        strict_integrity=strict_integrity,
    )
    if feature_schema:
        logger.info("   Schema de features do treino carregado para alinhamento.")
    else:
        logger.warning(
            "   Schema de features nao encontrado. "
            "Inferencia usara fallback legado baseado no scaler/config."
        )

    # Carregar scaler principal do treino (NUNCA re-ajustar)
    if scaler is None:
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler principal nao encontrado: {scaler_path}. "
                "Execute run_experiment() primeiro para treinar os modelos."
            )
        scaler = joblib.load(scaler_path)
        logger.info(f"   Scaler carregado de {scaler_path} (sem re-ajuste)")
    else:
        logger.info("   Scaler carregado via manifesto (sem re-ajuste)")

    expected_main_features = _resolve_expected_main_features(feature_schema, scaler)
    if expected_main_features:
        df = _align_df_with_expected_ra(
            df,
            expected_main_features,
            context_label="Scaler principal",
        )
        remaining_cols = [c for c in df.columns if c not in expected_main_features]
        df = df[expected_main_features + remaining_cols]
        proc.features_to_use = expected_main_features
    else:
        proc.features_to_use = inferred_features
        logger.warning(
            "   Nao foi possivel resolver ordem esperada do scaler. "
            "Usando features inferidas do lote atual (modo legado)."
        )

    proc.scaler = scaler
    df = proc.transform_scaler(df)
    if expected_main_features:
        _ensure_expected_features_present(
            df,
            expected_main_features,
            context_label="transform_scaler",
        )

    fallback_iso_features = get_features_for_model("isolation_forest", df.columns.tolist())
    fallback_hbos_features = get_features_for_model("hbos", df.columns.tolist())
    fallback_gru_features = get_features_for_model("gru", df.columns.tolist())
    legacy_schema_missing = not bool(feature_schema)

    iso_features = _resolve_feature_list(
        feature_schema,
        "iso_features",
        fallback_iso_features,
        context_label="ISO",
        warn_on_fallback=legacy_schema_missing,
    )
    hbos_features = _resolve_feature_list(
        feature_schema,
        "hbos_features",
        fallback_hbos_features,
        context_label="HBOS",
        warn_on_fallback=legacy_schema_missing,
    )
    gru_features = _resolve_feature_list(
        feature_schema, "gru_features", fallback_gru_features
    )

    x_iso = _build_feature_matrix(df, iso_features, "ISO")
    x_hbos = _build_feature_matrix(df, hbos_features, "HBOS")

    # Dois scalers: scaler.joblib (ISO/HBOS) e gru_scaler.joblib (Temporal).
    # Ver config/feature_config.py para explicacao do motivo da separacao.
    # GRU Scaler separado (lat/lon)
    if gru_scaler is not None:
        x_gru = gru_scaler.transform(_build_feature_matrix(df, gru_features, "GRU"))
        logger.info("   GRU Scaler carregado via manifesto")
    else:
        gru_scaler_path = os.path.join(models_dir, "gru_scaler.joblib")
        if os.path.exists(gru_scaler_path):
            gru_scaler = joblib.load(gru_scaler_path)
            x_gru = gru_scaler.transform(_build_feature_matrix(df, gru_features, "GRU"))
            logger.info(f"   GRU Scaler carregado de {gru_scaler_path}")
        else:
            logger.warning("gru_scaler.joblib nao encontrado. GRU nao sera executado.")
            x_gru = None

    # 4. Scores ISO
    logger.info("ETAPA 4: Scores Isolation Forest")
    score_columns_audit = []
    thresholds_loaded = load_thresholds(
        models_dir,
        percentile,
        manifest=manifest,
        strict_integrity=strict_integrity,
        require_hash=require_hash,
    )
    optimizer = ThresholdOptimizer([percentile])

    for model_name, iso_model in iso_models.items():
        scores = -iso_model.score_samples(x_iso)
        tag = model_name
        df[f"{tag}_score"] = scores
        score_columns_audit.append(f"{tag}_score")
        if thresholds_loaded and tag in thresholds_loaded:
            thresh = thresholds_loaded[tag]
            df[f"{tag}_p{percentile}_label"] = (scores >= thresh).astype(float)
            logger.info(f"   {tag}: threshold={thresh:.4f} (do treino)")
        else:
            df, _ = optimizer.apply_dynamic_thresholds(
                df,
                f"{tag}_score",
                tag,
                calibration_scores=scores,  # fallback: calibrar nos novos dados
            )
            logger.warning(f"   {tag}: threshold calculado nos NOVOS dados (degradado)")

    # 5. Scores HBOS
    logger.info("ETAPA 5: Scores HBOS")
    for model_name, hbos_model in hbos_models.items():
        scores = hbos_model.decision_function(x_hbos)
        tag = model_name
        df[f"{tag}_score"] = scores
        score_columns_audit.append(f"{tag}_score")
        if thresholds_loaded and tag in thresholds_loaded:
            thresh = thresholds_loaded[tag]
            df[f"{tag}_p{percentile}_label"] = (scores >= thresh).astype(float)
            logger.info(f"   {tag}: threshold={thresh:.4f} (do treino)")
        else:
            df, _ = optimizer.apply_dynamic_thresholds(
                df, f"{tag}_score", tag, calibration_scores=scores
            )
            logger.warning(f"   {tag}: threshold calculado nos novos dados (degradado)")

    # 6. Scores Temporal (GRU)
    if x_gru is not None and temporal_entries:
        logger.info("ETAPA 6: Scores Temporal (GRU Autoencoder)")
        temporal_config = config.get("parametros", {}).get("temporal", {})
        window_size = temporal_config.get("window_size", 3)
        gap_seconds = (
            config.get("configuracoes_gerais", {}).get("gap_segmentation_seconds", 1800)
        )

        temporal_pipe = TemporalAutoencoder(
            X_data=x_gru,
            vehicle_ids=df[map_cols["placa"]].values,
            timestamps=df[map_cols["timestamp"]].values,
            original_indices=df.index.values,
            window_size=window_size,
            max_gap_seconds=gap_seconds,
            arch_type=temporal_config.get("arch_type", "gru"),
        )
        x_seq_all, indices_all, _ = temporal_pipe.create_sequences_with_index()

        for model_info in temporal_entries:
            model_path = model_info["path"]
            model_name = model_info["tag"]
            if not os.path.exists(model_path):
                logger.warning(f"Modelo temporal nao encontrado: {model_path}")
                continue
            temporal_model = tf.keras.models.load_model(model_path)
            if len(x_seq_all) > 0:
                x_pred = temporal_model.predict(x_seq_all, verbose=0)
                mse = np.mean(np.power(x_seq_all - x_pred, 2), axis=(1, 2))
                tag = model_name
                df.loc[indices_all, f"{tag}_score"] = mse
                score_columns_audit.append(f"{tag}_score")
                if thresholds_loaded and tag in thresholds_loaded:
                    thresh = thresholds_loaded[tag]
                    df[f"{tag}_p{percentile}_label"] = np.where(
                        df[f"{tag}_score"].isna(),
                        np.nan,
                        (df[f"{tag}_score"] >= thresh).astype(float),
                    )
                else:
                    calib = mse  # fallback degradado
                    df, _ = optimizer.apply_dynamic_thresholds(
                        df, f"{tag}_score", tag, calibration_scores=calib
                    )
                logger.info(f"   {tag}: {len(indices_all)} sequencias avaliadas")
            else:
                logger.warning(f"   {tag}: nenhuma sequencia formavel no novo lote")

    # 7. Decisao do ensemble
    logger.info("ETAPA 7: Decisao final do ensemble")
    df = compute_ensemble_decision(df, percentile=percentile)
    if "ensemble_alert" in df.columns:
        vehicle_risk = compute_vehicle_risk_summary(
            df,
            placa_col=map_cols["placa"],
            percentile=percentile,
        )
    else:
        logger.warning(
            "Nenhuma coluna de decisao final foi gerada; pulando ranking de risco."
        )
        vehicle_risk = pd.DataFrame()

    # 8. Exportar resultados
    logger.info("ETAPA 8: Exportando resultados")
    result_path = os.path.join(output_dir, "inference_result.parquet")
    df.to_parquet(result_path, index=False)

    alerts_only = df[df["ensemble_alert"] == 1.0].copy()
    alerts_path = os.path.join(metrics_dir, "alertas_ensemble.csv")

    id_cols = [
        map_cols["placa"],
        map_cols["timestamp"],
        map_cols["latitude"],
        map_cols["longitude"],
    ]
    alert_export_cols = [c for c in id_cols if c in df.columns] + [
        "ensemble_alert",
        "ensemble_vote_pct",
        "n_models_scored",
    ]
    alerts_only[alert_export_cols].to_csv(alerts_path, index=False)

    if not vehicle_risk.empty:
        vehicle_risk.to_csv(
            os.path.join(metrics_dir, "vehicle_risk_ranking.csv"), index=False
        )

    logger.info(f"   resultado: {result_path}")
    logger.info(f"   alertas:   {alerts_path} ({len(alerts_only):,} alertas)")
    logger.info("INFERENCIA CONCLUIDA!")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSP-DF - Modo de Inferencia")
    parser.add_argument(
        "--models-dir",
        required=True,
        help=(
            "Diretorio com modelos treinados (trust boundary): trate como origem "
            "estritamente confiavel, preferencialmente outputs/<run_id>/models_saved."
        ),
    )
    parser.add_argument("--input", required=True, help="Dados novos (.csv ou .parquet)")
    parser.add_argument("--config", default="config_mapeamento.yaml")
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default="outputs_inference",
        help="Diretorio de saida da inferencia",
    )
    parser.add_argument("--percentile", type=int, default=95)
    parser.add_argument(
        "--tf-device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help=(
            "Dispositivo TensorFlow na inferencia: "
            "auto (usa GPU se houver), cpu (forca CPU), gpu (exige GPU)."
        ),
    )
    parser.add_argument(
        "--allow-legacy-manifest",
        action="store_true",
        help=(
            "Permite manifesto sem hashes SHA256 (compatibilidade com runs antigas). "
            "Quando nao informado, a inferencia exige e valida hashes quando disponiveis. "
            "NAO recomendado para ambiente institucional de producao."
        ),
    )
    args = parser.parse_args()
    predict(
        input_path=args.input,
        models_dir=args.models_dir,
        config_path=args.config,
        output_dir=args.output,
        percentile=args.percentile,
        strict_integrity=not args.allow_legacy_manifest,
        tf_device=args.tf_device,
    )
