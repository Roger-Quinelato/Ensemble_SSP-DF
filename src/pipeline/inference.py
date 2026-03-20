"""
Modo de inferencia do pipeline SSP-DF.

Carrega modelos ja treinados e classifica novos dados sem re-treinar.

Uso:
    python -m src.pipeline.inference \
        --models-dir outputs/<run_id>/models_saved \
        --config config_mapeamento.yaml \
        --input novos_dados.csv \
        --output outputs_inference/
"""

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

from config.feature_config import get_features_for_model
from src.data.data_processor import DataProcessor
from src.models.temporal_autoencoder import TemporalAutoencoder
from src.utils.ensemble_decision import (
    compute_ensemble_decision,
    compute_vehicle_risk_summary,
)
from src.utils.evaluation import ThresholdOptimizer
from src.utils.artifact_utils import verify_artifact_strict

logger = logging.getLogger("sspdf")


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
    manifest["_integrity_available"] = any(integrity_fields)
    return manifest


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
            gru_scaler = joblib.load(gru_scaler_path)

    return iso_models, hbos_models, temporal_entries, scaler, gru_scaler


def load_thresholds(models_dir, percentile=95, manifest=None):
    """
    Carrega thresholds de producao salvos pelo treinamento.
    Se nao existir arquivo de thresholds, usa os percentis padrao dos scores.
    """
    manifest_thresh = None
    thresh_path = os.path.join(models_dir, f"thresholds_p{percentile}.json")
    if manifest and isinstance(manifest.get("thresholds"), dict):
        manifest_thresh = manifest["thresholds"].get(str(percentile))
        if isinstance(manifest_thresh, dict) and manifest_thresh.get("path"):
            thresh_path = _resolve_path(models_dir, manifest_thresh["path"])

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
            else:
                logger.warning(
                    f"sha256 de thresholds nao disponivel para p{percentile} no manifesto."
                )
        return thresholds
    logger.warning(
        f"thresholds_p{percentile}.json nao encontrado em {models_dir}. "
        "Os thresholds serao recalculados nos novos dados (modo degradado). "
        "Para usar thresholds do treino, adicione geracao de thresholds_p95.json "
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
):
    """
    Classifica novos dados usando modelos ja treinados.

    Args:
        input_path: Caminho para os novos dados (.csv ou .parquet).
        models_dir: Diretorio com modelos e scalers salvos pelo treinamento.
        config_path: Caminho do YAML de configuracao.
        output_dir: Diretorio para salvar os resultados da inferencia.
        percentile: Percentil a usar para threshold (default: 95).
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

    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MODO DE INFERENCIA - SSP-DF Pipeline")
    logger.info(f"   Input:      {input_path}")
    logger.info(f"   Models dir: {models_dir}")
    logger.info(f"   Output:     {output_dir}")
    logger.info("=" * 80)

    with open(config_path) as f:
        config = yaml.safe_load(f)

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

    # Carregar scaler principal do treino (NUNCA re-ajustar)
    if scaler is None:
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler principal nao encontrado: {scaler_path}. "
                "Execute run_experiment() primeiro para treinar os modelos."
            )
        df = proc.transform_scaler(df, scaler_path=scaler_path)
        logger.info(f"   Scaler carregado de {scaler_path} (sem re-ajuste)")
    else:
        proc.scaler = scaler
        df = proc.transform_scaler(df)
        logger.info("   Scaler carregado via manifesto (sem re-ajuste)")

    iso_features = get_features_for_model("isolation_forest", df.columns.tolist())
    hbos_features = get_features_for_model("hbos", df.columns.tolist())
    gru_features = get_features_for_model("gru", df.columns.tolist())

    x_iso = df[iso_features].values
    x_hbos = df[hbos_features].values

    # Dois scalers: scaler.joblib (ISO/HBOS) e gru_scaler.joblib (Temporal).
    # Ver config/feature_config.py para explicacao do motivo da separacao.
    # GRU Scaler separado (lat/lon)
    if gru_scaler is not None:
        x_gru = gru_scaler.transform(df[gru_features].values)
        logger.info("   GRU Scaler carregado via manifesto")
    else:
        gru_scaler_path = os.path.join(models_dir, "gru_scaler.joblib")
        if os.path.exists(gru_scaler_path):
            gru_scaler = joblib.load(gru_scaler_path)
            x_gru = gru_scaler.transform(df[gru_features].values)
            logger.info(f"   GRU Scaler carregado de {gru_scaler_path}")
        else:
            logger.warning("gru_scaler.joblib nao encontrado. GRU nao sera executado.")
            x_gru = None

    # 4. Scores ISO
    logger.info("ETAPA 4: Scores Isolation Forest")
    score_columns_audit = []
    thresholds_loaded = load_thresholds(models_dir, percentile, manifest=manifest)
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
            temporal_model = keras.models.load_model(model_path)
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
        vehicle_risk = compute_vehicle_risk_summary(df, placa_col=map_cols["placa"])
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
        "--models-dir", required=True, help="Diretorio com modelos treinados"
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
        "--allow-legacy-manifest",
        action="store_true",
        help=(
            "Permite manifesto sem hashes SHA256 (compatibilidade com runs antigas). "
            "Quando nao informado, a inferencia exige e valida hashes quando disponiveis."
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
    )
