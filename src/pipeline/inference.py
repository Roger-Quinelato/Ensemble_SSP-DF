"""
Modo de inferencia do pipeline SSP-DF.

Carrega modelos ja treinados e classifica novos dados sem re-treinar.

Uso:
    python -m src.pipeline.inference \
        --models-dir outputs/models_saved \
        --config config_mapeamento.yaml \
        --input novos_dados.csv \
        --output outputs_inference/
"""

import argparse
import json
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
from src.utils.logger_utils import logger


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
        manifest = {"iso": [], "hbos": [], "temporal": [], "scalers": {}}
        for fname in os.listdir(models_dir):
            fpath = os.path.join(models_dir, fname)
            if fname.startswith("iso_") and fname.endswith(".joblib"):
                manifest["iso"].append(
                    {"name": fname.replace(".joblib", "").upper(), "path": fpath}
                )
            elif fname.startswith("hbos_") and fname.endswith(".joblib"):
                manifest["hbos"].append(
                    {"name": fname.replace(".joblib", "").upper(), "path": fpath}
                )
            elif fname.startswith("temporal_") and fname.endswith(".h5"):
                name = fname.replace(".h5", "").replace("temporal_", "Temporal_")
                manifest["temporal"].append({"name": name, "path": fpath})
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
    for key, value in manifest.get("scalers", {}).items():
        manifest["scalers"][key] = _resolve_path(models_dir, value)
    return manifest


def load_thresholds(models_dir, percentile=95):
    """
    Carrega thresholds de producao salvos pelo treinamento.
    Se nao existir arquivo de thresholds, usa os percentis padrao dos scores.
    """
    thresh_path = os.path.join(models_dir, f"thresholds_p{percentile}.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            return json.load(f)
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

    # Carregar scaler principal do treino (NUNCA re-ajustar)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler principal nao encontrado: {scaler_path}. "
            "Execute run_experiment() primeiro para treinar os modelos."
        )
    df = proc.transform_scaler(df, scaler_path=scaler_path)
    logger.info(f"   Scaler carregado de {scaler_path} (sem re-ajuste)")

    # 3. Carregar manifesto e modelos
    logger.info("ETAPA 3: Carregando modelos")
    manifest = load_models_manifest(models_dir)

    iso_features = get_features_for_model("isolation_forest", df.columns.tolist())
    hbos_features = get_features_for_model("hbos", df.columns.tolist())
    gru_features = get_features_for_model("gru", df.columns.tolist())

    x_iso = df[iso_features].values
    x_hbos = df[hbos_features].values

    # GRU Scaler separado (lat/lon)
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
    thresholds_loaded = load_thresholds(models_dir, percentile)
    optimizer = ThresholdOptimizer([percentile])

    for model_info in manifest.get("iso", []):
        model_path = _resolve_path(models_dir, model_info["path"])
        model_name = model_info["name"]
        if not os.path.exists(model_path):
            logger.warning(f"Modelo nao encontrado: {model_path}")
            continue
        iso_model = joblib.load(model_path)
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
    for model_info in manifest.get("hbos", []):
        model_path = _resolve_path(models_dir, model_info["path"])
        model_name = model_info["name"]
        if not os.path.exists(model_path):
            logger.warning(f"Modelo nao encontrado: {model_path}")
            continue
        hbos_model = joblib.load(model_path)
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
    if x_gru is not None and manifest.get("temporal"):
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
        x_seq_all, indices_all = temporal_pipe.create_sequences_with_index()

        for model_info in manifest.get("temporal", []):
            model_path = _resolve_path(models_dir, model_info["path"])
            model_name = model_info["name"]
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
    parser.add_argument("--output", default="outputs_inference")
    parser.add_argument("--percentile", type=int, default=95)
    args = parser.parse_args()
    predict(
        input_path=args.input,
        models_dir=args.models_dir,
        config_path=args.config,
        output_dir=args.output,
        percentile=args.percentile,
    )
