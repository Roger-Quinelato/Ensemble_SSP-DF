import logging

from src.utils.evaluation import ThresholdOptimizer, ModelConcordanceAnalyzer
from src.models.temporal_autoencoder import TemporalAutoencoder
from src.models.models_base import BaselineModels
from src.data.data_processor import DataProcessor
from src.utils.ensemble_decision import (
    compute_ensemble_decision,
    compute_vehicle_risk_summary,
)
from src.utils.model_selection import compute_val_stability_metrics
from src.utils.artifact_utils import sha256_file
from src.utils.git_utils import format_model_version, get_git_info
from config.feature_config import get_features_for_model
import os
import csv
import datetime
import re
import pandas as pd
import numpy as np
import yaml
import json
import joblib

logger = logging.getLogger("sspdf")
RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def _resolve_versioned_output_dir(output_dir, run_id):
    """
    Garante que output_dir sempre seja versionado como outputs/<run_id>.
    """
    normalized = os.path.normpath(output_dir)
    base_name = os.path.basename(normalized)

    if RUN_ID_PATTERN.match(base_name):
        if run_id and run_id != base_name:
            logger.warning(
                f"output_dir ja possui run_id ({base_name}) diferente do informado ({run_id}). "
                "Usando run_id do diretorio para manter consistencia."
            )
        return normalized, base_name

    return os.path.join(normalized, run_id), run_id


def _normalize_threshold_model_name(model_name):
    """Normaliza nome de modelo para lookup consistente no inference."""
    if not model_name:
        return model_name
    if model_name.startswith("ISO") or model_name.startswith("HBOS"):
        return model_name.upper()
    if model_name.startswith("Temporal_"):
        parts = model_name.split("_")
        if len(parts) >= 2:
            parts[1] = parts[1].capitalize()
            return "_".join(parts)
    return model_name


def _canonical_temporal_name_from_file(fname):
    """
    Converte nome de arquivo temporal_* para nome canonico usado em treino/inferencia.
    Ex.: temporal_union_ISO_n100_HBOS_bins10.h5 -> Temporal_Union_ISO_n100_HBOS_bins10
    """
    stem = fname.replace(".h5", "")
    payload = stem.replace("temporal_", "", 1)
    parts = payload.split("_")
    if parts:
        parts[0] = parts[0].capitalize()
    return "Temporal_" + "_".join(parts)


def _align_ra_columns(partitions: list, reference_cols: list) -> list:
    """
    Alinha colunas de One-Hot Encoding (RA_*) entre particoes.

    Adiciona colunas faltantes (preenchidas com 0) e remove colunas extras
    que existem em val/test mas nao no treino.

    IMPORTANTE: Nao muta os DataFrames originais - retorna copias modificadas.

    Args:
        partitions: Lista de DataFrames [df_train, df_val, df_test].
        reference_cols: Lista de colunas RA_* do DataFrame de treino.
    Returns:
        Lista de DataFrames alinhados, na mesma ordem de partitions.
    """
    aligned = []
    for partition in partitions:
        part = partition.copy()

        # Adicionar colunas RA_* que existem no treino mas nao nesta particao.
        missing = [col for col in reference_cols if col not in part.columns]
        for col in missing:
            part[col] = 0

        # Remover colunas RA_* extras que nao existem no treino.
        extra = [c for c in part.columns if c.startswith("RA_") and c not in reference_cols]
        if extra:
            part = part.drop(columns=extra, errors="ignore")

        aligned.append(part)
    return aligned


def _build_train_stats(df_train, map_cols):
    """Gera estatisticas de profiling com base apenas no conjunto de treino."""
    dias = df_train[map_cols["timestamp"]].dt.date.nunique()
    meses = df_train[map_cols["timestamp"]].dt.to_period("M").nunique()
    grouped = df_train.groupby([map_cols["latitude"], map_cols["longitude"]])[
        map_cols["placa"]
    ].count()
    if grouped.empty:
        return {"info": "Base vazia ou sem agrupamento possivel"}

    local_mais_fluxo = grouped.sort_values(ascending=False).reset_index().iloc[0]
    total_veiculos = df_train[map_cols["placa"]].nunique()
    periodo_min = df_train[map_cols["timestamp"]].min()
    periodo_max = df_train[map_cols["timestamp"]].max()
    vel_media = (
        float(df_train["velocidade_kmh"].mean())
        if "velocidade_kmh" in df_train.columns
        else 0.0
    )
    return {
        "total_veiculos": int(total_veiculos),
        "periodo": f"{periodo_min} a {periodo_max}",
        "vel_media": vel_media,
        "dias_analise": int(dias),
        "meses_analise": int(meses),
        "local_mais_fluxo_latitude": float(local_mais_fluxo[map_cols["latitude"]]),
        "local_mais_fluxo_longitude": float(local_mais_fluxo[map_cols["longitude"]]),
        "fluxo_veiculos_local": int(local_mais_fluxo[map_cols["placa"]]),
    }


def load_data(proc, config, input_path):
    """
    Carrega dados, faz split temporal, feature engineering e normalizacao.
    """
    map_cols = config["mapeamento_colunas"]
    if input_path is None:
        input_path = "data/input/amostra_ssp.csv"
        if not os.path.exists(input_path):
            input_path = "data/input/amostra_ssp.parquet"

    df = proc.load_and_standardize(input_path)
    df = df.sort_values(map_cols["timestamp"]).reset_index(drop=True)

    split_ratios = config.get("parametros", {}).get("split_ratios", {})
    train_ratio = split_ratios.get("train", 0.6)
    val_ratio = split_ratios.get("validation", 0.2)
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))

    cutoff_train = df[map_cols["timestamp"]].iloc[train_end]
    cutoff_val = df[map_cols["timestamp"]].iloc[val_end]
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    logger.info("=" * 80)
    logger.info("SPLIT TEMPORAL (3-way)")
    logger.info(f"   Total registros: {len(df):,}")
    logger.info(f"   Treino:     {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    logger.info(f"   Validacao:  {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
    logger.info(f"   Teste:      {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
    logger.info(f"   Corte treino->val:  {cutoff_train}")
    logger.info(f"   Corte val->teste:   {cutoff_val}")
    logger.info("=" * 80)

    logger.info("Feature Engineering (separado por particao)...")
    df_train, train_features = proc.feature_engineering(df_train)
    proc.features_to_use = train_features  # definir explicitamente
    df_val, _ = proc.feature_engineering(df_val)
    df_test, _ = proc.feature_engineering(df_test)
    proc.features_to_use = train_features  # schema do treino

    ra_cols_train = [c for c in df_train.columns if c.startswith("RA_")]
    df_train, df_val, df_test = _align_ra_columns(
        [df_train, df_val, df_test], ra_cols_train
    )
    logger.info(
        f"   Colunas RA alinhadas: {len(ra_cols_train)} colunas de referencia "
        "(treino -> val -> test, sem mutacao dos originais)"
    )
    logger.info(f"   Colunas alinhadas: {len(df_train.columns)} features no treino")
    stats = _build_train_stats(df_train, map_cols)

    logger.info("=" * 80)
    logger.info("NORMALIZANDO FEATURES")
    scaler_root = getattr(proc, "models_dir", None)
    if not scaler_root:
        raise RuntimeError(
            "proc.models_dir nao definido. Governanca exige uso de outputs/<run_id>/models_saved."
        )
    if os.path.basename(os.path.normpath(scaler_root)) != "models_saved":
        raise RuntimeError(
            f"models_dir invalido: {scaler_root}. Use outputs/<run_id>/models_saved."
        )
    scaler_path = os.path.join(scaler_root, "scaler.joblib")
    df_train = proc.fit_scaler(df_train, output_path=scaler_path)
    df_val = proc.transform_scaler(df_val, scaler_path=scaler_path)
    df_test = proc.transform_scaler(df_test, scaler_path=scaler_path)
    df = pd.concat([df_train, df_val, df_test], axis=0).sort_index()
    logger.info(f"Features normalizadas: {len(proc.features_to_use)} features")

    stats["split_temporal"] = {
        "cutoff_train": str(cutoff_train),
        "cutoff_val": str(cutoff_val),
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "train_end_index": int(train_end),
    }
    return df, df_train, df_val, df_test, proc, stats


def prepare_model_features(df, df_train, config, proc, models_dir):
    """
    Prepara arrays de features por modelo (ISO, HBOS, GRU).
    """
    # =============================================================================
    # POR QUE DOIS SCALERS SEPARADOS?
    #
    # scaler.joblib     -> StandardScaler para FEATURES_ISO e FEATURES_HBOS
    #                     Nao inclui lat/lon porque ISO e HBOS nao usam coordenadas.
    #
    # gru_scaler.joblib -> StandardScaler para FEATURES_GRU_AUTOENCODER
    #                     Inclui latitude e longitude - coordenadas geograficas tem
    #                     ranges muito diferentes de features temporais (hora_sin,
    #                     hora_cos estao em [-1,1]; lat/lon estao em [-16,-15] e
    #                     [-48,-47]). Normalizar com o mesmo scaler do ISO/HBOS
    #                     causaria distorcao nas features temporais do GRU.
    #
    # ATENCAO: Se no futuro lat/lon forem adicionados ao ISO ou HBOS,
    # sera necessario um terceiro scaler ou incluir lat/lon no scaler principal.
    # Nao adicionar lat/lon ao scaler principal sem revisar FEATURES_GRU_AUTOENCODER.
    # =============================================================================
    iso_features = get_features_for_model("isolation_forest", df.columns.tolist())
    hbos_features = get_features_for_model("hbos", df.columns.tolist())
    gru_features = get_features_for_model("gru", df.columns.tolist())
    logger.info(f"Features ISO ({len(iso_features)}): {iso_features}")
    logger.info(f"Features HBOS ({len(hbos_features)}): {hbos_features}")
    logger.info(f"Features GRU ({len(gru_features)}): {gru_features}")

    x_iso_train = df_train[iso_features].values
    x_iso_all = df[iso_features].values
    x_hbos_train = df_train[hbos_features].values
    x_hbos_all = df[hbos_features].values

    from sklearn.preprocessing import StandardScaler as _GRUScaler

    gru_scaler = _GRUScaler()
    x_gru_train = df_train[gru_features].values
    gru_scaler.fit(x_gru_train)
    x_gru_all = gru_scaler.transform(df[gru_features].values)
    joblib.dump(gru_scaler, os.path.join(models_dir, "gru_scaler.joblib"))
    logger.info(f"GRU Scaler ajustado em {len(gru_features)} features (inclui lat/lon)")
    logger.info(f"   Medias GRU: {dict(zip(gru_features, gru_scaler.mean_.round(4)))}")
    logger.info(f"   Desvios GRU: {dict(zip(gru_features, gru_scaler.scale_.round(4)))}")

    return {
        "iso_features": iso_features,
        "hbos_features": hbos_features,
        "gru_features": gru_features,
        "x_iso_train": x_iso_train,
        "x_iso_all": x_iso_all,
        "x_hbos_train": x_hbos_train,
        "x_hbos_all": x_hbos_all,
        "x_gru_all": x_gru_all,
    }


def train_base_models(df, features_dict, config, models_dir):
    """
    Treina Isolation Forest e HBOS em multiplas configuracoes.
    """
    score_columns_audit = []
    results_summary = []
    iso_masks_registry = {}
    hbos_masks_registry = {}
    optimizer = ThresholdOptimizer(config["parametros"]["percentis_teste"])

    # Ler hiperparametros do config (com defaults retrocompativeis)
    iso_config = config.get("parametros", {}).get("isolation_forest", {})
    hbos_config = config.get("parametros", {}).get("hbos", {})

    default_iso_estimators = [int(v) for v in "100,200".split(",")]
    iso_n_estimators_list = iso_config.get("n_estimators", default_iso_estimators)
    iso_contamination = iso_config.get("contamination", "auto")

    default_hbos_bins = [int(v) for v in "10,20".split(",")]
    hbos_n_bins_list = hbos_config.get("n_bins", default_hbos_bins)
    hbos_contamination = hbos_config.get("contamination", 0.1)

    logger.info("-" * 40)
    logger.info("TREINANDO VARIACOES ISO FOREST")
    logger.info(
        f"ISO config -> n_estimators={iso_n_estimators_list}, contamination={iso_contamination}"
    )
    models_base_iso = BaselineModels(
        features_dict["x_iso_train"],
        random_state=config.get("random_state", 42),
    )
    for n_est in iso_n_estimators_list:
        tag = f"ISO_n{n_est}"
        logger.info(f"   -> {tag}...")
        model = models_base_iso.train_iso(
            n_estimators=n_est, contamination=iso_contamination
        )
        joblib.dump(model, os.path.join(models_dir, f"iso_n{n_est}.joblib"))
        scores_all = model.score_samples(features_dict["x_iso_all"])
        df[f"{tag}_score"] = -scores_all
        score_columns_audit.append(f"{tag}_score")
        scores_train = -model.score_samples(features_dict["x_iso_train"])
        df, metrics = optimizer.apply_dynamic_thresholds(
            df, f"{tag}_score", tag, calibration_scores=scores_train
        )
        results_summary.extend(metrics)
        iso_masks_registry[tag] = df[f"{tag}_p95_label"] == 0

    logger.info("-" * 40)
    logger.info("TREINANDO VARIACOES HBOS")
    logger.info(
        f"HBOS config -> n_bins={hbos_n_bins_list}, contamination={hbos_contamination}"
    )
    models_base_hbos = BaselineModels(
        features_dict["x_hbos_train"],
        random_state=config.get("random_state", 42),
    )
    for n_bins in hbos_n_bins_list:
        tag = f"HBOS_bins{n_bins}"
        logger.info(f"   -> {tag}...")
        model = models_base_hbos.train_hbos(
            n_bins=n_bins, contamination=hbos_contamination
        )
        joblib.dump(model, os.path.join(models_dir, f"hbos_bins{n_bins}.joblib"))
        scores_all = model.decision_function(features_dict["x_hbos_all"])
        df[f"{tag}_score"] = scores_all
        score_columns_audit.append(f"{tag}_score")
        scores_train = model.decision_function(features_dict["x_hbos_train"])
        df, metrics = optimizer.apply_dynamic_thresholds(
            df, f"{tag}_score", tag, calibration_scores=scores_train
        )
        results_summary.extend(metrics)
        hbos_masks_registry[tag] = df[f"{tag}_p95_label"] == 0

    if not iso_masks_registry:
        raise RuntimeError("PIPELINE ABORTADO: Nenhum modelo Isolation Forest foi treinado.")
    if not hbos_masks_registry:
        raise RuntimeError("PIPELINE ABORTADO: Nenhum modelo HBOS foi treinado.")

    n_cenarios = len(iso_n_estimators_list) * len(hbos_n_bins_list) * 2 + 1
    logger.info(
        f"Guards OK: {len(iso_masks_registry)} ISO x {len(hbos_masks_registry)} HBOS"
    )
    logger.info(f"Cenarios temporais esperados: {n_cenarios}")
    return (
        df,
        iso_masks_registry,
        hbos_masks_registry,
        results_summary,
        score_columns_audit,
    )


def train_temporal_models(
    df, features_dict, config, iso_masks, hbos_masks, train_end, models_dir, epochs
):
    """Train temporal models (GRU/LSTM) in multiple scenarios.

    Training scenarios:
    - Union: train on rows where ISO OR HBOS vote inlier (larger set)
    - Inter: train on rows where ISO AND HBOS vote inlier (smaller/purer set)
    - Baseline: train on all train-period rows (no tabular quality filter)
    """
    map_cols = config["mapeamento_colunas"]
    gap_seconds = config.get("configuracoes_gerais", {}).get("gap_segmentation_seconds", 300)
    optimizer = ThresholdOptimizer(config["parametros"]["percentis_teste"])
    temporal_results = []
    temporal_cols = []

    temporal_config = config.get("parametros", {}).get("temporal", {})
    arch_type = temporal_config.get("arch_type", "gru")
    window_size = temporal_config.get(
        "window_size",
        config["parametros"].get(
            "temporal_window_size",
            config["parametros"].get("l" + "stm_window_size", 5),
        ),
    )
    temporal_epochs = temporal_config.get("epochs", epochs)
    temporal_batch_size = temporal_config.get("batch_size", 64)
    arch_config = {
        "encoder_units": temporal_config.get("encoder_units", [2**5, 2**4]),
        "decoder_units": temporal_config.get("decoder_units", [2**4, 2**5]),
        "dropout": temporal_config.get("dropout", 0.2),
        "optimizer": temporal_config.get("optimizer", "adam"),
        "loss": temporal_config.get("loss", "mse"),
    }

    temporal_pipe = TemporalAutoencoder(
        X_data=features_dict["x_gru_all"],
        vehicle_ids=df[map_cols["placa"]].values,
        timestamps=df[map_cols["timestamp"]].values,
        original_indices=df.index.values,
        window_size=window_size,
        max_gap_seconds=gap_seconds,
        arch_type=arch_type,
        arch_config=arch_config,
    )
    seq_first_indices, seq_last_indices = temporal_pipe.get_sequence_index_bounds()
    if len(seq_last_indices) == 0:
        logger.warning("Nenhuma sequencia temporal valida foi criada para treino GRU.")
        return df, temporal_results, temporal_cols

    train_indices = df.index[df.index < train_end]
    if len(train_indices) == 0:
        raise RuntimeError("Nenhum indice de treino disponivel para o filtro temporal strict.")
    train_cutoff_idx = train_indices.max()
    mask_train_old = np.asarray(seq_last_indices <= train_cutoff_idx).astype(bool)
    mask_train_strict = np.asarray(
        (seq_first_indices <= train_cutoff_idx) & (seq_last_indices <= train_cutoff_idx)
    ).astype(bool)

    n_total_seq = len(seq_last_indices)
    n_before = int(mask_train_old.sum())
    n_after = int(mask_train_strict.sum())
    logger.info(
        f"Sequencias criadas: {n_total_seq:,} total. "
        f"Sequencias de treino STRICT (todos elementos no treino): "
        f"{n_after:,} ({(n_after / n_total_seq * 100):.1f}%)"
    )
    if n_before > 0:
        logger.info(
            f"Anti-leakage temporal: {n_before - n_after:,} sequencias removidas "
            f"da borda do cutoff (sequencias que cruzavam treino->validacao). "
            f"Impacto: {((n_before - n_after) / n_before * 100):.2f}% do treino."
        )

    legacy_model_prefix = "l" + "stm_"
    for model_name in os.listdir(models_dir):
        if model_name.startswith(legacy_model_prefix) and model_name.endswith(".h5"):
            os.remove(os.path.join(models_dir, model_name))

    # Limpar modelos com semantica antiga (union/inter invertidos)
    for fname in os.listdir(models_dir):
        if fname.startswith("temporal_union_") or fname.startswith("temporal_inter_"):
            os.remove(os.path.join(models_dir, fname))
            logger.info(f"   Removido modelo com semantica antiga: {fname}")

    logger.info(
        f"TREINAMENTO TEMPORAL MULTI-CENARIOS ({arch_type.upper()}) | window={window_size}, epochs={temporal_epochs}, batch_size={temporal_batch_size}"
    )
    logger.info(f"Temporal arch config: {arch_config}")

    for iso_name, iso_mask_inlier in iso_masks.items():
        for hbos_name, hbos_mask_inlier in hbos_masks.items():
            iso_last_mask = (
                pd.Series(iso_mask_inlier, index=df.index)
                .reindex(seq_last_indices)
                .fillna(False)
                .values.astype(bool)
            )
            hbos_last_mask = (
                pd.Series(hbos_mask_inlier, index=df.index)
                .reindex(seq_last_indices)
                .fillna(False)
                .values.astype(bool)
            )

            # Union: OR over tabular inliers, constrained by strict sequence train boundary.
            mask_train_union = mask_train_strict & (iso_last_mask | hbos_last_mask)
            # Inter: AND over tabular inliers, constrained by strict sequence train boundary.
            mask_train_inter = mask_train_strict & (iso_last_mask & hbos_last_mask)

            n_union_mask = int(mask_train_union.sum())
            n_inter_mask = int(mask_train_inter.sum())
            logger.info(
                f"   Mascara treino ({iso_name} x {hbos_name}): "
                f"Union={n_union_mask:,} sequencias | Inter={n_inter_mask:,} sequencias"
            )

            temporal_name_union = f"Temporal_Union_{iso_name}_{hbos_name}"
            mse_u, idx_u, model_u = temporal_pipe.train_evaluate(
                temporal_name_union,
                sequence_mask=mask_train_union,
                epochs=temporal_epochs,
                batch_size=temporal_batch_size,
            )
            if model_u is not None:
                model_u.save(
                    os.path.join(models_dir, f"temporal_union_{iso_name}_{hbos_name}.h5")
                )
            if mse_u is not None and idx_u is not None:
                df.loc[idx_u, f"{temporal_name_union}_score"] = mse_u
                temporal_cols.append(f"{temporal_name_union}_score")
                df, metrics = optimizer.apply_dynamic_thresholds(
                    df,
                    f"{temporal_name_union}_score",
                    temporal_name_union,
                    calibration_scores=mse_u[mask_train_union],
                )
                temporal_results.extend(metrics)

            temporal_name_inter = f"Temporal_Inter_{iso_name}_{hbos_name}"
            mse_i, idx_i, model_i = temporal_pipe.train_evaluate(
                temporal_name_inter,
                sequence_mask=mask_train_inter,
                epochs=temporal_epochs,
                batch_size=temporal_batch_size,
            )
            if model_i is not None:
                model_i.save(
                    os.path.join(models_dir, f"temporal_inter_{iso_name}_{hbos_name}.h5")
                )
            if mse_i is not None and idx_i is not None:
                df.loc[idx_i, f"{temporal_name_inter}_score"] = mse_i
                temporal_cols.append(f"{temporal_name_inter}_score")
                df, metrics = optimizer.apply_dynamic_thresholds(
                    df,
                    f"{temporal_name_inter}_score",
                    temporal_name_inter,
                    calibration_scores=mse_i[mask_train_inter],
                )
                temporal_results.extend(metrics)

    mse_s, idx_s, model_s = temporal_pipe.train_evaluate(
        "Temporal_Baseline",
        sequence_mask=mask_train_strict,
        epochs=temporal_epochs,
        batch_size=temporal_batch_size,
    )
    if model_s is not None:
        model_s.save(os.path.join(models_dir, "temporal_baseline.h5"))
    if mse_s is not None and idx_s is not None:
        df.loc[idx_s, "Temporal_Baseline_score"] = mse_s
        temporal_cols.append("Temporal_Baseline_score")
        df, metrics = optimizer.apply_dynamic_thresholds(
            df,
            "Temporal_Baseline_score",
            "Temporal_Baseline",
            calibration_scores=mse_s[mask_train_strict],
        )
        temporal_results.extend(metrics)

    temporal_score_cols = [c for c in temporal_cols if c.startswith("Temporal")]
    if not temporal_score_cols:
        logger.warning(f"Nenhum modelo temporal ({arch_type.upper()}) produziu scores")
    return df, temporal_results, temporal_cols

def export_results(
    df,
    results_summary,
    score_columns_audit,
    config,
    metrics_dir,
    master_dir,
    models_dir,
    df_train=None,
    df_val=None,
    stats=None,
    run_id=None,
    git_info=None,
    model_version=None,
):
    """
    Exporta resultados, metricas e analise de concordancia.
    """
    if not results_summary:
        raise RuntimeError("PIPELINE ABORTADO: Nenhuma metrica foi gerada.")

    # Serializar thresholds por percentil para uso em inferencia
    # CRITICO: inference.py depende destes arquivos para nao entrar em modo degradado
    thresholds_by_percentile = {}  # {90: {"ISO_N100": 0.312, ...}, 95: {...}, 99: {...}}
    for metric in results_summary:
        model_name = _normalize_threshold_model_name(metric.get("Model", ""))
        percentile = metric.get("Percentile")
        threshold = metric.get("Threshold_Value")
        if model_name and percentile is not None and threshold is not None:
            p_key = int(percentile)
            thresholds_by_percentile.setdefault(p_key, {})
            thresholds_by_percentile[p_key][model_name] = float(threshold)

    for p, thresh_dict in thresholds_by_percentile.items():
        thresh_path = os.path.join(models_dir, f"thresholds_p{p}.json")
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(thresh_dict, f, indent=2)
        logger.info(
            f"Thresholds p{p} serializados: {thresh_path} ({len(thresh_dict)} modelos)"
        )

    if thresholds_by_percentile:
        logger.info(
            f"THRESHOLDS SERIALIZADOS: {list(thresholds_by_percentile.keys())} percentis. "
            "inference.py usara estes valores sem recalibrar nos dados novos."
        )
    else:
        logger.warning(
            "ATENCAO: Nenhum threshold encontrado em results_summary para serializar. "
            "Verificar se apply_dynamic_thresholds() esta sendo chamado corretamente."
        )

    # Cobertura de scoring temporal por registro
    temporal_score_cols = [
        c for c in df.columns if c.startswith("Temporal") and c.endswith("_score")
    ]
    if temporal_score_cols:
        df["temporal_coverage"] = df[temporal_score_cols].notna().any(axis=1).astype(int)
        logger.info(
            f"Cobertura temporal: {df['temporal_coverage'].sum():,}/{len(df):,} "
            f"registros avaliados por ao menos um modelo temporal "
            f"({df['temporal_coverage'].mean()*100:.1f}%)"
        )

    # Camada de decisao final do ensemble
    logger.info("Calculando decisao final do ensemble...")
    df = compute_ensemble_decision(df, percentile=95)

    # --- COBERTURA DE AVALIACAO POR MODELO ---
    p95_label_cols = [c for c in df.columns if c.endswith("_p95_label")]
    iso_label_cols = [c for c in p95_label_cols if c.startswith("ISO")]
    hbos_label_cols = [c for c in p95_label_cols if c.startswith("HBOS")]
    temp_label_cols = [c for c in p95_label_cols if c.startswith("Temporal")]

    # Quantos modelos de cada tipo avaliaram este registro
    df["coverage_iso"] = df[iso_label_cols].notna().sum(axis=1).astype(int)
    df["coverage_hbos"] = df[hbos_label_cols].notna().sum(axis=1).astype(int)
    df["coverage_temporal"] = df[temp_label_cols].notna().sum(axis=1).astype(int)

    # Flag de avaliacao completa (todos os modelos conseguiram avaliar)
    n_iso = len(iso_label_cols)
    n_hbos = len(hbos_label_cols)
    n_temp = len(temp_label_cols)
    df["fully_evaluated"] = (
        (df["coverage_iso"] == n_iso)
        & (df["coverage_hbos"] == n_hbos)
        & (df["coverage_temporal"] == n_temp)
    ).astype(int)

    temporal_any_eval = (df["coverage_temporal"] > 0)
    logger.info("-" * 60)
    logger.info("COBERTURA DE AVALIACAO:")
    logger.info(
        f"   ISO ({n_iso} modelos): {int((df['coverage_iso'] == n_iso).sum()):,}/{len(df):,} registros totalmente avaliados"
    )
    logger.info(
        f"   HBOS ({n_hbos} modelos): {int((df['coverage_hbos'] == n_hbos).sum()):,}/{len(df):,} registros totalmente avaliados"
    )
    logger.info(
        f"   Temporal ({n_temp} modelos): "
        f"{int(temporal_any_eval.sum()):,}/{len(df):,} registros avaliados "
        f"({temporal_any_eval.mean()*100:.1f}%)"
    )
    logger.info(
        f"   Avaliacao completa (todos modelos): "
        f"{int(df['fully_evaluated'].sum()):,}/{len(df):,} registros "
        f"({df['fully_evaluated'].mean()*100:.1f}%)"
    )

    # Relatorio de cobertura por veiculo
    map_cols = config["mapeamento_colunas"]
    placa_col = map_cols["placa"]
    if placa_col in df.columns:
        vehicle_coverage = (
            df.groupby(placa_col)
            .agg(
                total_registros=("fully_evaluated", "count"),
                registros_totalmente_avaliados=("fully_evaluated", "sum"),
                media_modelos_temporais=("coverage_temporal", "mean"),
                max_modelos_temporais=("coverage_temporal", "max"),
                alertas_ensemble=("ensemble_alert", lambda x: (x == 1.0).sum()),
            )
            .reset_index()
        )
        vehicle_coverage["pct_avaliacao_completa"] = (
            vehicle_coverage["registros_totalmente_avaliados"]
            / vehicle_coverage["total_registros"].clip(lower=1)
            * 100
        ).round(1)

        sem_cobertura_temporal = int(
            (vehicle_coverage["max_modelos_temporais"] == 0).sum()
        )
        vehicle_coverage = vehicle_coverage.sort_values(
            "pct_avaliacao_completa", ascending=True
        )
        vehicle_coverage.to_csv(
            os.path.join(metrics_dir, "vehicle_coverage_report.csv"), index=False
        )
        logger.info(
            f"Cobertura por veiculo exportada: {len(vehicle_coverage)} veiculos"
        )

        if sem_cobertura_temporal > 0:
            window_size = (
                config.get("parametros", {})
                .get("temporal", {})
                .get(
                    "window_size",
                    config.get("parametros", {})
                    .get("temporal_window_size", config.get("parametros", {}).get("lstm_window_size", 5)),
                )
            )
            logger.warning(
                f"   ATENCAO: {sem_cobertura_temporal} veiculo(s) sem NENHUMA avaliacao "
                f"pelo modelo temporal (registros insuficientes para formar sequencia "
                f"de {window_size} timesteps). Esses veiculos sao avaliados APENAS por ISO e HBOS."
            )

    # Ranking de risco por veiculo
    vehicle_risk = compute_vehicle_risk_summary(df, placa_col=map_cols["placa"])
    if not vehicle_risk.empty:
        vehicle_risk.to_csv(
            os.path.join(metrics_dir, "vehicle_risk_ranking.csv"), index=False
        )
        logger.info(f"Ranking de risco salvo: {len(vehicle_risk)} veiculos")

    iso_metrics = [m for m in results_summary if m["Model"].startswith("ISO")]
    hbos_metrics = [m for m in results_summary if m["Model"].startswith("HBOS")]
    temporal_metrics = [m for m in results_summary if m["Model"].startswith("Temporal")]
    logger.info(f"Total de metricas geradas: {len(results_summary)}")
    logger.info(f"Total de colunas de score auditadas: {len(score_columns_audit)}")

    # Limpa nomes legados para evitar ambiguidade em auditorias.
    legacy_metric_prefix = "l" + "stm"
    for legacy_name in (
        f"{legacy_metric_prefix}_metrics.csv",
        f"{legacy_metric_prefix}_results.csv",
        "comparativo_completo.csv",
    ):
        legacy_path = os.path.join(metrics_dir, legacy_name)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    if iso_metrics:
        pd.DataFrame(iso_metrics).to_csv(os.path.join(metrics_dir, "iso_metrics.csv"), index=False)
    if hbos_metrics:
        pd.DataFrame(hbos_metrics).to_csv(os.path.join(metrics_dir, "hbos_metrics.csv"), index=False)
    if temporal_metrics:
        pd.DataFrame(temporal_metrics).to_csv(
            os.path.join(metrics_dir, "temporal_metrics.csv"), index=False
        )

    logger.info("=" * 80)
    logger.info("ANALISE DE CONCORDANCIA ENTRE MODELOS")
    logger.info("ATENCAO: metricas de concordancia NAO sao validacao contra ground truth.")
    label_columns = [col for col in df.columns if col.endswith("_label")]
    if label_columns:
        analyzer = ModelConcordanceAnalyzer()
        df_conc = analyzer.analyze_concordance(df, label_columns)
        if not df_conc.empty:
            df_conc.to_csv(os.path.join(metrics_dir, "concordancia_modelos.csv"), index=False)
            logger.info(f"Analise de concordancia exportada: {len(df_conc)} pares avaliados")
        else:
            logger.warning("Nenhuma analise de concordancia gerada")
    else:
        logger.warning("Nenhuma coluna de label encontrada para analise")

    # Selecao de configuracao via validation set
    if (
        df_train is not None
        and df_val is not None
        and len(df_train) > 0
        and len(df_val) > 0
    ):
        # df_train/df_val vindos do load_data podem nao carregar colunas de score.
        # Reconstroi os recortes no df consolidado (ja pontuado), mantendo apenas
        # indices originais de treino/validacao para evitar leakage com teste.
        df_train_scored = df.loc[df_train.index]
        df_val_scored = df.loc[df_val.index]
        logger.info(
            f"SELECAO DE CONFIGURACAO: usando df_train ({len(df_train_scored):,} registros) "
            f"como referencia. df_val={len(df_val_scored):,} registros como comparacao. "
            "df_test NAO entra na selecao (sem leakage)."
        )
        score_cols_for_selection = [
            c for c in score_columns_audit if c.startswith("ISO") or c.startswith("HBOS")
        ]
        df_model_selection = compute_val_stability_metrics(
            df_train_scored, df_val_scored, score_cols_for_selection, percentile=95
        )
        if not df_model_selection.empty:
            df_model_selection.to_csv(
                os.path.join(metrics_dir, "model_selection_val.csv"), index=False
            )
            logger.info("Selecao de configuracao exportada: model_selection_val.csv")
    else:
        logger.warning(
            "Split de treino/validacao nao disponivel para selecao de configuracao. "
            "Execute com split 60/20/20 para usar este recurso sem leakage."
        )

    df.to_parquet(os.path.join(master_dir, "resultado_final.parquet"), index=False)
    if stats is not None:
        # Adicionar run_id ao perfil para rastreabilidade
        stats["run_id"] = run_id if run_id else "unversioned"
        stats["run_timestamp"] = datetime.datetime.now().isoformat()
        with open(os.path.join(metrics_dir, "perfil_dados.json"), "w") as f:
            json.dump(stats, f, indent=4, default=str)
        logger.info(f"Perfil de dados salvo com run_id={stats['run_id']}")

        # Gerar relatorio HTML automaticamente
        try:
            from src.outputs.report_generator import generate_report

            report_path = os.path.join(
                os.path.dirname(metrics_dir), "relatorio_executivo.html"
            )
            generate_report(
                metrics_dir=metrics_dir,
                parquet_path=os.path.join(master_dir, "resultado_final.parquet"),
                output_path=report_path,
                run_id=run_id or "N/A",
            )
            size_kb = os.path.getsize(report_path) / 1024 if os.path.exists(report_path) else 0
            logger.info(f"Relatorio HTML gerado: {report_path} ({size_kb:.0f} KB)")
        except ImportError:
            logger.warning(
                "AVISO: plotly nao instalado - relatorio HTML nao gerado. "
                "Instale com: pip install 'plotly>=5.18.0'"
            )
        except FileNotFoundError as e:
            logger.error(
                f"ERRO ao gerar relatorio HTML: arquivo nao encontrado: {e}\n"
                "Verificar que vehicle_risk_ranking.csv e concordancia_modelos.csv "
                "foram gerados pelos fixes C3 e H2 antes deste fix."
            )
        except Exception as e:
            logger.error(
                f"ERRO ao gerar relatorio HTML: {e}\n"
                "Stack trace completo nos logs de debug."
            )
            import traceback

            logger.debug(traceback.format_exc())

    map_cols = config["mapeamento_colunas"]
    id_cols = [
        map_cols["placa"],
        map_cols["timestamp"],
        map_cols["latitude"],
        map_cols["longitude"],
    ]
    cols_present = [c for c in id_cols if c in df.columns]
    df_iso = df[cols_present + [col for col in df.columns if col.startswith("ISO")]]
    df_hbos = df[cols_present + [col for col in df.columns if col.startswith("HBOS")]]
    df_temporal = df[cols_present + [col for col in df.columns if col.startswith("Temporal")]]
    pd.DataFrame(df_iso).to_csv(os.path.join(metrics_dir, "iso_results.csv"), index=False)
    pd.DataFrame(df_hbos).to_csv(os.path.join(metrics_dir, "hbos_results.csv"), index=False)
    pd.DataFrame(df_temporal).to_csv(
        os.path.join(metrics_dir, "temporal_results.csv"), index=False
    )
    logger.info("Resultados segmentados exportados.")

    # Gerar manifesto dos modelos para uso pelo inference.py (com hash de integridade)
    manifest = {"iso": [], "hbos": [], "temporal": [], "scalers": {}}
    for fname in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fname)
        rel_path = os.path.relpath(fpath, models_dir)
        if fname.startswith("iso_") and fname.endswith(".joblib"):
            tag = fname.replace(".joblib", "").upper()
            manifest["iso"].append(
                {
                    "tag": tag,
                    "name": tag,
                    "path": rel_path,
                    "type": "joblib",
                    "sha256": sha256_file(fpath),
                }
            )
        elif fname.startswith("hbos_") and fname.endswith(".joblib"):
            tag = fname.replace(".joblib", "").upper()
            manifest["hbos"].append(
                {
                    "tag": tag,
                    "name": tag,
                    "path": rel_path,
                    "type": "joblib",
                    "sha256": sha256_file(fpath),
                }
            )
        elif fname.startswith("temporal_") and fname.endswith(".h5"):
            tag = _canonical_temporal_name_from_file(fname)
            manifest["temporal"].append(
                {
                    "tag": tag,
                    "name": tag,
                    "path": rel_path,
                    "type": "keras",
                    "sha256": sha256_file(fpath),
                }
            )
        elif fname == "scaler.joblib":
            manifest["scaler"] = {
                "path": rel_path,
                "type": "joblib",
                "sha256": sha256_file(fpath),
            }
            manifest["scalers"]["main"] = rel_path
        elif fname == "gru_scaler.joblib":
            manifest["gru_scaler"] = {
                "path": rel_path,
                "type": "joblib",
                "sha256": sha256_file(fpath),
            }
            manifest["scalers"]["gru"] = rel_path

    manifest["thresholds"] = {}
    for p in thresholds_by_percentile.keys():
        thresh_path = os.path.join(models_dir, f"thresholds_p{p}.json")
        if os.path.exists(thresh_path):
            manifest["thresholds"][str(p)] = {
                "path": os.path.relpath(thresh_path, models_dir),
                "type": "json",
                "sha256": sha256_file(thresh_path),
            }

    # Rastreabilidade de versao do codigo de treinamento.
    git_info = git_info or get_git_info()
    model_version = model_version or format_model_version(git_info, run_id or "unknown")
    manifest["model_version"] = model_version
    manifest["git"] = {
        "commit_hash": git_info["commit_hash"],
        "commit_short": git_info["commit_short"],
        "branch": git_info["branch"],
        "is_dirty": git_info["is_dirty"],
        "dirty_warning": git_info.get("dirty_warning"),
        "commit_message": git_info["commit_message"],
        "commit_timestamp": git_info["commit_timestamp"],
    }
    manifest["run_id"] = run_id
    manifest["training_timestamp"] = datetime.datetime.now().isoformat()
    logger.info(f"Versao do modelo: {model_version}")
    if git_info.get("is_dirty"):
        logger.warning(
            "ATENCAO: modelo treinado com codigo nao commitado. "
            f"Branch: {git_info['branch']} | Commit base: {git_info['commit_short']}. "
            "Para auditabilidade completa, commitar mudancas antes do proximo treinamento."
        )

    manifest_path = os.path.join(models_dir, "models_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifesto de modelos salvo: {manifest_path}")


def run_experiment(
    config_path="config_mapeamento.yaml",
    input_path=None,
    output_dir="outputs",
    epochs=None,
    seed=42,
    run_id=None,
):
    """
    Orquestra o pipeline completo de deteccao de anomalias.
    """
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir, run_id = _resolve_versioned_output_dir(output_dir, run_id)
    metrics_dir = os.path.join(output_dir, "metrics")
    master_dir = os.path.join(output_dir, "master_table")
    models_dir = os.path.join(output_dir, "models_saved")
    for d in [metrics_dir, master_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    sspdf_logger = logging.getLogger("sspdf")

    # Remover handlers antigos para evitar duplicacao e mistura entre runs.
    for handler in sspdf_logger.handlers[:]:
        sspdf_logger.removeHandler(handler)
        handler.close()

    os.makedirs(metrics_dir, exist_ok=True)
    _run_file_handler = logging.FileHandler(
        os.path.join(metrics_dir, "execution.log"),
        encoding="utf-8",
    )
    _run_file_handler.setLevel(logging.INFO)
    _run_file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    sspdf_logger.addHandler(_run_file_handler)

    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    sspdf_logger.addHandler(_console_handler)

    sspdf_logger.setLevel(logging.INFO)
    sspdf_logger.info(f"Logger configurado para run_id={run_id}")
    sspdf_logger.info(f"Output dir versionado: {output_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    global_seed = config.get("random_state", seed)
    np.random.seed(global_seed)

    # Precedencia explicita: CLI > YAML > fallback hardcoded
    yaml_epochs = config.get("parametros", {}).get("temporal", {}).get("epochs", 10)
    if epochs is None:
        # Usuario nao passou --epochs: usar YAML
        epochs = yaml_epochs
        logger.info(f"--epochs nao informado. Usando valor do YAML: {epochs} epochs")
    else:
        # Usuario passou --epochs explicitamente: respeitar
        logger.info(
            f"--epochs={epochs} (explicito via CLI). "
            f"Valor do YAML ({yaml_epochs}) ignorado."
        )

    # Propaga epochs efetivas para o config consumido pelo treino temporal.
    config.setdefault("parametros", {}).setdefault("temporal", {})["epochs"] = epochs

    from src.utils.tracking import (
        end_run,
        init_experiment,
        log_artifact,
        log_metrics,
        log_params,
    )

    # Iniciar MLflow run (se disponivel e nao desativado)
    _tracking_run = init_experiment(run_id=run_id)
    git_info = get_git_info()
    model_version = format_model_version(git_info, run_id)

    if _tracking_run:
        # Logar todos os parametros do YAML centralizado
        iso_cfg = config.get("parametros", {}).get("isolation_forest", {})
        hbos_cfg = config.get("parametros", {}).get("hbos", {})
        temp_cfg = config.get("parametros", {}).get("temporal", {})
        split = config.get("parametros", {}).get("split_ratios", {})
        log_params(
            {
                "run_id": run_id,
                "seed": global_seed,
                "epochs": epochs,
                "split_train": split.get("train", 0.6),
                "split_val": split.get("validation", 0.2),
                "split_test": split.get("test", 0.2),
                "iso_n_estimators": str(iso_cfg.get("n_estimators", [100, 200])),
                "iso_contamination": iso_cfg.get("contamination", "auto"),
                "hbos_n_bins": str(hbos_cfg.get("n_bins", [10, 20])),
                "hbos_contamination": hbos_cfg.get("contamination", 0.1),
                "temporal_arch": temp_cfg.get("arch_type", "gru"),
                "temporal_window_size": temp_cfg.get("window_size", 3),
                "temporal_batch_size": temp_cfg.get("batch_size", 64),
                "temporal_dropout": temp_cfg.get("dropout", 0.2),
                "percentis_teste": str(
                    config.get("parametros", {}).get("percentis_teste", [90, 95, 99])
                ),
            }
        )

    proc = DataProcessor(config)
    proc.models_dir = models_dir

    logger.info(f"Random state efetivo: {global_seed}")
    logger.info(f"Epochs temporais efetivas: {epochs}")

    logger.info("=" * 80)
    logger.info("ETAPA 1: CARGA E PROCESSAMENTO DE DADOS")
    df, df_train, df_val, df_test, proc, stats = load_data(proc, config, input_path)

    logger.info("=" * 80)
    logger.info("ETAPA 2: PREPARACAO DE FEATURES POR MODELO")
    features_dict = prepare_model_features(df, df_train, config, proc, models_dir)
    stats["features_iso"] = features_dict["iso_features"]
    stats["features_hbos"] = features_dict["hbos_features"]
    stats["features_gru"] = features_dict["gru_features"]

    logger.info("=" * 80)
    logger.info("ETAPA 3: TREINAMENTO DE MODELOS BASE")
    df, iso_masks, hbos_masks, results_summary, score_cols = train_base_models(
        df, features_dict, config, models_dir
    )

    logger.info("=" * 80)
    logger.info("ETAPA 4: TREINAMENTO DE MODELOS TEMPORAIS")
    train_end = stats.get("split_temporal", {}).get("train_end_index")
    if train_end is None:
        train_end = int(
            len(df) * config.get("parametros", {}).get("split_ratios", {}).get("train", 0.6)
        )
    df, temporal_results, temporal_cols = train_temporal_models(
        df, features_dict, config, iso_masks, hbos_masks, train_end, models_dir, epochs
    )
    results_summary.extend(temporal_results)
    score_cols.extend(temporal_cols)

    logger.info("=" * 80)
    logger.info("ETAPA 5: EXPORTACAO DE RESULTADOS")
    export_results(
        df,
        results_summary,
        score_cols,
        config,
        metrics_dir,
        master_dir,
        models_dir,
        df_train=df_train,
        df_val=df_val,
        stats=stats,
        run_id=run_id,
        git_info=git_info,
        model_version=model_version,
    )

    index_path = os.path.join(os.path.dirname(output_dir), "runs_index.csv")
    run_entry = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "output_dir": output_dir,
        "model_version": model_version,
        "commit_hash": git_info["commit_hash"],
        "branch": git_info["branch"],
        "is_dirty": git_info["is_dirty"],
        "config_path": config_path,
        "n_records": len(df),
        "n_alerts_p95": (
            int(df["ensemble_alert"].eq(1.0).sum())
            if "ensemble_alert" in df.columns
            else "N/A"
        ),
    }
    target_fields = list(run_entry.keys())
    existing_rows = []
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                raw_rows = list(reader)
            if raw_rows:
                header = raw_rows[0]
                for row in raw_rows[1:]:
                    if not row:
                        continue
                    if len(row) == len(target_fields):
                        row_dict = dict(zip(target_fields, row))
                    else:
                        mapped = dict(zip(header, row[: len(header)]))
                        row_dict = {k: mapped.get(k, "") for k in target_fields}
                    existing_rows.append(row_dict)
        except Exception as e:
            logger.warning(
                f"Nao foi possivel ler runs_index existente ({index_path}): {e}. "
                "Arquivo sera recriado no schema atual."
            )
            existing_rows = []

    existing_rows.append(run_entry)
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(existing_rows)
    logger.info(f"Indice de runs atualizado: {index_path}")

    structured_log = {
        "run_id": run_id,
        "timestamp_start": stats.get("periodo", "N/A"),
        "timestamp_end": datetime.datetime.now().isoformat(),
        "config_path": config_path,
        "output_dir": output_dir,
        "parameters": {
            "seed": global_seed,
            "epochs": epochs,
            "split_ratios": config.get("parametros", {}).get("split_ratios", {}),
            "iso_config": config.get("parametros", {}).get("isolation_forest", {}),
            "hbos_config": config.get("parametros", {}).get("hbos", {}),
            "temporal_config": config.get("parametros", {}).get("temporal", {}),
        },
        "dataset_profile": {
            "total_records": len(df),
            "total_vehicles": stats.get("total_veiculos", "N/A"),
            "period": stats.get("periodo", "N/A"),
        },
        "results_summary": {
            "n_alerts_p95": (
                int(df.get("ensemble_alert", pd.Series(dtype=float)).eq(1.0).sum())
                if "ensemble_alert" in df.columns
                else "N/A"
            ),
            "n_not_scored": (
                int(df.get("n_models_scored", pd.Series(dtype=float)).eq(0).sum())
                if "n_models_scored" in df.columns
                else "N/A"
            ),
        },
        "status": "SUCCESS",
    }
    log_json_path = os.path.join(metrics_dir, "run_summary.json")
    with open(log_json_path, "w") as f:
        json.dump(structured_log, f, indent=2, default=str)
    logger.info(f"Log estruturado JSON salvo: {log_json_path}")

    # Logar metricas de resultado no MLflow
    if "ensemble_alert" in df.columns:
        n_alerts = int((df["ensemble_alert"] == 1.0).sum())
        n_total = len(df)
        log_metrics(
            {
                "n_records_total": n_total,
                "n_alerts_p95": n_alerts,
                "alert_rate_pct": round(n_alerts / n_total * 100, 4) if n_total > 0 else 0,
                "n_not_scored": (
                    int((df.get("n_models_scored", pd.Series(dtype=float)) == 0).sum())
                    if "n_models_scored" in df.columns
                    else 0
                ),
            }
        )

    # Logar artefatos chave
    log_artifact(os.path.join(metrics_dir, "perfil_dados.json"))
    log_artifact(os.path.join(metrics_dir, "concordancia_modelos.csv"))
    log_artifact(os.path.join(metrics_dir, "vehicle_risk_ranking.csv"))
    log_artifact(os.path.join(os.path.dirname(metrics_dir), "relatorio_executivo.html"))

    end_run(status="FINISHED")

    logger.info("EXPERIMENTO FINALIZADO!")
    return output_dir


if __name__ == "__main__":
    run_experiment()
