import logging
import os

import joblib
import numpy as np
import pandas as pd

from config.feature_config import get_features_for_model
from src.models.models_base import BaselineModels
from src.utils.evaluation import ThresholdOptimizer

logger = logging.getLogger("sspdf")


def _align_ra_columns(partitions: list, reference_cols: list) -> list:
    """
    Alinha colunas de One-Hot Encoding (RA_*) entre particoes.
    """
    aligned = []
    for partition in partitions:
        part = partition.copy()

        missing = [col for col in reference_cols if col not in part.columns]
        for col in missing:
            part[col] = 0

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
    proc.features_to_use = train_features
    df_val, _ = proc.feature_engineering(df_val)
    df_test, _ = proc.feature_engineering(df_test)
    proc.features_to_use = train_features

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


def train_base_models(
    df,
    features_dict,
    config,
    models_dir,
    operational_percentile,
):
    """
    Treina Isolation Forest e HBOS em multiplas configuracoes.
    """
    score_columns_audit = []
    results_summary = []
    iso_masks_registry = {}
    hbos_masks_registry = {}
    optimizer = ThresholdOptimizer(config["parametros"]["percentis_teste"])

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
        label_col = f"{tag}_p{operational_percentile}_label"
        if label_col not in df.columns:
            raise RuntimeError(
                f"Label operacional ausente para {tag}: {label_col}. "
                f"Percentis configurados: {config.get('parametros', {}).get('percentis_teste', [])}"
            )
        iso_masks_registry[tag] = df[label_col] == 0

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
        label_col = f"{tag}_p{operational_percentile}_label"
        if label_col not in df.columns:
            raise RuntimeError(
                f"Label operacional ausente para {tag}: {label_col}. "
                f"Percentis configurados: {config.get('parametros', {}).get('percentis_teste', [])}"
            )
        hbos_masks_registry[tag] = df[label_col] == 0

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
    """Train temporal models (GRU/LSTM) in multiple scenarios."""
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

    from src.models.temporal_autoencoder import TemporalAutoencoder

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

            mask_train_union = mask_train_strict & (iso_last_mask | hbos_last_mask)
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
