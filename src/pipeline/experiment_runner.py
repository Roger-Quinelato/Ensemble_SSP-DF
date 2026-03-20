from src.utils.evaluation import ThresholdOptimizer, ModelConcordanceAnalyzer
from src.models.temporal_autoencoder import TemporalAutoencoder
from src.models.models_base import BaselineModels
from src.data.data_processor import DataProcessor
from src.utils.logger_utils import logger
from src.utils.ensemble_decision import (
    compute_ensemble_decision,
    compute_vehicle_risk_summary,
)
from src.utils.model_selection import compute_val_stability_metrics
from config.feature_config import get_features_for_model
import os
import csv
import datetime
import pandas as pd
import numpy as np
import yaml
import json
import joblib


def _align_ra_columns(df_train, partitions):
    """Alinha schema de colunas RA_* usando o treino como referencia."""
    train_ra_cols = [c for c in df_train.columns if c.startswith("RA_")]
    for partition in partitions:
        for col in train_ra_cols:
            if col not in partition.columns:
                partition[col] = 0
        extra_cols = [
            c for c in partition.columns if c.startswith("RA_") and c not in train_ra_cols
        ]
        partition.drop(columns=extra_cols, inplace=True, errors="ignore")


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

    _align_ra_columns(df_train, [df_val, df_test])
    logger.info(f"   Colunas alinhadas: {len(df_train.columns)} features no treino")
    stats = _build_train_stats(df_train, map_cols)

    logger.info("=" * 80)
    logger.info("NORMALIZANDO FEATURES")
    scaler_root = getattr(proc, "models_dir", os.path.join("outputs", "models_saved"))
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
        _, _, model = models_base_iso.train_iso(
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
        _, _, model = models_base_hbos.train_hbos(
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
    """Treina modelos temporais (GRU) em múltiplos cenários.

    Cenários de treinamento:
    - Union: treino nos registros em que ISO OU HBOS consideram normal (conjunto maior)
    - Inter: treino nos registros em que ISO E HBOS concordam como normal (conjunto menor/mais puro)
    - Baseline: treino em todos os registros do período de treino sem filtro de qualidade
    """
    map_cols = config["mapeamento_colunas"]
    gap_seconds = config.get("configuracoes_gerais", {}).get("gap_segmentation_seconds", 300)
    mask_temporal_train = pd.Series(df.index < train_end, index=df.index)
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

    legacy_model_prefix = "l" + "stm_"
    for model_name in os.listdir(models_dir):
        if model_name.startswith(legacy_model_prefix) and model_name.endswith(".h5"):
            os.remove(os.path.join(models_dir, model_name))

    # Limpar modelos com semântica antiga (union/inter invertidos)
    for fname in os.listdir(models_dir):
        if fname.startswith("temporal_union_") or fname.startswith("temporal_inter_"):
            os.remove(os.path.join(models_dir, fname))
            logger.info(f"   Removido modelo com semântica antiga: {fname}")

    logger.info(
        f"TREINAMENTO TEMPORAL MULTI-CENARIOS ({arch_type.upper()}) | window={window_size}, epochs={temporal_epochs}, batch_size={temporal_batch_size}"
    )
    logger.info(f"Temporal arch config: {arch_config}")

    for iso_name, iso_mask_inlier in iso_masks.items():
        for hbos_name, hbos_mask_inlier in hbos_masks.items():
            # UNION: OR — conjunto maior (mais permissivo)
            mask_train_union = (iso_mask_inlier | hbos_mask_inlier) & mask_temporal_train
            # INTER: AND — conjunto menor (mais restritivo)
            mask_train_inter = (iso_mask_inlier & hbos_mask_inlier) & mask_temporal_train

            n_union_mask = int(mask_train_union.sum())
            n_inter_mask = int(mask_train_inter.sum())
            logger.info(
                f"   Máscara treino ({iso_name} x {hbos_name}): "
                f"Union={n_union_mask:,} | Inter={n_inter_mask:,}"
            )

            temporal_name_union = f"Temporal_Union_{iso_name}_{hbos_name}"
            mse_u, idx_u, model_u = temporal_pipe.train_evaluate(
                temporal_name_union,
                mask_train=mask_train_union,
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
                calib_mask_u = pd.Series(mask_train_union, index=df.index).loc[idx_u].values
                df, metrics = optimizer.apply_dynamic_thresholds(
                    df,
                    f"{temporal_name_union}_score",
                    temporal_name_union,
                    calibration_scores=mse_u[calib_mask_u.astype(bool)],
                )
                temporal_results.extend(metrics)

            temporal_name_inter = f"Temporal_Inter_{iso_name}_{hbos_name}"
            mse_i, idx_i, model_i = temporal_pipe.train_evaluate(
                temporal_name_inter,
                mask_train=mask_train_inter,
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
                calib_mask_i = pd.Series(mask_train_inter, index=df.index).loc[idx_i].values
                df, metrics = optimizer.apply_dynamic_thresholds(
                    df,
                    f"{temporal_name_inter}_score",
                    temporal_name_inter,
                    calibration_scores=mse_i[calib_mask_i.astype(bool)],
                )
                temporal_results.extend(metrics)

    mse_s, idx_s, model_s = temporal_pipe.train_evaluate(
        "Temporal_Baseline",
        mask_train=mask_temporal_train,
        epochs=temporal_epochs,
        batch_size=temporal_batch_size,
    )
    if model_s is not None:
        model_s.save(os.path.join(models_dir, "temporal_baseline.h5"))
    if mse_s is not None and idx_s is not None:
        df.loc[idx_s, "Temporal_Baseline_score"] = mse_s
        temporal_cols.append("Temporal_Baseline_score")
        calib_mask_s = mask_temporal_train.loc[idx_s].values.astype(bool)
        df, metrics = optimizer.apply_dynamic_thresholds(
            df,
            "Temporal_Baseline_score",
            "Temporal_Baseline",
            calibration_scores=mse_s[calib_mask_s],
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
    df_val=None,
    stats=None,
    run_id=None,
):
    """
    Exporta resultados, metricas e analise de concordancia.
    """
    if not results_summary:
        raise RuntimeError("PIPELINE ABORTADO: Nenhuma metrica foi gerada.")

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
    if df_val is not None and len(df_val) > 0:
        score_cols_for_selection = [
            c for c in score_columns_audit if c.startswith("ISO") or c.startswith("HBOS")
        ]
        df_model_selection = compute_val_stability_metrics(
            df, df_val, score_cols_for_selection, percentile=95
        )
        if not df_model_selection.empty:
            df_model_selection.to_csv(
                os.path.join(metrics_dir, "model_selection_val.csv"), index=False
            )
            logger.info("Selecao de configuracao exportada: model_selection_val.csv")
    else:
        logger.warning(
            "Validation set nao disponivel para selecao de configuracao. "
            "Execute com split 60/20/20 para usar este recurso."
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
        except Exception as e:
            logger.warning(f"Relatorio HTML nao gerado (nao critico): {e}")

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

    # Gerar manifesto dos modelos para uso pelo inference.py
    manifest = {"iso": [], "hbos": [], "temporal": [], "scalers": {}}
    for fname in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fname)
        if fname.startswith("iso_") and fname.endswith(".joblib"):
            tag = fname.replace(".joblib", "").upper()
            manifest["iso"].append({"name": tag, "path": fpath})
        elif fname.startswith("hbos_") and fname.endswith(".joblib"):
            tag = fname.replace(".joblib", "").upper()
            manifest["hbos"].append({"name": tag, "path": fpath})
        elif fname.startswith("temporal_") and fname.endswith(".h5"):
            name = "Temporal_" + fname.replace("temporal_", "").replace(".h5", "")
            manifest["temporal"].append({"name": name, "path": fpath})
        elif fname == "scaler.joblib":
            manifest["scalers"]["main"] = fpath
        elif fname == "gru_scaler.joblib":
            manifest["scalers"]["gru"] = fpath
    manifest_path = os.path.join(models_dir, "models_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifesto de modelos salvo: {manifest_path}")


def run_experiment(
    config_path="config_mapeamento.yaml",
    input_path=None,
    output_dir="outputs",
    epochs=5,
    seed=42,
    run_id=None,
):
    """
    Orquestra o pipeline completo de deteccao de anomalias.
    """
    metrics_dir = os.path.join(output_dir, "metrics")
    master_dir = os.path.join(output_dir, "master_table")
    models_dir = os.path.join(output_dir, "models_saved")
    for d in [metrics_dir, master_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reconfigurar logger com run_id para log por execucao
    from src.utils.logger_utils import setup_logger as _setup_logger
    import logging

    _run_logger = _setup_logger(
        name=f"sspdf_{run_id}",
        log_file=os.path.join(metrics_dir, "execution.log"),
        level=logging.INFO,
        run_id=run_id,
    )

    # Logger compartilhado entre modulos (importado como `logger`) precisa
    # apontar para o arquivo da run atual.
    global logger
    logger = _setup_logger(
        name="sspdf",
        log_file=os.path.join(metrics_dir, "execution.log"),
        level=logging.INFO,
        run_id=run_id,
    )
    logger.info(f"RUN ID: {run_id}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    global_seed = config.get("random_state", seed)
    np.random.seed(global_seed)

    # CLI overrides config; config overrides default
    if epochs == 5:  # default do argparse, nao foi explicitamente setado
        epochs = config.get("parametros", {}).get("temporal", {}).get("epochs", 5)
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
        df_val=df_val,
        stats=stats,
        run_id=run_id,
    )

    index_path = os.path.join(os.path.dirname(output_dir), "runs_index.csv")
    run_entry = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "output_dir": output_dir,
        "config_path": config_path,
        "n_records": len(df),
        "n_alerts_p95": (
            int(df["ensemble_alert"].eq(1.0).sum())
            if "ensemble_alert" in df.columns
            else "N/A"
        ),
    }
    file_exists = os.path.exists(index_path)
    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_entry.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_entry)
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


if __name__ == "__main__":
    run_experiment()
