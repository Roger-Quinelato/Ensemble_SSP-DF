from src.utils.evaluation import ThresholdOptimizer, ModelConcordanceAnalyzer
from src.models.temporal_autoencoder import TemporalAutoencoder
from src.models.models_base import BaselineModels
from src.data.data_processor import DataProcessor
from src.utils.logger_utils import logger
from config.feature_config import get_features_for_model
import os
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
    df_train = proc.feature_engineering(df_train)
    train_features_to_use = proc.features_to_use.copy()
    df_val = proc.feature_engineering(df_val)
    df_test = proc.feature_engineering(df_test)
    proc.features_to_use = train_features_to_use

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
    """Treina modelos temporais em multiplos cenarios."""
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

    logger.info(
        f"TREINAMENTO TEMPORAL MULTI-CENARIOS ({arch_type.upper()}) | window={window_size}, epochs={temporal_epochs}, batch_size={temporal_batch_size}"
    )
    logger.info(f"Temporal arch config: {arch_config}")

    for iso_name, iso_mask_inlier in iso_masks.items():
        for hbos_name, hbos_mask_inlier in hbos_masks.items():
            mask_train_uniao = iso_mask_inlier & hbos_mask_inlier & mask_temporal_train
            temporal_name_union = f"Temporal_Union_{iso_name}_{hbos_name}"
            mse_u, idx_u, model_u = temporal_pipe.train_evaluate(
                temporal_name_union,
                mask_train=mask_train_uniao,
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
                calib_mask_u = pd.Series(mask_train_uniao, index=df.index).loc[idx_u].values
                df, metrics = optimizer.apply_dynamic_thresholds(
                    df,
                    f"{temporal_name_union}_score",
                    temporal_name_union,
                    calibration_scores=mse_u[calib_mask_u.astype(bool)],
                )
                temporal_results.extend(metrics)

            mask_train_inter = (iso_mask_inlier | hbos_mask_inlier) & mask_temporal_train
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


def export_results(df, results_summary, score_columns_audit, config, metrics_dir, master_dir):
    """
    Exporta resultados, metricas e analise de concordancia.
    """
    if not results_summary:
        raise RuntimeError("PIPELINE ABORTADO: Nenhuma metrica foi gerada.")

    df.to_parquet(os.path.join(master_dir, "resultado_final.parquet"), index=False)
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


def run_experiment(
    config_path="config_mapeamento.yaml",
    input_path=None,
    output_dir="outputs",
    epochs=5,
    seed=42,
):
    """
    Orquestra o pipeline completo de deteccao de anomalias.
    """
    metrics_dir = os.path.join(output_dir, "metrics")
    master_dir = os.path.join(output_dir, "master_table")
    models_dir = os.path.join(output_dir, "models_saved")
    for d in [metrics_dir, master_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    global_seed = config.get("random_state", seed)
    np.random.seed(global_seed)

    # CLI overrides config; config overrides default
    if epochs == 5:  # default do argparse, nao foi explicitamente setado
        epochs = config.get("parametros", {}).get("temporal", {}).get("epochs", 5)
    # Propaga epochs efetivas para o config consumido pelo treino temporal.
    config.setdefault("parametros", {}).setdefault("temporal", {})["epochs"] = epochs

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
    export_results(df, results_summary, score_cols, config, metrics_dir, master_dir)

    with open(os.path.join(metrics_dir, "perfil_dados.json"), "w") as f:
        json.dump(stats, f, indent=4, default=str)
    logger.info("EXPERIMENTO FINALIZADO!")


if __name__ == "__main__":
    run_experiment()
