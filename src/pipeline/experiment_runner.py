from src.utils.evaluation import ThresholdOptimizer, GroundTruthComparator
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


def run_experiment(
    config_path="config_mapeamento.yaml",
    input_path=None,
    output_dir="outputs",
    epochs=5,
    seed=42,
):
    """
    Executa o pipeline completo de deteccao de anomalias (Experimento End-to-End).
    """
    np.random.seed(seed)

    metrics_dir = os.path.join(output_dir, "metrics")
    master_dir = os.path.join(output_dir, "master_table")
    models_dir = os.path.join(output_dir, "models_saved")

    logger.info("📂 Inicializando diretórios...")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(master_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    logger.info("⚙️ Carregando configurações...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    map_cols = config["mapeamento_colunas"]

    proc = DataProcessor(config)
    logger.info("=" * 80)
    logger.info("📂 Carregando e Processando Dados...")

    if input_path is None:
        input_path = "data/input/amostra_ssp.csv"
        if not os.path.exists(input_path):
            input_path = "data/input/amostra_ssp.parquet"

    df = proc.load_and_standardize(input_path)
    df = proc.feature_engineering(df)

    dias = df[map_cols["timestamp"]].dt.date.nunique()
    meses = df[map_cols["timestamp"]].dt.to_period("M").nunique()

    grouped = df.groupby([map_cols["latitude"], map_cols["longitude"]])[
        map_cols["placa"]
    ].count()

    if not grouped.empty:
        local_mais_fluxo = grouped.sort_values(ascending=False).reset_index().iloc[0]
        total_veiculos = df[map_cols["placa"]].nunique()
        periodo_min = df[map_cols["timestamp"]].min()
        periodo_max = df[map_cols["timestamp"]].max()
        if "velocidade_kmh" in df.columns:
            vel_media = float(df["velocidade_kmh"].mean())
        else:
            vel_media = 0.0
        dias_analise = int(dias)
        meses_analise = int(meses)
        local_mais_fluxo_latitude = float(local_mais_fluxo[map_cols["latitude"]])
        local_mais_fluxo_longitude = float(local_mais_fluxo[map_cols["longitude"]])
        fluxo_veiculos_local = int(local_mais_fluxo[map_cols["placa"]])
        stats = {
            "total_veiculos": int(total_veiculos),
            "periodo": f"{periodo_min} a {periodo_max}",
            "vel_media": vel_media,
            "dias_analise": dias_analise,
            "meses_analise": meses_analise,
            "local_mais_fluxo_latitude": local_mais_fluxo_latitude,
            "local_mais_fluxo_longitude": local_mais_fluxo_longitude,
            "fluxo_veiculos_local": fluxo_veiculos_local,
        }
    else:
        stats = {"info": "Base vazia ou sem agrupamento possível"}

    # ==========================================================================
    # SPLIT TEMPORAL (OUT-OF-TIME VALIDATION)
    # ==========================================================================
    df = df.sort_values(map_cols["timestamp"]).reset_index(drop=True)

    train_ratio = 0.7
    cutoff_idx = int(len(df) * train_ratio)
    cutoff_timestamp = df[map_cols["timestamp"]].iloc[cutoff_idx]

    df_train = df.iloc[:cutoff_idx].copy()
    df_test = df.iloc[cutoff_idx:].copy()

    logger.info("=" * 80)
    logger.info("📊 SPLIT TEMPORAL")
    logger.info(f"   Total registros: {len(df):,}")
    logger.info(f"   Treino: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    logger.info(f"   Teste:  {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
    logger.info(f"   Corte temporal: {cutoff_timestamp}")
    logger.info("=" * 80)
    logger.info(
        "Split temporal aplicado | total=%s treino=%s teste=%s cutoff=%s",
        len(df),
        len(df_train),
        len(df_test),
        cutoff_timestamp,
    )

    # ==========================================================================
    # NORMALIZACAO (StandardScaler - fit no treino, transform no teste)
    # ==========================================================================
    logger.info("=" * 80)
    logger.info("📏 NORMALIZANDO FEATURES")

    scaler_path = os.path.join(models_dir, "scaler.joblib")
    df_train = proc.fit_scaler(df_train, output_path=scaler_path)
    df_test = proc.transform_scaler(df_test, scaler_path=scaler_path)

    df = pd.concat([df_train, df_test], axis=0).sort_index()

    x_train = df_train[proc.features_to_use].values
    x_all = df[proc.features_to_use].values

    logger.info(f"✅ Features normalizadas: {len(proc.features_to_use)} features")
    logger.info("=" * 80)

    # Features especificas por modelo
    iso_features = get_features_for_model("isolation_forest", df.columns.tolist())
    hbos_features = get_features_for_model("hbos", df.columns.tolist())
    gru_features = get_features_for_model("gru", df.columns.tolist())

    x_iso_train = df_train[iso_features].values
    x_iso_all = df[iso_features].values
    x_hbos_train = df_train[hbos_features].values
    x_hbos_all = df[hbos_features].values
    x_gru_all = df[gru_features].values

    stats["features_iso"] = iso_features
    stats["features_hbos"] = hbos_features
    stats["features_gru"] = gru_features
    stats["split_temporal"] = {
        "cutoff_timestamp": str(cutoff_timestamp),
        "train_size": len(df_train),
        "test_size": len(df_test),
        "train_ratio": train_ratio,
    }
    with open(os.path.join(metrics_dir, "perfil_dados.json"), "w") as f:
        json.dump(stats, f, indent=4, default=str)

    df["hora"] = df[map_cols["timestamp"]].dt.hour
    df["dia_semana"] = df[map_cols["timestamp"]].dt.day_name()

    score_columns_audit = []
    results_summary = []
    iso_masks_registry = {}
    hbos_masks_registry = {}

    models_base_iso = BaselineModels(x_iso_train)
    optimizer = ThresholdOptimizer(config["parametros"]["percentis_teste"])

    # ==========================================================================
    # 2. TREINAMENTO DOS MODELOS BASE
    # ==========================================================================
    logger.info("-" * 40)
    logger.info("🌲 TREINANDO VARIAÇÕES ISO FOREST")
    for n_est in [100, 200]:
        tag = f"ISO_n{n_est}"
        logger.info(f"   ↳ {tag}...")
        _, _, model = models_base_iso.train_iso(n_estimators=n_est)

        joblib.dump(model, os.path.join(models_dir, f"iso_n{n_est}.joblib"))

        scores_all = model.score_samples(x_iso_all)
        df[f"{tag}_score"] = -scores_all
        score_columns_audit.append(f"{tag}_score")

        scores_train_for_thresh = -model.score_samples(x_iso_train)
        df, metrics = optimizer.apply_dynamic_thresholds(
            df, f"{tag}_score", tag, calibration_scores=scores_train_for_thresh
        )
        results_summary.extend(metrics)
        iso_masks_registry[tag] = df[f"{tag}_p95_label"] == 0

    logger.info("-" * 40)
    logger.info("📊 TREINANDO VARIAÇÕES HBOS")
    models_base_hbos = BaselineModels(x_hbos_train)
    for n_bins in [10, 20]:
        tag = f"HBOS_bins{n_bins}"
        logger.info(f"   ↳ {tag}...")
        _, _, model = models_base_hbos.train_hbos(n_bins=n_bins)
        joblib.dump(model, os.path.join(models_dir, f"hbos_bins{n_bins}.joblib"))

        scores_all = model.decision_function(x_hbos_all)
        df[f"{tag}_score"] = scores_all
        score_columns_audit.append(f"{tag}_score")

        scores_train_for_thresh = model.decision_function(x_hbos_train)
        df, metrics = optimizer.apply_dynamic_thresholds(
            df, f"{tag}_score", tag, calibration_scores=scores_train_for_thresh
        )
        results_summary.extend(metrics)
        hbos_masks_registry[tag] = df[f"{tag}_p95_label"] == 0

    # ==========================================================================
    # GUARD DE INTEGRIDADE
    # ==========================================================================
    if not iso_masks_registry:
        raise RuntimeError(
            "🚨 PIPELINE ABORTADO: Nenhum modelo Isolation Forest foi treinado. "
            "Verifique os dados de entrada e as features."
        )

    if not hbos_masks_registry:
        raise RuntimeError(
            "🚨 PIPELINE ABORTADO: Nenhum modelo HBOS foi treinado. "
            "Verifique as dependências (pyod) e os dados de entrada."
        )

    n_cenarios_esperados = len(iso_masks_registry) * len(hbos_masks_registry) * 2 + 1
    logger.info(
        f"✅ Guards OK: {len(iso_masks_registry)} ISO × {len(hbos_masks_registry)} HBOS"
    )
    logger.info(f"📊 Cenários LSTM esperados: {n_cenarios_esperados}")

    # ==========================================================================
    # 3. COMBINATORIAL GROUND TRUTH & LSTM LOOP
    # ==========================================================================
    logger.info("=" * 80)
    logger.info("🧠 TREINAMENTO LSTM MULTI-CENÁRIOS")
    logger.info("=" * 80)

    gap_seconds = config.get("configuracoes_gerais", {}).get("gap_segmentation_seconds", 300)
    mask_temporal_train = pd.Series(df.index < cutoff_idx, index=df.index)

    lstm_pipe = TemporalAutoencoder(
        X_data=x_gru_all,
        vehicle_ids=df[map_cols["placa"]].values,
        timestamps=df[map_cols["timestamp"]].values,
        original_indices=df.index.values,
        window_size=config["parametros"]["lstm_window_size"],
        max_gap_seconds=gap_seconds,
        arch_type="gru",
    )

    for iso_name, iso_mask_inlier in iso_masks_registry.items():
        for hbos_name, hbos_mask_inlier in hbos_masks_registry.items():
            gt_base_name = f"GT_({iso_name}+{hbos_name})"
            logger.info(f"⚡ Cenário: {gt_base_name}")

            mask_train_uniao = iso_mask_inlier & hbos_mask_inlier & mask_temporal_train
            lstm_name_uniao = f"LSTM_Uniao_{iso_name}_{hbos_name}"
            logger.info(f"   ↳ Treinando {lstm_name_uniao}...")

            mse_uniao, idx_uniao, model_uniao = lstm_pipe.train_evaluate(
                lstm_name_uniao, mask_train=mask_train_uniao, epochs=epochs
            )

            if model_uniao is not None:
                model_uniao.save(
                    os.path.join(models_dir, f"lstm_uniao_{iso_name}_{hbos_name}.h5")
                )

            if mse_uniao is not None and idx_uniao is not None:
                df.loc[idx_uniao, f"{lstm_name_uniao}_score"] = mse_uniao
                score_columns_audit.append(f"{lstm_name_uniao}_score")
                calibration_mask_uniao = pd.Series(mask_train_uniao, index=df.index).loc[
                    idx_uniao
                ].values.astype(bool)
                calibration_scores_uniao = mse_uniao[calibration_mask_uniao]
                df, metrics = optimizer.apply_dynamic_thresholds(
                    df,
                    f"{lstm_name_uniao}_score",
                    lstm_name_uniao,
                    calibration_scores=calibration_scores_uniao,
                )
                results_summary.extend(metrics)

            mask_train_inter = (
                (iso_mask_inlier | hbos_mask_inlier) & mask_temporal_train
            )
            lstm_name_inter = f"LSTM_Inter_{iso_name}_{hbos_name}"
            logger.info(f"   ↳ Treinando {lstm_name_inter}...")

            mse_inter, idx_inter, model_inter = lstm_pipe.train_evaluate(
                lstm_name_inter, mask_train=mask_train_inter, epochs=epochs
            )

            if model_inter is not None:
                model_inter.save(
                    os.path.join(models_dir, f"lstm_inter_{iso_name}_{hbos_name}.h5")
                )

            if mse_inter is not None and idx_inter is not None:
                df.loc[idx_inter, f"{lstm_name_inter}_score"] = mse_inter
                score_columns_audit.append(f"{lstm_name_inter}_score")
                calibration_mask_inter = pd.Series(mask_train_inter, index=df.index).loc[
                    idx_inter
                ].values.astype(bool)
                calibration_scores_inter = mse_inter[calibration_mask_inter]
                df, metrics = optimizer.apply_dynamic_thresholds(
                    df,
                    f"{lstm_name_inter}_score",
                    lstm_name_inter,
                    calibration_scores=calibration_scores_inter,
                )
                results_summary.extend(metrics)

    logger.info("⚡ Cenário: Controle (Sem GT)")
    mse_sujo, idx_sujo, model_sujo = lstm_pipe.train_evaluate(
        "LSTM_Sujo", mask_train=mask_temporal_train, epochs=epochs
    )

    if model_sujo is not None:
        model_sujo.save(os.path.join(models_dir, "lstm_sujo.h5"))

    if mse_sujo is not None and idx_sujo is not None:
        df.loc[idx_sujo, "LSTM_Sujo_score"] = mse_sujo
        score_columns_audit.append("LSTM_Sujo_score")
        calibration_mask_sujo = mask_temporal_train.loc[idx_sujo].values.astype(bool)
        calibration_scores_sujo = mse_sujo[calibration_mask_sujo]
        df, metrics = optimizer.apply_dynamic_thresholds(
            df,
            "LSTM_Sujo_score",
            "LSTM_Sujo",
            calibration_scores=calibration_scores_sujo,
        )
        results_summary.extend(metrics)

    lstm_score_cols = [c for c in score_columns_audit if "LSTM" in c]
    if not lstm_score_cols:
        logger.warning("🚨 AVISO: Nenhum modelo LSTM/GRU produziu scores.")
        logger.warning("Possíveis causas:")
        logger.warning("- Dados insuficientes para gerar sequências temporais")
        logger.warning("- Gap temporal muito restritivo")
        logger.warning("- Window size maior que registros por veículo")

    # ==========================================================================
    # 4. EXPORTACAO
    # ==========================================================================
    logger.info("=" * 80)
    logger.info("💾 SALVANDO RESULTADOS")

    df.to_parquet(os.path.join(master_dir, "resultado_final.parquet"), index=False)

    iso_metrics = [m for m in results_summary if m["Model"].startswith("ISO")]
    hbos_metrics = [m for m in results_summary if m["Model"].startswith("HBOS")]
    lstm_metrics = [m for m in results_summary if m["Model"].startswith("LSTM")]

    if not results_summary:
        raise RuntimeError(
            "🚨 PIPELINE ABORTADO: Nenhuma métrica foi gerada. "
            "Verifique se os modelos treinaram corretamente."
        )

    logger.info(f"📊 Total de métricas geradas: {len(results_summary)}")

    if iso_metrics:
        pd.DataFrame(iso_metrics).to_csv(
            os.path.join(metrics_dir, "iso_metrics.csv"), index=False
        )
    if hbos_metrics:
        pd.DataFrame(hbos_metrics).to_csv(
            os.path.join(metrics_dir, "hbos_metrics.csv"), index=False
        )
    if lstm_metrics:
        pd.DataFrame(lstm_metrics).to_csv(
            os.path.join(metrics_dir, "lstm_metrics.csv"), index=False
        )

    logger.info("✅ Métricas exportadas.")

    # ==========================================================================
    # 5. COMPARATIVO DE MODELOS (Ground Truth Comparator)
    # ==========================================================================
    logger.info("=" * 80)
    logger.info("📊 COMPARATIVO DE MODELOS")

    label_columns = [col for col in df.columns if col.endswith("_label")]

    if label_columns:
        gt_candidates = [
            col for col in label_columns if col.startswith("ISO") and "p95" in col
        ]

        if gt_candidates:
            gt_col = sorted(gt_candidates)[-1]
            other_labels = [col for col in label_columns if col != gt_col]

            comparator = GroundTruthComparator()
            df_comparativo = comparator.compare_all_models(df, gt_col, other_labels)

            if not df_comparativo.empty:
                df_comparativo.to_csv(
                    os.path.join(metrics_dir, "comparativo_completo.csv"), index=False
                )
                logger.info(
                    f"✅ Comparativo exportado: {len(df_comparativo)} modelos avaliados"
                )
                logger.info(f"📌 Ground Truth usado: {gt_col}")
                logger.info("\n%s", df_comparativo.to_string(index=False))
            else:
                logger.warning("⚠️ Nenhuma métrica comparativa gerada")
        else:
            logger.warning("⚠️ Nenhum label ISO p95 encontrado para usar como GT")
    else:
        logger.warning("⚠️ Nenhuma coluna de label encontrada para comparação")

    id_cols = [
        map_cols["placa"],
        map_cols["timestamp"],
        map_cols["latitude"],
        map_cols["longitude"],
    ]
    cols_present = [c for c in id_cols if c in df.columns]

    df_iso = df[cols_present + [col for col in df.columns if col.startswith("ISO")]]
    df_hbos = df[cols_present + [col for col in df.columns if col.startswith("HBOS")]]
    df_lstm = df[cols_present + [col for col in df.columns if col.startswith("LSTM")]]

    def safe_save(dframe, fname):
        pd.DataFrame(dframe).to_csv(fname, index=False)

    safe_save(df_iso, os.path.join(metrics_dir, "iso_results.csv"))
    safe_save(df_hbos, os.path.join(metrics_dir, "hbos_results.csv"))
    safe_save(df_lstm, os.path.join(metrics_dir, "lstm_results.csv"))

    logger.info("✅ Resultados segmentados exportados.")
    logger.info("✅ EXPERIMENTO FINALIZADO!")


if __name__ == "__main__":
    run_experiment()
