import datetime
import json
import logging
import os

import pandas as pd

from src.utils.artifact_utils import sha256_file
from src.utils.evaluation import ModelConcordanceAnalyzer
from src.utils.ensemble_decision import (
    compute_ensemble_decision,
    compute_vehicle_risk_summary,
)
from src.utils.git_utils import format_model_version, get_git_info
from src.utils.model_selection import compute_temporal_strategy_validation, compute_val_stability_metrics

logger = logging.getLogger("sspdf")
FEATURE_SCHEMA_FILENAME = "feature_schema.json"


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


def _build_feature_schema_payload(stats):
    """
    Monta payload de schema de features usado no treino.

    Guarda a ordem esperada para alinhamento da inferencia.
    """
    if not isinstance(stats, dict):
        return {}

    payload = {
        "schema_version": 1,
        "main_scaler_features": list(stats.get("main_scaler_features", []) or []),
        "iso_features": list(stats.get("features_iso", []) or []),
        "hbos_features": list(stats.get("features_hbos", []) or []),
        "gru_features": list(stats.get("features_gru", []) or []),
        "generated_at": datetime.datetime.now().isoformat(),
    }

    if not any(
        payload.get(k)
        for k in ("main_scaler_features", "iso_features", "hbos_features", "gru_features")
    ):
        return {}
    return payload


def _resolve_primary_percentile(config):
    params = (config or {}).get("parametros", {})
    raw_percentiles = params.get("percentis_teste", [])
    parsed = []
    for value in raw_percentiles:
        try:
            p = int(value)
        except Exception:
            continue
        if 0 <= p <= 100 and p not in parsed:
            parsed.append(p)
    if not parsed:
        parsed = [95]
    return 95 if 95 in parsed else parsed[0]


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
    operational_percentile=None,
    temporal_selection_audit_df=None,
):
    """
    Exporta resultados, metricas e analise de concordancia.
    """
    if not results_summary:
        raise RuntimeError("PIPELINE ABORTADO: Nenhuma metrica foi gerada.")

    thresholds_by_percentile = {}
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

    decision_percentile = (
        int(operational_percentile)
        if operational_percentile is not None
        else _resolve_primary_percentile(config)
    )

    logger.info("Calculando decisao final do ensemble...")
    df = compute_ensemble_decision(df, percentile=decision_percentile)

    p_label_suffix = f"_p{decision_percentile}_label"
    percentile_label_cols = [c for c in df.columns if c.endswith(p_label_suffix)]
    iso_label_cols = [c for c in percentile_label_cols if c.startswith("ISO")]
    hbos_label_cols = [c for c in percentile_label_cols if c.startswith("HBOS")]
    temp_label_cols = [c for c in percentile_label_cols if c.startswith("Temporal")]

    df["coverage_iso"] = df[iso_label_cols].notna().sum(axis=1).astype(int)
    df["coverage_hbos"] = df[hbos_label_cols].notna().sum(axis=1).astype(int)
    df["coverage_temporal"] = df[temp_label_cols].notna().sum(axis=1).astype(int)

    n_iso = len(iso_label_cols)
    n_hbos = len(hbos_label_cols)
    n_temp = len(temp_label_cols)
    df["fully_evaluated"] = (
        (df["coverage_iso"] == n_iso)
        & (df["coverage_hbos"] == n_hbos)
        & (df["coverage_temporal"] == n_temp)
    ).astype(int)

    temporal_any_eval = df["coverage_temporal"] > 0
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

        sem_cobertura_temporal = int((vehicle_coverage["max_modelos_temporais"] == 0).sum())
        vehicle_coverage = vehicle_coverage.sort_values("pct_avaliacao_completa", ascending=True)
        vehicle_coverage.to_csv(
            os.path.join(metrics_dir, "vehicle_coverage_report.csv"), index=False
        )
        logger.info(f"Cobertura por veiculo exportada: {len(vehicle_coverage)} veiculos")

        if sem_cobertura_temporal > 0:
            window_size = (
                config.get("parametros", {})
                .get("temporal", {})
                .get(
                    "window_size",
                    config.get("parametros", {})
                    .get(
                        "temporal_window_size",
                        config.get("parametros", {}).get("lstm_window_size", 5),
                    ),
                )
            )
            logger.warning(
                f"   ATENCAO: {sem_cobertura_temporal} veiculo(s) sem NENHUMA avaliacao "
                f"pelo modelo temporal (registros insuficientes para formar sequencia "
                f"de {window_size} timesteps). Esses veiculos sao avaliados APENAS por ISO e HBOS."
            )

    vehicle_risk = compute_vehicle_risk_summary(
        df,
        placa_col=map_cols["placa"],
        percentile=decision_percentile,
    )
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
        pd.DataFrame(hbos_metrics).to_csv(
            os.path.join(metrics_dir, "hbos_metrics.csv"),
            index=False,
        )
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
            df_conc.to_csv(
                os.path.join(metrics_dir, "concordancia_modelos.csv"),
                index=False,
            )
            logger.info(f"Analise de concordancia exportada: {len(df_conc)} pares avaliados")
        else:
            logger.warning("Nenhuma analise de concordancia gerada")
    else:
        logger.warning("Nenhuma coluna de label encontrada para analise")

    if (
        df_train is not None
        and df_val is not None
        and len(df_train) > 0
        and len(df_val) > 0
    ):
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
            df_train_scored,
            df_val_scored,
            score_cols_for_selection,
            percentile=decision_percentile,
        )
        temporal_selection = (
            temporal_selection_audit_df.copy()
            if isinstance(temporal_selection_audit_df, pd.DataFrame)
            else compute_temporal_strategy_validation(
                df_train=df_train_scored,
                df_val=df_val_scored,
                score_cols=[c for c in score_columns_audit if c.startswith("Temporal_")],
                percentile=decision_percentile,
            )
        )
        temporal_cfg = str(
            (stats or {}).get(
                "temporal_strategy_configured",
                config.get("parametros", {}).get("temporal", {}).get("temporal_strategy", "all"),
            )
        )
        temporal_effective = str(
            (stats or {}).get(
                "temporal_strategy_effective",
                temporal_cfg if temporal_cfg != "all" else "all",
            )
        )
        temporal_source = str(
            (stats or {}).get(
                "temporal_strategy_selection_source",
                "explicit_config"
                if temporal_cfg != "all"
                else "legacy_all_no_single_selection",
            )
        )
        if not df_model_selection.empty:
            df_model_selection = df_model_selection.copy()
            df_model_selection["temporal_strategy_configured"] = temporal_cfg
            df_model_selection["temporal_strategy_effective"] = temporal_effective
            df_model_selection["temporal_strategy_selection_source"] = temporal_source
            df_model_selection.to_csv(
                os.path.join(metrics_dir, "model_selection_val.csv"), index=False
            )
            logger.info("Selecao de configuracao exportada: model_selection_val.csv")
        if not temporal_selection.empty:
            temporal_selection = temporal_selection.copy()
            temporal_selection["temporal_strategy_configured"] = temporal_cfg
            temporal_selection["temporal_strategy_effective"] = temporal_effective
            temporal_selection["temporal_strategy_selection_source"] = temporal_source
            temporal_selection.to_csv(
                os.path.join(metrics_dir, "temporal_strategy_selection_val.csv"),
                index=False,
            )
            logger.info(
                "Selecao temporal por validacao exportada: temporal_strategy_selection_val.csv"
            )
    else:
        logger.warning(
            "Split de treino/validacao nao disponivel para selecao de configuracao. "
            "Execute com split 60/20/20 para usar este recurso sem leakage."
        )

    report_required = bool(
        config.get("configuracoes_gerais", {}).get("report_required", True)
    )
    report_status = "not_attempted"
    report_error = None
    report_path = os.path.join(os.path.dirname(metrics_dir), "relatorio_executivo.html")

    df.to_parquet(os.path.join(master_dir, "resultado_final.parquet"), index=False)
    if stats is not None:
        stats["run_id"] = run_id if run_id else "unversioned"
        stats["run_timestamp"] = datetime.datetime.now().isoformat()
        stats["operational_percentile"] = int(decision_percentile)
        with open(os.path.join(metrics_dir, "perfil_dados.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, default=str)
        logger.info(f"Perfil de dados salvo com run_id={stats['run_id']}")

        try:
            from src.outputs.report_generator import generate_report

            generate_report(
                metrics_dir=metrics_dir,
                parquet_path=os.path.join(master_dir, "resultado_final.parquet"),
                output_path=report_path,
                run_id=run_id or "N/A",
                operational_percentile=decision_percentile,
            )
            size_kb = os.path.getsize(report_path) / 1024 if os.path.exists(report_path) else 0
            logger.info(f"Relatorio HTML gerado: {report_path} ({size_kb:.0f} KB)")
            report_status = "success"
        except ImportError:
            report_status = "failed"
            report_error = (
                "plotly nao instalado - relatorio HTML nao gerado. "
                "Instale com: pip install 'plotly>=5.18.0'"
            )
            logger.warning(f"AVISO: {report_error}")
        except FileNotFoundError as e:
            report_status = "failed"
            report_error = (
                f"arquivo nao encontrado: {e}. "
                "Verificar que vehicle_risk_ranking.csv e concordancia_modelos.csv "
                "foram gerados."
            )
            logger.error(f"ERRO ao gerar relatorio HTML: {report_error}")
        except Exception as e:
            report_status = "failed"
            report_error = str(e)
            logger.error(f"ERRO ao gerar relatorio HTML: {e}")
            import traceback

            logger.debug(traceback.format_exc())
    else:
        report_status = "failed"
        report_error = "stats indisponivel para gerar relatorio."
        logger.error("Relatorio HTML nao foi gerado porque stats estao indisponiveis.")

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

    feature_schema_path = None
    feature_schema_payload = _build_feature_schema_payload(stats)
    if feature_schema_payload:
        feature_schema_path = os.path.join(models_dir, FEATURE_SCHEMA_FILENAME)
        with open(feature_schema_path, "w", encoding="utf-8") as f:
            json.dump(feature_schema_payload, f, indent=2)
        logger.info(
            "Schema de features do treino salvo para inferencia: "
            f"{feature_schema_path}"
        )

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

    if feature_schema_path and os.path.exists(feature_schema_path):
        manifest["feature_schema"] = {
            "path": os.path.relpath(feature_schema_path, models_dir),
            "type": "json",
            "sha256": sha256_file(feature_schema_path),
        }

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
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifesto de modelos salvo: {manifest_path}")

    return {
        "report_required": report_required,
        "report_status": report_status,
        "report_error": report_error,
        "report_path": report_path,
    }
