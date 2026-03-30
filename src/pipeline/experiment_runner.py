import logging

from src.utils.evaluation import ThresholdOptimizer, ModelConcordanceAnalyzer
from src.models.models_base import BaselineModels
from src.data.data_processor import DataProcessor
from src.utils.ensemble_decision import (
    compute_ensemble_decision,
    compute_vehicle_risk_summary,
)
from src.utils.model_selection import (
    compute_temporal_strategy_validation,
    compute_val_stability_metrics,
    normalize_temporal_strategy,
)
from src.utils.artifact_utils import sha256_file
from src.utils.git_utils import format_model_version, get_git_info
from src.utils.logger_utils import (
    bind_run_id,
    ensure_run_id_filter,
    resolve_os_user,
    unbind_run_id,
)
from src.utils.tf_runtime import setup_deterministic_runtime
from src.utils.config_schema import validate_training_config
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
FEATURE_SCHEMA_FILENAME = "feature_schema.json"


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


def _resolve_main_scaler_features(proc):
    """
    Resolve ordem canonica do scaler principal para persistencia no schema.

    Prioridade:
    1) scaler.feature_names_in_ (fonte de verdade apos fit)
    2) proc.features_to_use (fallback de compatibilidade)
    """
    scaler = getattr(proc, "scaler", None)
    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and len(scaler_features) > 0:
        return list(scaler_features)

    fallback_features = list(getattr(proc, "features_to_use", []) or [])
    if fallback_features:
        logger.warning(
            "Nao foi possivel obter scaler.feature_names_in_; "
            "persistindo main_scaler_features via fallback proc.features_to_use."
        )
    return fallback_features


def _resolve_primary_percentile(config):
    """
    Resolve percentil operacional principal para decisao final.

    Mantido por compatibilidade com chamadas legadas.
    A resolucao canonica fica centralizada em
    _resolve_operational_percentile_config().
    """
    operational, _ = _resolve_operational_percentile_config(config)
    return operational


def _resolve_operational_percentile_config(config):
    """
    Normaliza percentis configurados e define percentil operacional.

    Regra:
    - lista vazia/invalida -> [95]
    - se 95 existir -> operacional=95
    - caso contrario -> operacional=primeiro percentil valido

    Returns:
        tuple: (operational_percentile, normalized_percentiles)
    """
    params = config.setdefault("parametros", {})
    raw_percentiles = params.get("percentis_teste", [])

    parsed = []
    for value in raw_percentiles:
        try:
            p = int(value)
        except Exception:
            continue
        if not 0 <= p <= 100:
            continue
        if p not in parsed:
            parsed.append(p)

    if not parsed:
        parsed = [95]

    params["percentis_teste"] = parsed
    operational = 95 if 95 in parsed else parsed[0]
    return operational, parsed


def _sanitize_config_name(config_path):
    """Retorna apenas o nome do arquivo de configuracao (sem path sensivel)."""
    if config_path is None:
        return "N/A"
    return os.path.basename(os.path.normpath(str(config_path)))


def _build_public_run_path(output_dir, run_id):
    """
    Gera identificador de path minimo para auditoria sem serializar caminho absoluto.
    Ex.: outputs/20260329_101010
    """
    normalized = os.path.normpath(str(output_dir))
    parent_name = os.path.basename(os.path.dirname(normalized)) or "outputs"
    return f"{parent_name}/{run_id}"


def _write_run_summary(metrics_dir, summary_payload):
    """Persist run_summary.json de forma resiliente."""
    os.makedirs(metrics_dir, exist_ok=True)
    log_json_path = os.path.join(metrics_dir, "run_summary.json")
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)
    return log_json_path


def _iter_exception_chain(exc):
    """Itera excecao principal e encadeamentos (__cause__/__context__)."""
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_tf_resource_exhausted_error(exc):
    """
    Detecta ResourceExhaustedError de TensorFlow sem importar TF cedo.

    Usa deteccao por nome/classe para cobrir excecoes encapsuladas
    sem forcar import de TensorFlow em caminho de falha.
    """
    for candidate in _iter_exception_chain(exc):
        cls = candidate.__class__
        class_name = cls.__name__
        class_module = (cls.__module__ or "").lower()
        if class_name == "ResourceExhaustedError":
            return True
        if "resourceexhausted" in class_name.lower() and "tensorflow" in class_module:
            return True
    return False


def _compute_alert_counters(df, operational_percentile):
    """
    Consolida contadores de alerta para manter semantica consistente entre
    runs_index, run_summary e tracking.
    """
    has_df = isinstance(df, pd.DataFrame)
    has_ensemble_alert = has_df and "ensemble_alert" in df.columns

    n_alerts_operational = (
        int(df["ensemble_alert"].eq(1.0).sum()) if has_ensemble_alert else "N/A"
    )
    n_alerts_p95 = (
        n_alerts_operational
        if has_ensemble_alert and int(operational_percentile) == 95
        else "N/A"
    )
    n_not_scored = (
        int(df["n_models_scored"].eq(0).sum())
        if has_df and "n_models_scored" in df.columns
        else "N/A"
    )

    return {
        "n_alerts_operational": n_alerts_operational,
        "n_alerts_p95": n_alerts_p95,
        "n_not_scored": n_not_scored,
    }


def _temporal_strategy_from_model_name(model_name):
    if str(model_name).startswith("Temporal_Union_"):
        return "union"
    if str(model_name).startswith("Temporal_Inter_"):
        return "inter"
    if str(model_name).startswith("Temporal_Baseline"):
        return "baseline"
    return None


def _temporal_strategy_from_score_col(score_col):
    if str(score_col).startswith("Temporal_Union_"):
        return "union"
    if str(score_col).startswith("Temporal_Inter_"):
        return "inter"
    if str(score_col).startswith("Temporal_Baseline"):
        return "baseline"
    return None


def _apply_temporal_strategy_policy(
    df,
    results_summary,
    score_cols,
    temporal_strategy,
    models_dir,
):
    """
    Aplica politica operacional dos cenarios temporais antes da exportacao.
    """
    strategy = normalize_temporal_strategy(temporal_strategy)
    if strategy == "all":
        return df, results_summary, score_cols

    kept_metrics = []
    allowed_temporal_model_names = set()
    temporal_model_names_seen = set()

    for metric in results_summary:
        model_name = metric.get("Model")
        temp_kind = _temporal_strategy_from_model_name(model_name)
        if temp_kind is None:
            kept_metrics.append(metric)
            continue
        temporal_model_names_seen.add(model_name)
        if temp_kind == strategy:
            kept_metrics.append(metric)
            allowed_temporal_model_names.add(model_name)

    kept_score_cols = []
    for col in score_cols:
        temp_kind = _temporal_strategy_from_score_col(col)
        if temp_kind is None or temp_kind == strategy:
            kept_score_cols.append(col)

    df_filtered = df.copy()
    drop_cols = []
    for col in df_filtered.columns:
        temp_kind = _temporal_strategy_from_score_col(col)
        if temp_kind is not None and temp_kind != strategy:
            drop_cols.append(col)
            continue
        if col.startswith("Temporal_"):
            prefix = col.split("_p", 1)[0]
            if prefix in temporal_model_names_seen and prefix not in allowed_temporal_model_names:
                drop_cols.append(col)
    if drop_cols:
        df_filtered = df_filtered.drop(columns=sorted(set(drop_cols)), errors="ignore")

    # Manter manifesto alinhado ao modo operacional explicito removendo artefatos nao eleitos.
    cleanup_map = {
        "union": ("temporal_inter_", "temporal_baseline.h5"),
        "inter": ("temporal_union_", "temporal_baseline.h5"),
        "baseline": ("temporal_union_", "temporal_inter_"),
    }
    for token in cleanup_map.get(strategy, ()):
        for fname in os.listdir(models_dir):
            should_remove = fname.startswith(token) if token.endswith("_") else fname == token
            if should_remove and fname.endswith(".h5"):
                fpath = os.path.join(models_dir, fname)
                try:
                    os.remove(fpath)
                    logger.info(
                        "Politica temporal operacional (%s): removido artefato nao eleito %s",
                        strategy,
                        fname,
                    )
                except OSError as exc:
                    logger.warning(
                        "Falha ao remover artefato temporal nao eleito (%s): %s",
                        fpath,
                        exc,
                    )

    logger.info(
        "Politica temporal operacional aplicada: strategy=%s | temporal_metrics_antes=%d | temporal_metrics_depois=%d",
        strategy,
        sum(1 for m in results_summary if str(m.get("Model", "")).startswith("Temporal_")),
        sum(1 for m in kept_metrics if str(m.get("Model", "")).startswith("Temporal_")),
    )
    return df_filtered, kept_metrics, kept_score_cols


def load_data(proc, config, input_path):
    """
    Wrapper de compatibilidade para testes e monkeypatch no modulo runner.
    """
    from src.pipeline.experiment_stages import load_data as _load_data

    return _load_data(proc, config, input_path)


def prepare_model_features(df, df_train, config, proc, models_dir):
    """
    Wrapper de compatibilidade para testes e monkeypatch no modulo runner.
    """
    from src.pipeline.experiment_stages import prepare_model_features as _prepare_model_features

    return _prepare_model_features(df, df_train, config, proc, models_dir)


def train_base_models(
    df,
    features_dict,
    config,
    models_dir,
    operational_percentile,
):
    """
    Wrapper de compatibilidade para testes e monkeypatch no modulo runner.
    """
    from src.pipeline.experiment_stages import train_base_models as _train_base_models

    return _train_base_models(
        df,
        features_dict,
        config,
        models_dir,
        operational_percentile,
    )


def train_temporal_models(
    df, features_dict, config, iso_masks, hbos_masks, train_end, models_dir, epochs
):
    """
    Wrapper de compatibilidade para testes e monkeypatch no modulo runner.
    """
    from src.pipeline.experiment_stages import train_temporal_models as _train_temporal_models

    return _train_temporal_models(
        df,
        features_dict,
        config,
        iso_masks,
        hbos_masks,
        train_end,
        models_dir,
        epochs,
    )

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
    Wrapper de compatibilidade para testes e monkeypatch no modulo runner.
    """
    from src.pipeline.experiment_export import export_results as _export_results

    return _export_results(
        df=df,
        results_summary=results_summary,
        score_columns_audit=score_columns_audit,
        config=config,
        metrics_dir=metrics_dir,
        master_dir=master_dir,
        models_dir=models_dir,
        df_train=df_train,
        df_val=df_val,
        stats=stats,
        run_id=run_id,
        git_info=git_info,
        model_version=model_version,
        operational_percentile=operational_percentile,
        temporal_selection_audit_df=temporal_selection_audit_df,
    )

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

    decision_percentile = (
        int(operational_percentile)
        if operational_percentile is not None
        else _resolve_primary_percentile(config)
    )

    # Camada de decisao final do ensemble
    logger.info("Calculando decisao final do ensemble...")
    df = compute_ensemble_decision(df, percentile=decision_percentile)

    # --- COBERTURA DE AVALIACAO POR MODELO ---
    p_label_suffix = f"_p{decision_percentile}_label"
    percentile_label_cols = [c for c in df.columns if c.endswith(p_label_suffix)]
    iso_label_cols = [c for c in percentile_label_cols if c.startswith("ISO")]
    hbos_label_cols = [c for c in percentile_label_cols if c.startswith("HBOS")]
    temp_label_cols = [c for c in percentile_label_cols if c.startswith("Temporal")]

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
        temporal_cfg = normalize_temporal_strategy(
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
                "explicit_config" if temporal_cfg != "all" else "legacy_all_no_single_selection",
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
        # Adicionar run_id ao perfil para rastreabilidade
        stats["run_id"] = run_id if run_id else "unversioned"
        stats["run_timestamp"] = datetime.datetime.now().isoformat()
        stats["operational_percentile"] = int(decision_percentile)
        with open(os.path.join(metrics_dir, "perfil_dados.json"), "w") as f:
            json.dump(stats, f, indent=4, default=str)
        logger.info(f"Perfil de dados salvo com run_id={stats['run_id']}")

        # Gerar relatorio HTML automaticamente
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
        logger.error(
            "Relatorio HTML nao foi gerado porque stats estao indisponiveis."
        )

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

    if feature_schema_path and os.path.exists(feature_schema_path):
        manifest["feature_schema"] = {
            "path": os.path.relpath(feature_schema_path, models_dir),
            "type": "json",
            "sha256": sha256_file(feature_schema_path),
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

    return {
        "report_required": report_required,
        "report_status": report_status,
        "report_error": report_error,
        "report_path": report_path,
    }


def run_experiment(
    config_path="config_mapeamento.yaml",
    input_path=None,
    output_dir="outputs",
    epochs=None,
    seed=42,
    run_id=None,
    tf_device="auto",
):
    """
    Orquestra o pipeline completo de deteccao de anomalias.
    """
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir, run_id = _resolve_versioned_output_dir(output_dir, run_id)
    config_name_for_audit = _sanitize_config_name(config_path)
    run_path_for_audit = _build_public_run_path(output_dir, run_id)
    executor_os_user = resolve_os_user()
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
            "%(asctime)s - %(name)s - %(levelname)s - [run_id=%(run_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    ensure_run_id_filter(_run_file_handler)
    sspdf_logger.addHandler(_run_file_handler)

    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(
        logging.Formatter("%(levelname)s - [run_id=%(run_id)s] - %(message)s")
    )
    ensure_run_id_filter(_console_handler)
    sspdf_logger.addHandler(_console_handler)

    sspdf_logger.setLevel(logging.INFO)
    sspdf_logger.info(f"Logger configurado para run_id={run_id}")
    sspdf_logger.info(f"Executor SO: {executor_os_user}")
    sspdf_logger.info(f"Run path (publico): {run_path_for_audit}")
    from src.utils.tracking import (
        end_run,
        init_experiment,
        log_artifact,
        log_metrics,
        log_params,
    )

    config = {}
    stats = {}
    df = pd.DataFrame()
    global_seed = seed
    current_stage = "INICIALIZACAO"
    failed_stage = None
    run_status = "SUCCESS"
    error_message = None
    captured_exception = None
    _tracking_run = None
    report_required = True
    report_status = "not_attempted"
    report_error = None
    operational_percentile = 95
    configured_percentiles = [95]
    temporal_strategy_configured = "all"
    temporal_strategy_effective = "all"
    temporal_strategy_selection_source = "legacy_all_no_single_selection"
    temporal_status = "not_attempted"
    temporal_error = None
    temporal_degraded_mode = False
    temporal_failed = False
    degraded_mode = False
    temporal_selection_audit_df = pd.DataFrame()
    run_started_at = datetime.datetime.now().isoformat()
    run_log_token = bind_run_id(run_id)
    git_info = {
        "commit_hash": "unknown",
        "commit_short": "unknown",
        "branch": "unknown",
        "is_dirty": None,
        "dirty_warning": None,
        "commit_message": None,
        "commit_timestamp": None,
    }
    model_version = f"unknown-{run_id}"

    try:
        current_stage = "CONFIGURACAO"
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        config = validate_training_config(config)

        global_seed = config.get("random_state", seed)
        operational_percentile, configured_percentiles = _resolve_operational_percentile_config(
            config
        )
        temporal_strategy_configured = normalize_temporal_strategy(
            config.get("parametros", {})
            .get("temporal", {})
            .get("temporal_strategy", "all")
        )
        temporal_strategy_effective = temporal_strategy_configured
        temporal_strategy_selection_source = (
            "explicit_config" if temporal_strategy_configured != "all"
            else "legacy_all_no_single_selection"
        )
        logger.info(
            f"Percentil operacional efetivo: p{operational_percentile} "
            f"| percentis_teste={configured_percentiles}"
        )
        logger.info(
            "Politica temporal configurada: temporal_strategy=%s",
            temporal_strategy_configured,
        )
        report_required = bool(
            config.get("configuracoes_gerais", {}).get("report_required", True)
        )

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

        _, tf_runtime = setup_deterministic_runtime(
            seed=global_seed,
            tf_device=tf_device,
        )
        logger.info(
            f"Runtime deterministico ativo no run_experiment | seed={global_seed} "
            f"| tf_device={tf_runtime['requested']} | tf_ativo={tf_runtime['active']} "
            f"| gpus_detectadas={tf_runtime['gpu_count']}"
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
                    "temporal_strategy": temporal_strategy_configured,
                    "percentis_teste": str(configured_percentiles),
                    "operational_percentile": operational_percentile,
                }
            )

        proc = DataProcessor(config)
        proc.models_dir = models_dir

        logger.info(f"Random state efetivo: {global_seed}")
        logger.info(f"Epochs temporais efetivas: {epochs}")

        current_stage = "ETAPA 1: CARGA E PROCESSAMENTO DE DADOS"
        logger.info("=" * 80)
        logger.info(current_stage)
        df, df_train, df_val, df_test, proc, stats = load_data(proc, config, input_path)
        stats["main_scaler_features"] = _resolve_main_scaler_features(proc)

        current_stage = "ETAPA 2: PREPARACAO DE FEATURES POR MODELO"
        logger.info("=" * 80)
        logger.info(current_stage)
        features_dict = prepare_model_features(df, df_train, config, proc, models_dir)
        stats["features_iso"] = features_dict["iso_features"]
        stats["features_hbos"] = features_dict["hbos_features"]
        stats["features_gru"] = features_dict["gru_features"]

        current_stage = "ETAPA 3: TREINAMENTO DE MODELOS BASE"
        logger.info("=" * 80)
        logger.info(current_stage)
        df, iso_masks, hbos_masks, results_summary, score_cols = train_base_models(
            df,
            features_dict,
            config,
            models_dir,
            operational_percentile=operational_percentile,
        )

        current_stage = "ETAPA 4: TREINAMENTO DE MODELOS TEMPORAIS"
        logger.info("=" * 80)
        logger.info(current_stage)
        train_end = stats.get("split_temporal", {}).get("train_end_index")
        if train_end is None:
            train_end = int(
                len(df)
                * config.get("parametros", {}).get("split_ratios", {}).get("train", 0.6)
            )
        temporal_status = "running"
        try:
            df, temporal_results, temporal_cols = train_temporal_models(
                df, features_dict, config, iso_masks, hbos_masks, train_end, models_dir, epochs
            )
            results_summary.extend(temporal_results)
            score_cols.extend(temporal_cols)
            temporal_status = "success"
        except Exception as temporal_exc:
            temporal_degraded_mode = True
            temporal_failed = True
            degraded_mode = True
            if isinstance(temporal_exc, MemoryError):
                temporal_error = f"MemoryError: {temporal_exc}"
            elif _is_tf_resource_exhausted_error(temporal_exc):
                temporal_error = f"ResourceExhaustedError: {temporal_exc}"
            else:
                temporal_error = f"{type(temporal_exc).__name__}: {temporal_exc}"
            temporal_status = "failed"
            logger.exception(
                "Falha na etapa temporal (%s). Continuando em modo degradado "
                "tabular-only (ISO/HBOS).",
                temporal_error,
            )

        stats["temporal_status"] = temporal_status
        stats["temporal_error"] = temporal_error
        stats["temporal_degraded_mode"] = temporal_degraded_mode
        stats["temporal_failed"] = temporal_failed
        stats["degraded_mode"] = degraded_mode
        stats["temporal_strategy_configured"] = temporal_strategy_configured

        # Auditoria de estrategia temporal deve usar scores completos pre-filtragem.
        if (
            isinstance(df_train, pd.DataFrame)
            and isinstance(df_val, pd.DataFrame)
            and len(df_train) > 0
            and len(df_val) > 0
        ):
            df_train_scored_pre_filter = df.loc[df_train.index]
            df_val_scored_pre_filter = df.loc[df_val.index]
            temporal_selection_audit_df = compute_temporal_strategy_validation(
                df_train=df_train_scored_pre_filter,
                df_val=df_val_scored_pre_filter,
                score_cols=[c for c in score_cols if c.startswith("Temporal_")],
                percentile=operational_percentile,
            )

        if temporal_degraded_mode:
            temporal_strategy_effective = "tabular_only"
            temporal_strategy_selection_source = (
                "temporal_stage_failed_degraded_tabular_only"
            )
        else:
            if temporal_strategy_configured != "all":
                df, results_summary, score_cols = _apply_temporal_strategy_policy(
                    df=df,
                    results_summary=results_summary,
                    score_cols=score_cols,
                    temporal_strategy=temporal_strategy_configured,
                    models_dir=models_dir,
                )
                temporal_strategy_effective = temporal_strategy_configured
                temporal_strategy_selection_source = "explicit_config"
            else:
                temporal_strategy_effective = "all"
                temporal_strategy_selection_source = "legacy_all_no_single_selection"
        stats["temporal_strategy_effective"] = temporal_strategy_effective
        stats["temporal_strategy_selection_source"] = temporal_strategy_selection_source

        current_stage = "ETAPA 5: EXPORTACAO DE RESULTADOS"
        logger.info("=" * 80)
        logger.info(current_stage)
        export_meta = export_results(
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
            operational_percentile=operational_percentile,
            temporal_selection_audit_df=temporal_selection_audit_df,
        )
        report_required = bool(export_meta.get("report_required", report_required))
        report_status = export_meta.get("report_status", "not_attempted")
        report_error = export_meta.get("report_error")
        if report_required and report_status != "success":
            raise RuntimeError(
                "Relatorio HTML obrigatorio nao foi gerado: "
                f"{report_error or report_status}"
            )

        index_path = os.path.join(os.path.dirname(output_dir), "runs_index.csv")
        alert_counters = _compute_alert_counters(df, operational_percentile)
        run_entry = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "output_dir": run_path_for_audit,
            "run_path": run_path_for_audit,
            "model_version": model_version,
            "commit_hash": git_info["commit_hash"],
            "branch": git_info["branch"],
            "is_dirty": git_info["is_dirty"],
            "config_path": config_name_for_audit,
            "config_name": config_name_for_audit,
            "executor_os_user": executor_os_user,
            "n_records": len(df),
            "operational_percentile": operational_percentile,
            "n_alerts_operational": alert_counters["n_alerts_operational"],
            # Campo legado de compatibilidade para dashboards antigos.
            # Somente preenchido quando o percentil operacional efetivo e 95.
            "n_alerts_p95": alert_counters["n_alerts_p95"],
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

    except KeyboardInterrupt as e:
        run_status = "FAILED"
        failed_stage = current_stage
        error_message = "KeyboardInterrupt: execucao interrompida pelo usuario."
        captured_exception = e
        logger.warning(error_message)
    except Exception as e:
        run_status = "FAILED"
        failed_stage = current_stage
        error_message = f"{type(e).__name__}: {e}"
        captured_exception = e
        logger.exception(
            f"Falha em {current_stage}. Erro: {e}"
        )
    finally:
        alert_counters = _compute_alert_counters(df, operational_percentile)
        summary_payload = {
            "run_id": run_id,
            "run_timestamp_start": run_started_at,
            "run_timestamp_end": datetime.datetime.now().isoformat(),
            "config_path": config_name_for_audit,
            "config_name": config_name_for_audit,
            "output_dir": run_path_for_audit,
            "run_path": run_path_for_audit,
            "executor_os_user": executor_os_user,
            "status": run_status,
            "failed_stage": failed_stage,
            "error_message": error_message,
            "temporal_status": temporal_status,
            "temporal_error": temporal_error,
            "temporal_failed": temporal_failed,
            "temporal_degraded_mode": temporal_degraded_mode,
            "degraded_mode": degraded_mode,
            "temporal_strategy_configured": temporal_strategy_configured,
            "temporal_strategy_effective": temporal_strategy_effective,
            "temporal_strategy_selection_source": temporal_strategy_selection_source,
            "report_required": report_required,
            "report_status": report_status,
            "report_error": report_error,
            "parameters": {
                "seed": global_seed,
                "epochs": epochs,
                "percentis_teste": configured_percentiles,
                "operational_percentile": operational_percentile,
                "split_ratios": config.get("parametros", {}).get("split_ratios", {}),
                "iso_config": config.get("parametros", {}).get("isolation_forest", {}),
                "hbos_config": config.get("parametros", {}).get("hbos", {}),
                "temporal_config": config.get("parametros", {}).get("temporal", {}),
                "temporal_strategy_configured": temporal_strategy_configured,
                "temporal_strategy_effective": temporal_strategy_effective,
                "temporal_strategy_selection_source": temporal_strategy_selection_source,
            },
            "dataset_profile": {
                "total_records": len(df) if isinstance(df, pd.DataFrame) else "N/A",
                "total_vehicles": stats.get("total_veiculos", "N/A"),
                "period": stats.get("periodo", "N/A"),
            },
            "results_summary": {
                "operational_percentile": operational_percentile,
                "n_alerts_operational": alert_counters["n_alerts_operational"],
                # Campo legado de compatibilidade para consumidores historicos.
                "n_alerts_p95": alert_counters["n_alerts_p95"],
                "n_not_scored": alert_counters["n_not_scored"],
            },
        }

        run_summary_path = None
        try:
            run_summary_path = _write_run_summary(metrics_dir, summary_payload)
            logger.info(f"Log estruturado JSON salvo: {run_summary_path}")
        except Exception as summary_exc:
            logger.error(f"Falha ao persistir run_summary.json: {summary_exc}")

        # Logar metricas de resultado no MLflow apenas quando houve sucesso.
        if (
            run_status == "SUCCESS"
            and isinstance(df, pd.DataFrame)
            and "ensemble_alert" in df.columns
        ):
            n_alerts = int((df["ensemble_alert"] == 1.0).sum())
            n_total = len(df)
            metrics_payload = {
                "n_records_total": n_total,
                "operational_percentile": float(operational_percentile),
                "n_alerts_operational": n_alerts,
                "alert_rate_pct": (
                    round(n_alerts / n_total * 100, 4) if n_total > 0 else 0
                ),
                "n_not_scored": (
                    int(
                        (
                            df.get("n_models_scored", pd.Series(dtype=float)) == 0
                        ).sum()
                    )
                    if "n_models_scored" in df.columns
                    else 0
                ),
            }
            if int(operational_percentile) == 95:
                # Mantido para retrocompatibilidade semantica em pipelines legados.
                metrics_payload["n_alerts_p95"] = n_alerts
            log_metrics(metrics_payload)

        # Logar artefatos chave (existentes).
        log_artifact(os.path.join(metrics_dir, "perfil_dados.json"))
        log_artifact(os.path.join(metrics_dir, "concordancia_modelos.csv"))
        log_artifact(os.path.join(metrics_dir, "vehicle_risk_ranking.csv"))
        log_artifact(os.path.join(os.path.dirname(metrics_dir), "relatorio_executivo.html"))
        if run_summary_path:
            log_artifact(run_summary_path)

        if run_status == "FAILED":
            end_run(status="FAILED")
        else:
            end_run(status="FINISHED")
            logger.info("EXPERIMENTO FINALIZADO!")
        unbind_run_id(run_log_token)

    if captured_exception is not None:
        raise captured_exception

    return output_dir


if __name__ == "__main__":
    run_experiment()
