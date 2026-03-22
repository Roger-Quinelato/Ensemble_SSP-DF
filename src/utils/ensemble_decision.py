"""
Final decision layer for the anomaly detection ensemble.

Implements simple and auditable majority voting across model outputs.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("sspdf")


def compute_ensemble_decision(df, percentile=95):
    """
    Calcula score final do ensemble por votacao por familia.

    Cada familia (ISO, HBOS, Temporal) tem peso igual (1/3),
    independente do numero de variantes em cada familia.
    Isso evita que familias com mais variantes dominem a decisao.

    Args:
        df: DataFrame com colunas de label geradas pelo pipeline.
        percentile: Percentil operacional (default: 95).
    Returns:
        pd.DataFrame com colunas de decisao adicionadas.
    """
    p_suffix = f"_p{percentile}_label"
    iso_cols = [
        c for c in df.columns if c.startswith("ISO") and c.endswith(p_suffix)
    ]
    hbos_cols = [
        c for c in df.columns if c.startswith("HBOS") and c.endswith(p_suffix)
    ]
    temp_cols = [
        c for c in df.columns if c.startswith("Temporal") and c.endswith(p_suffix)
    ]

    if not iso_cols and not hbos_cols and not temp_cols:
        logger.warning(f"Nenhuma coluna de label para p{percentile} encontrada")
        return df

    logger.info("=" * 80)
    logger.info("CAMADA DE DECISAO FINAL DO ENSEMBLE (votacao por familia)")
    logger.info(f"   Percentil operacional: p{percentile}")
    logger.info(f"   ISO      ({len(iso_cols)} variantes): {iso_cols}")
    logger.info(f"   HBOS     ({len(hbos_cols)} variantes): {hbos_cols}")
    if temp_cols:
        logger.info(
            f"   Temporal ({len(temp_cols)} variantes): primeiras 3 = {temp_cols[:3]}..."
        )
    else:
        logger.info("   Temporal (0 variantes): []")
    logger.info("   MÉTODO: voto por família — cada família tem peso 1/3")

    # Score por familia = media interna (0 a 1)
    # NaN se nenhum modelo da familia avaliou este registro
    if iso_cols:
        df["vote_iso"] = df[iso_cols].mean(axis=1, skipna=True)
    else:
        df["vote_iso"] = np.nan
        logger.warning("   Familia ISO sem modelos - vote_iso=NaN")

    if hbos_cols:
        df["vote_hbos"] = df[hbos_cols].mean(axis=1, skipna=True)
    else:
        df["vote_hbos"] = np.nan
        logger.warning("   Familia HBOS sem modelos - vote_hbos=NaN")

    if temp_cols:
        df["vote_temp"] = df[temp_cols].mean(axis=1, skipna=True)
    else:
        df["vote_temp"] = np.nan
        logger.warning("   Familia Temporal sem modelos - vote_temp=NaN")

    # Score ensemble = media das familias disponiveis
    family_vote_cols = ["vote_iso", "vote_hbos", "vote_temp"]
    df["n_families_scored"] = df[family_vote_cols].notna().sum(axis=1).astype(int)
    df["ensemble_vote_pct"] = df[family_vote_cols].mean(axis=1, skipna=True)

    # n_models_scored: total de modelos individuais que avaliaram (auditoria)
    all_label_cols = iso_cols + hbos_cols + temp_cols
    df["n_models_scored"] = df[all_label_cols].notna().sum(axis=1).astype(int)

    # ensemble_alert: NaN se nenhuma familia avaliou
    df["ensemble_alert"] = np.where(
        df["ensemble_vote_pct"].isna(),
        np.nan,
        (df["ensemble_vote_pct"] >= 0.5).astype(float),
    )

    # Alertas por familia (transparencia para auditoria)
    df["iso_alert"] = np.where(
        df["vote_iso"].isna(), np.nan, (df["vote_iso"] >= 0.5).astype(float)
    )
    df["hbos_alert"] = np.where(
        df["vote_hbos"].isna(), np.nan, (df["vote_hbos"] >= 0.5).astype(float)
    )
    df["temp_alert"] = np.where(
        df["vote_temp"].isna(), np.nan, (df["vote_temp"] >= 0.5).astype(float)
    )

    # Estatisticas
    n_total = len(df)
    n_alerts = int((df["ensemble_alert"] == 1.0).sum())
    n_iso_al = int((df["iso_alert"] == 1.0).sum()) if "iso_alert" in df.columns else "N/A"
    n_hbos_al = (
        int((df["hbos_alert"] == 1.0).sum()) if "hbos_alert" in df.columns else "N/A"
    )
    n_temp_al = (
        int((df["temp_alert"] == 1.0).sum()) if "temp_alert" in df.columns else "N/A"
    )
    n_agree_all3 = int(
        (
            (df["iso_alert"] == 1.0)
            & (df["hbos_alert"] == 1.0)
            & (df["temp_alert"] == 1.0)
        ).sum()
    )
    pct_base = n_total if n_total > 0 else 1

    logger.info("RESULTADO DO ENSEMBLE (votacao por familia):")
    logger.info(f"   Total de registros:           {n_total:,}")
    if isinstance(n_iso_al, int):
        logger.info(
            f"   Alertas ISO:                  {n_iso_al:,} ({n_iso_al / pct_base * 100:.2f}%)"
        )
    else:
        logger.info(f"   Alertas ISO:                  {n_iso_al}")
    if isinstance(n_hbos_al, int):
        logger.info(
            f"   Alertas HBOS:                 {n_hbos_al:,} ({n_hbos_al / pct_base * 100:.2f}%)"
        )
    else:
        logger.info(f"   Alertas HBOS:                 {n_hbos_al}")
    if isinstance(n_temp_al, int):
        logger.info(
            f"   Alertas Temporal:             {n_temp_al:,} ({n_temp_al / pct_base * 100:.2f}%)"
        )
    else:
        logger.info(f"   Alertas Temporal:             {n_temp_al}")
    logger.info(
        f"   Concordancia total (3 de 3):  {n_agree_all3:,} ({n_agree_all3 / pct_base * 100:.2f}%)"
    )
    logger.info(
        f"   ALERTAS FINAIS (ensemble):    {n_alerts:,} ({n_alerts / pct_base * 100:.2f}%)"
    )
    logger.info("=" * 80)

    return df


def compute_vehicle_risk_summary(df, placa_col="placa", percentile=95):
    """
    Aggregate ensemble alerts per vehicle for operational ranking.

    Args:
        df: DataFrame with ensemble_alert and ensemble_vote_pct columns.
        placa_col: Vehicle plate column name.
        percentile: Percentile used in final decision (for logging context).
    Returns:
        pd.DataFrame with vehicle risk ranking.
    """
    if "ensemble_alert" not in df.columns:
        raise ValueError("Execute compute_ensemble_decision() antes de chamar esta funcao")

    if placa_col not in df.columns:
        logger.warning(f"Coluna '{placa_col}' nao encontrada. Pulando resumo por veiculo.")
        return pd.DataFrame()

    logger.info(f"Gerando ranking de risco por veiculo usando p{percentile}...")
    vehicle_summary = (
        df.groupby(placa_col)
        .agg(
            total_registros=("ensemble_alert", "count"),
            registros_avaliados=("ensemble_vote_pct", lambda x: x.notna().sum()),
            alertas_ensemble=("ensemble_alert", lambda x: (x == 1.0).sum()),
            score_medio=("ensemble_vote_pct", "mean"),
            score_maximo=("ensemble_vote_pct", "max"),
        )
        .reset_index()
    )

    vehicle_summary["taxa_alerta"] = (
        vehicle_summary["alertas_ensemble"]
        / vehicle_summary["registros_avaliados"].clip(lower=1)
    )

    vehicle_summary = vehicle_summary.sort_values("score_maximo", ascending=False)
    vehicle_summary["ranking_risco"] = range(1, len(vehicle_summary) + 1)

    logger.info(f"Ranking de risco gerado: {len(vehicle_summary)} veiculos analisados")
    top5 = vehicle_summary.head(5)
    for _, row in top5.iterrows():
        logger.info(
            f"   #{int(row['ranking_risco'])} {row[placa_col]}: "
            f"score_max={row['score_maximo']:.3f}, "
            f"alertas={int(row['alertas_ensemble'])}/{int(row['registros_avaliados'])}"
        )
    return vehicle_summary
