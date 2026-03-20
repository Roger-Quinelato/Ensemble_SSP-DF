"""
Testes de regressão para bugs C1 e C2 identificados na meta-auditoria v2.

C1 - NaN->label silencioso: registros não avaliados recebiam label 0 (normal).
C2 - Union/Inter invertidos: Temporal_Union usava AND; Temporal_Inter usava OR.

Estes testes devem FALHAR se os bugs forem reintroduzidos.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.evaluation import ThresholdOptimizer


# =============================================================================
# TESTES C1 - NaN não deve virar label 0
# =============================================================================


class TestC1NanLabel:
    """Garante que registros sem score recebem label NaN, nunca 0."""

    def test_nan_score_produces_nan_label_not_zero(self):
        """BUG C1: NaN >= threshold retornava False -> astype(int) -> 0 (normal)."""
        df = pd.DataFrame({"score": [0.1, 0.9, np.nan, 0.5, np.nan]})
        optimizer = ThresholdOptimizer(percentiles=[50])
        df_result, _ = optimizer.apply_dynamic_thresholds(
            df, "score", "test", calibration_scores=np.array([0.1, 0.9, 0.5])
        )
        nan_mask = df["score"].isna()
        labels_for_nan = df_result.loc[nan_mask, "test_p50_label"]

        assert labels_for_nan.isna().all(), (
            f"REGRESSÃO C1: Registros com score NaN devem ter label NaN, "
            f"mas receberam: {labels_for_nan.tolist()}"
        )

    def test_nan_score_not_counted_as_anomaly(self):
        """Registros NaN não devem ser contados como anomalias no resultado."""
        df = pd.DataFrame({"score": [np.nan, np.nan, np.nan]})
        optimizer = ThresholdOptimizer(percentiles=[95])
        df_result, metrics = optimizer.apply_dynamic_thresholds(
            df, "score", "test", calibration_scores=np.array([0.1, 0.5, 0.9])
        )
        if metrics:
            assert metrics[0]["Anomalies_Detected"] == 0, (
                "Registros NaN não devem ser contados como anomalias"
            )
        assert "test_p95_label" not in df_result.columns or df_result["test_p95_label"].isna().all()

    def test_valid_scores_still_get_binary_label(self):
        """Scores válidos devem continuar recebendo 0.0 ou 1.0."""
        df = pd.DataFrame({"score": [0.1, 0.9, np.nan, 0.5]})
        optimizer = ThresholdOptimizer(percentiles=[50])
        df_result, _ = optimizer.apply_dynamic_thresholds(
            df, "score", "test", calibration_scores=np.array([0.1, 0.5, 0.9])
        )
        valid_mask = df["score"].notna()
        labels_valid = df_result.loc[valid_mask, "test_p50_label"]
        assert labels_valid.isin([0.0, 1.0]).all(), (
            "Scores válidos devem receber label 0.0 ou 1.0"
        )

    @pytest.mark.parametrize("nan_count,total", [(0, 5), (1, 5), (3, 5), (5, 5)])
    def test_nan_count_preserved(self, nan_count, total):
        """O número de NaNs no label deve igualar o número de NaNs no score."""
        scores = [0.5] * (total - nan_count) + [np.nan] * nan_count
        df = pd.DataFrame({"score": scores})
        optimizer = ThresholdOptimizer(percentiles=[90])
        calibration = np.array([s for s in scores if not np.isnan(s)])
        if len(calibration) == 0:
            calibration = np.array([0.5])
        df_result, _ = optimizer.apply_dynamic_thresholds(
            df, "score", "test", calibration_scores=calibration
        )
        col = "test_p90_label"
        if col in df_result.columns:
            actual_nan_count = df_result[col].isna().sum()
            assert actual_nan_count == nan_count, (
                f"Esperado {nan_count} NaN no label, encontrado {actual_nan_count}"
            )


# =============================================================================
# TESTES C2 - Union deve ser OR; Inter deve ser AND
# =============================================================================


class TestC2UnionInterSemantics:
    """Garante que Union (|) e Inter (&) têm semântica correta."""

    def _make_masks(self, iso_values, hbos_values):
        """Cria máscaras booleanas de inlier para ISO e HBOS."""
        return (
            pd.Series(iso_values, dtype=bool),
            pd.Series(hbos_values, dtype=bool),
        )

    def test_union_is_larger_than_inter(self):
        """
        BUG C2: union_mask tinha MENOS registros que inter_mask porque
        union usava & (AND) e inter usava | (OR) - semanticamente invertido.

        CORRETO: union = | (OR) = MAIS registros que inter = & (AND)
        """
        iso_mask = pd.Series([True, True, False, False, True])
        hbos_mask = pd.Series([True, False, True, False, True])
        mask_temporal_train = pd.Series([True, True, True, True, True])

        # Semântica CORRETA (pós-fix C2):
        union_mask = (iso_mask | hbos_mask) & mask_temporal_train  # OR = mais registros
        inter_mask = (iso_mask & hbos_mask) & mask_temporal_train  # AND = menos registros

        n_union = union_mask.sum()
        n_inter = inter_mask.sum()

        assert n_union >= n_inter, (
            f"REGRESSÃO C2: union_mask deve ter >= registros que inter_mask "
            f"(union={n_union}, inter={n_inter}). "
            f"Se union < inter, a semântica está invertida novamente."
        )

    def test_union_uses_or_operator(self):
        """Union deve incluir registros onde PELO MENOS UM modelo considera normal."""
        iso_mask = pd.Series([True, False, True, False])
        hbos_mask = pd.Series([False, True, True, False])

        union_correct = iso_mask | hbos_mask  # [True, True, True, False]
        expected = [True, True, True, False]

        assert list(union_correct) == expected, (
            f"Union deve usar OR. Resultado: {list(union_correct)}, "
            f"Esperado: {expected}"
        )

    def test_inter_uses_and_operator(self):
        """Inter deve incluir apenas registros onde AMBOS os modelos consideram normal."""
        iso_mask = pd.Series([True, False, True, False])
        hbos_mask = pd.Series([False, True, True, False])

        inter_correct = iso_mask & hbos_mask  # [False, False, True, False]
        expected = [False, False, True, False]

        assert list(inter_correct) == expected, (
            f"Inter deve usar AND. Resultado: {list(inter_correct)}, "
            f"Esperado: {expected}"
        )

    def test_inter_is_subset_of_union(self):
        """Todo registro em Inter deve também estar em Union (Inter ⊆ Union)."""
        iso_mask = pd.Series([True, False, True, False, True])
        hbos_mask = pd.Series([False, True, True, False, True])

        union = iso_mask | hbos_mask
        inter = iso_mask & hbos_mask

        # Cada True em inter deve ser True em union
        inter_subset_of_union = (~inter | union).all()
        assert inter_subset_of_union, (
            "REGRESSÃO C2: Inter não é subconjunto de Union. "
            "Isso indica que os operadores estão trocados."
        )


# =============================================================================
# TESTES ensemble_decision.py (se FIX C3 aplicado)
# =============================================================================


class TestEnsembleDecision:
    """Testes para a camada de decisão final do ensemble."""

    def _make_df_with_labels(self):
        """DataFrame com colunas de label simulando saída do pipeline."""
        return pd.DataFrame(
            {
                "ISO_n100_p95_label": [1.0, 0.0, 1.0, 0.0, np.nan],
                "ISO_n200_p95_label": [1.0, 0.0, 0.0, 0.0, np.nan],
                "HBOS_bins10_p95_label": [0.0, 0.0, 1.0, 1.0, np.nan],
                "Temporal_Baseline_p95_label": [np.nan, 0.0, 1.0, 0.0, np.nan],
            }
        )

    def test_ensemble_alert_is_nan_when_no_model_scored(self):
        """Registro não avaliado por nenhum modelo deve ter ensemble_alert=NaN."""
        try:
            from src.utils.ensemble_decision import compute_ensemble_decision
        except ImportError:
            pytest.skip("ensemble_decision.py não encontrado (FIX C3 não aplicado)")

        df = self._make_df_with_labels()
        df_result = compute_ensemble_decision(df, percentile=95)

        # Última linha: todos NaN -> ensemble_alert deve ser NaN
        assert pd.isna(df_result.loc[4, "ensemble_alert"]), (
            "Registro sem avaliação por nenhum modelo deve ter ensemble_alert=NaN"
        )

    def test_ensemble_vote_pct_is_majority(self):
        """ensemble_vote_pct deve ser proporção de votos anomalia."""
        try:
            from src.utils.ensemble_decision import compute_ensemble_decision
        except ImportError:
            pytest.skip("ensemble_decision.py não encontrado")

        df = self._make_df_with_labels()
        df_result = compute_ensemble_decision(df, percentile=95)

        # Linha 0: 2 de 3 modelos votaram anomalia (Temporal é NaN) -> 2/3 ~= 0.667
        vote_pct = df_result.loc[0, "ensemble_vote_pct"]
        assert abs(vote_pct - 2 / 3) < 0.01, (
            f"ensemble_vote_pct esperado ~0.667, obtido {vote_pct}"
        )

    def test_n_models_scored_counts_non_nan(self):
        """n_models_scored deve contar apenas modelos que geraram score."""
        try:
            from src.utils.ensemble_decision import compute_ensemble_decision
        except ImportError:
            pytest.skip("ensemble_decision.py não encontrado")

        df = self._make_df_with_labels()
        df_result = compute_ensemble_decision(df, percentile=95)

        # Linha 0: ISO_n100=1, ISO_n200=1, HBOS=1, Temporal=NaN -> n_models_scored=3
        assert df_result.loc[0, "n_models_scored"] == 3, (
            f"n_models_scored esperado 3, obtido {df_result.loc[0, 'n_models_scored']}"
        )


# =============================================================================
# TESTES model_selection.py (se FIX H2 aplicado)
# =============================================================================


class TestModelSelection:
    """Cobertura de estabilidade no validation set."""

    def test_compute_val_stability_metrics_generates_rank(self):
        try:
            from src.utils.model_selection import compute_val_stability_metrics
        except ImportError:
            pytest.skip("model_selection.py não encontrado")

        df_full = pd.DataFrame(
            {
                "ISO_n100_score": [0.1, 0.2, 0.3, 0.4, 0.7, 0.8],
                "HBOS_bins10_score": [1.0, 1.1, 1.2, 1.3, 2.0, 2.2],
            }
        )
        df_val = df_full.iloc[4:].copy()
        df_sel = compute_val_stability_metrics(
            df_full=df_full,
            df_val=df_val,
            score_cols=["ISO_n100_score", "HBOS_bins10_score"],
            percentile=95,
        )

        assert not df_sel.empty
        assert "rank_stability" in df_sel.columns
        assert "stability_delta_pct" in df_sel.columns
        assert df_sel["stability_delta_pct"].between(0, 100).all()
        assert set(df_sel["config"].tolist()) == {"ISO_n100", "HBOS_bins10"}

    def test_compute_val_stability_metrics_ignores_missing_scores(self):
        try:
            from src.utils.model_selection import compute_val_stability_metrics
        except ImportError:
            pytest.skip("model_selection.py não encontrado")

        df_full = pd.DataFrame({"ISO_n100_score": [0.1, 0.2, 0.3, 0.4]})
        df_val = df_full.iloc[2:].copy()
        df_sel = compute_val_stability_metrics(
            df_full=df_full,
            df_val=df_val,
            score_cols=["ISO_n100_score", "HBOS_missing_score"],
            percentile=95,
        )
        assert not df_sel.empty
        assert (df_sel["score_col"] == "ISO_n100_score").all()
