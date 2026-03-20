# src/evaluation.py

import numpy as np
import pandas as pd
from src.utils.logger_utils import log_execution
from src.utils.logger_utils import logger

class ThresholdOptimizer:
    """
    Otimiza thresholds de detecÃ§Ã£o usando mÃºltiplos percentis.
    Ãštil para balancear Precision vs Recall em detecÃ§Ã£o de anomalias.
    """
    def __init__(self, percentiles=None):
        """
        Inicializa o otimizador de thresholds para detecÃ§Ã£o de anomalias.
        Args:
            percentiles (list, opcional): Lista de percentis (0-100). Default: [90, 95, 99].
        """
        if percentiles is None:
            self.percentiles = [90, 95, 99]
        else:
            self.percentiles = sorted(percentiles)

        # Validar range
        for p in self.percentiles:
            if not 0 <= p <= 100:
                raise ValueError(
                    f"Percentil invÃ¡lido: {p}. Todos devem estar entre 0 e 100."
                )

    def apply_dynamic_thresholds(self, df, score_col, model_name, calibration_scores=None):
        """
        Calcula múltiplos thresholds baseados em percentis e cria labels.

        IMPORTANTE: Registros com score NaN recebem label NaN (not_scored),
        NUNCA 0 (normal). Isso garante que ausência de avaliação não seja
        confundida com normalidade.

        Args:
            df: DataFrame com os scores.
            score_col: Nome da coluna com scores brutos.
            model_name: Nome do modelo.
            calibration_scores (np.ndarray, optional): Se fornecido, os thresholds
                são calculados nestes scores (tipicamente do split de treino).
        Returns:
            tuple: (DataFrame atualizado, lista de dicts com estatísticas)
        """
        if score_col not in df.columns:
            raise ValueError(f"Coluna '{score_col}' não encontrada no DataFrame")

        all_scores = df[score_col].dropna().values
        not_scored_count = df[score_col].isna().sum()

        if calibration_scores is not None:
            cal_scores = calibration_scores[~np.isnan(calibration_scores)]
            logger.info(f"Thresholds calibrados no TREINO para {model_name}:")
            logger.info(
                f"   Calibração: {len(cal_scores):,} scores | "
                f"Aplicação: {len(all_scores):,} scores | "
                f"Não avaliados (NaN): {not_scored_count:,}"
            )
        else:
            cal_scores = all_scores
            logger.info(f"Testando percentis para {model_name} (sem split):")

        if not_scored_count > 0:
            pct_not_scored = not_scored_count / len(df) * 100
            logger.warning(
                f"   ATENÇÃO: {not_scored_count:,} registros ({pct_not_scored:.1f}%) "
                f"não foram avaliados por {model_name} e receberão label NaN (not_scored)."
            )

        if len(cal_scores) == 0:
            logger.warning(f"Nenhum score válido para calibração de {model_name}")
            return df, []

        if len(all_scores) == 0:
            logger.warning(f"Nenhum score válido para aplicação de {model_name}")
            return df, []

        logger.info(
            f"Score range (calibração): [{cal_scores.min():.4f}, {cal_scores.max():.4f}]"
        )
        logger.info(
            f"Score range (aplicação):  [{all_scores.min():.4f}, {all_scores.max():.4f}]"
        )

        results = []
        for p in self.percentiles:
            thresh = np.percentile(cal_scores, p)
            col_name = f"{model_name}_p{p}_label"

            # CORREÇÃO CRÍTICA: NaN permanece NaN - não é convertido para 0
            # Registros não avaliados devem ser distinguíveis de registros normais
            df[col_name] = np.where(
                df[score_col].isna(),
                np.nan,  # não avaliado -> NaN
                (df[score_col] >= thresh).astype(float),  # avaliado -> 0.0 ou 1.0
            )

            # Contar apenas os avaliados
            evaluated_mask = df[score_col].notna()
            n_anomalies = (df.loc[evaluated_mask, col_name] == 1.0).sum()
            n_evaluated = evaluated_mask.sum()
            n_total = len(df)

            pct_anomalies_of_evaluated = (
                n_anomalies / n_evaluated * 100
            ) if n_evaluated > 0 else 0
            pct_anomalies_of_total = (n_anomalies / n_total * 100) if n_total > 0 else 0

            results.append({
                "Model": model_name,
                "Percentile": p,
                "Threshold_Value": round(thresh, 6),
                "Anomalies_Detected": int(n_anomalies),
                "N_Evaluated": int(n_evaluated),
                "N_Not_Scored": int(not_scored_count),
                "Pct_Anomalies_Of_Evaluated": round(pct_anomalies_of_evaluated, 2),
                "Pct_Anomalies_Of_Total": round(pct_anomalies_of_total, 2),
                "Calibrated_On": "train" if calibration_scores is not None else "full",
            })
            logger.info(
                f"   P{p}: threshold={thresh:.4f} -> {n_anomalies:,} anomalias "
                f"({pct_anomalies_of_evaluated:.2f}% dos avaliados, "
                f"{pct_anomalies_of_total:.2f}% do total)"
            )
        return df, results


class ModelConcordanceAnalyzer:
    """
    Analisa concordancia entre modelos de deteccao de anomalias.

    IMPORTANTE: Estas metricas medem CONCORDANCIA entre modelos,
    NAO performance contra ground truth real. Sem anotacao humana
    ou validacao externa, nao e possivel medir Precision/Recall reais.
    """

    @staticmethod
    def pairwise_agreement(y_a, y_b):
        """Calcula taxa de concordancia entre dois arrays binarios."""
        agreement = np.mean(y_a == y_b)
        both_anomaly = np.sum((y_a == 1) & (y_b == 1))
        either_anomaly = np.sum((y_a == 1) | (y_b == 1))
        jaccard = both_anomaly / either_anomaly if either_anomaly > 0 else 0.0
        return agreement, jaccard

    @staticmethod
    def analyze_concordance(df, label_cols):
        """
        Calcula concordancia par-a-par entre todos os modelos.

        Args:
            df: DataFrame com colunas de labels binarios.
            label_cols: Lista de colunas de labels para comparar.
        Returns:
            pd.DataFrame com Agreement e Jaccard para cada par.
        """
        results = []
        valid_cols = [c for c in label_cols if c in df.columns]

        for i, col_a in enumerate(valid_cols):
            for col_b in valid_cols[i + 1:]:
                mask = df[[col_a, col_b]].notna().all(axis=1)
                if mask.sum() == 0:
                    continue
                y_a = df.loc[mask, col_a].astype(int).values
                y_b = df.loc[mask, col_b].astype(int).values
                agreement, jaccard = ModelConcordanceAnalyzer.pairwise_agreement(
                    y_a, y_b
                )
                n_a = int(y_a.sum())
                n_b = int(y_b.sum())
                results.append(
                    {
                        "Model_A": col_a,
                        "Model_B": col_b,
                        "Agreement": round(agreement, 4),
                        "Jaccard_Anomalies": round(jaccard, 4),
                        "Anomalies_A": n_a,
                        "Anomalies_B": n_b,
                        "N_Samples": int(mask.sum()),
                        "Type": "CONCORDANCE (not validation)",
                    }
                )

        return pd.DataFrame(results)

