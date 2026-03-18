# src/evaluation.py

import numpy as np
import pandas as pd
from src.utils.logger_utils import log_execution
from src.utils.logger_utils import logger

class ThresholdOptimizer:
    """
    Otimiza thresholds de detecção usando múltiplos percentis.
    Útil para balancear Precision vs Recall em detecção de anomalias.
    """
    def __init__(self, percentiles=None):
        """
        Inicializa o otimizador de thresholds para detecção de anomalias.
        Args:
            percentiles (list, opcional): Lista de percentis para testar. Se None, usa padrão.
        """
        # Padrão: ISO = 200, HBOS = 20, mas permite variantes
        if percentiles is None:
            self.percentiles = sorted([90, 95, 99, 200, 20])
        else:
            self.percentiles = sorted(percentiles)
        
    def apply_dynamic_thresholds(self, df, score_col, model_name, calibration_scores=None):
        """
        Calcula múltiplos thresholds baseados em percentis e cria labels.

        Args:
            df: DataFrame com os scores.
            score_col: Nome da coluna com scores brutos.
            model_name: Nome do modelo.
            calibration_scores (np.ndarray, optional): Se fornecido, os thresholds
                são calculados nestes scores (tipicamente do split de treino) em vez
                dos scores do DataFrame completo. Isso evita data leakage.
        Returns:
            tuple: (DataFrame atualizado, lista de dicts com estatísticas)
        """
        if score_col not in df.columns:
            raise ValueError(f"❌ Coluna '{score_col}' não encontrada no DataFrame")

        # Scores para APLICAR labels (dataset completo)
        all_scores = df[score_col].dropna().values

        # Scores para CALIBRAR thresholds (treino ou completo se não especificado)
        if calibration_scores is not None:
            cal_scores = calibration_scores[~np.isnan(calibration_scores)]
            logger.info(f"📊 Thresholds calibrados no TREINO para {model_name}:")
            logger.info(
                f"   Calibração: {len(cal_scores):,} scores | Aplicação: {len(all_scores):,} scores"
            )
        else:
            cal_scores = all_scores
            logger.info(f"📊 Testando percentis para {model_name} (sem split):")

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
            # Threshold calculado nos dados de CALIBRAÇÃO (treino)
            thresh = np.percentile(cal_scores, p)

            col_name = f"{model_name}_p{p}_label"
            # Labels aplicados no dataset COMPLETO
            df[col_name] = (df[score_col] >= thresh).astype(int)

            n_anomalies = df[col_name].sum()
            total_len = len(df)

            pct_anomalies = (n_anomalies / total_len) * 100
            results.append({
                'Model': model_name,
                'Percentile': p,
                'Threshold_Value': round(thresh, 6),
                'Anomalies_Detected': int(n_anomalies),
                'Pct_Dataset': round(pct_anomalies, 2),
                'Calibrated_On': 'train' if calibration_scores is not None else 'full',
            })
            logger.info(
                f"   P{p}: threshold={thresh:.4f} → {n_anomalies:,} anomalias ({pct_anomalies:.2f}%)"
            )
        return df, results


class GroundTruthComparator:
    """
    Compara predições dos modelos contra ground truths (União e Interseção).
    Calcula métricas de classificação.
    """
    @staticmethod
    def compute_metrics(y_true, y_pred, model_name):
        """
        Calcula métricas de classificação (Precision, Recall, F1, Accuracy) para um modelo.
        Args:
            y_true (np.ndarray): Array binário (ground truth).
            y_pred (np.ndarray): Array binário (predições).
            model_name (str): Nome do modelo.
        Returns:
            dict: Métricas calculadas.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        # Evitar divisão por zero
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'Model': model_name,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1_Score': round(f1, 4),
            'Accuracy': round(accuracy, 4)
        }
    
    @staticmethod
    def compare_all_models(df, ground_truth_col, label_cols):
        """
        Compara múltiplos modelos contra um ground truth e retorna métricas de todos.
        Args:
            df (pd.DataFrame): DataFrame com todas as colunas.
            ground_truth_col (str): Nome da coluna de GT.
            label_cols (list): Lista de colunas de predições para comparar.
        Returns:
            pd.DataFrame: DataFrame com métricas de todos os modelos.
        """
        results = []
        
        for col in label_cols:
            if col not in df.columns:
                logger.warning(f"Coluna {col} não encontrada, pulando...")
                continue
            
            metrics = GroundTruthComparator.compute_metrics(
                df[ground_truth_col].values,
                df[col].values,
                model_name=col
            )
            results.append(metrics)
        
        return pd.DataFrame(results)
