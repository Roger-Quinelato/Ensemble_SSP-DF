# src/evaluation.py

import numpy as np
import pandas as pd
from .logger_utils import log_execution

class ThresholdOptimizer:
    """
    Otimiza thresholds de detecção usando múltiplos percentis.
    Útil para balancear Precision vs Recall em detecção de anomalias.
    """
    def __init__(self, percentiles):
        """
        Inicializa o otimizador de thresholds para detecção de anomalias.
        Args:
            percentiles (list): Lista de percentis para testar.
        """
        """
        Inicializa o otimizador de thresholds para detecção de anomalias.
        Args:
            percentiles (list): Lista de percentis para testar. Ex: [90, 95, 99]
        """
        self.percentiles = sorted(percentiles)  # Garantir ordem crescente
        
    def apply_dynamic_thresholds(self, df, score_col, model_name):
        """
        Calcula múltiplos thresholds baseados em percentis e cria labels para cada variação.
        Suporta Dask DataFrame.
        Args:
            df (pd.DataFrame ou dask.dataframe.DataFrame): DataFrame com os scores.
            score_col (str): Nome da coluna com scores brutos.
            model_name (str): Nome do modelo (para nomear colunas de output).
        Returns:
            tuple: (DataFrame atualizado, lista de dicts com estatísticas por percentil)
        """
        """
        Calcula múltiplos thresholds baseados em percentis e cria labels para cada variação.
        Suporta Dask DataFrame.
        """
        import dask.dataframe as dd
        if score_col not in df.columns:
            raise ValueError(f"❌ Coluna '{score_col}' não encontrada no DataFrame")
        # Se for Dask, converte para pandas para cálculo de percentis
        if isinstance(df, dd.DataFrame):
            scores = df[score_col].dropna().compute().values
        else:
            scores = df[score_col].dropna().values
        if len(scores) == 0:
            print(f"   ⚠️ Nenhum score válido para {model_name}")
            return df, []
        results = []
        print(f"\n   📊 Testando percentis para {model_name}:")
        print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}")
        for p in self.percentiles:
            thresh = np.percentile(scores, p)
            col_name = f"{model_name}_p{p}_label"
            df[col_name] = (df[score_col] >= thresh).astype(int)
            if isinstance(df, dd.DataFrame):
                n_anomalies = df[col_name].sum().compute()
                total_len = len(df)
            else:
                n_anomalies = df[col_name].sum()
                total_len = len(df)
            pct_anomalies = (n_anomalies / total_len) * 100
            results.append({
                'Model': model_name,
                'Percentile': p,
                'Threshold_Value': round(thresh, 6),
                'Anomalies_Detected': int(n_anomalies),
                'Pct_Dataset': round(pct_anomalies, 2)
            })
            print(f"      P{p}: threshold={thresh:.4f} → {n_anomalies:,} anomalias ({pct_anomalies:.2f}%)")
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
                print(f"   ⚠️ Coluna {col} não encontrada, pulando...")
                continue
            
            metrics = GroundTruthComparator.compute_metrics(
                df[ground_truth_col].values,
                df[col].values,
                model_name=col
            )
            results.append(metrics)
        
        return pd.DataFrame(results)