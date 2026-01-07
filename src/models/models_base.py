# src/models_base.py

from sklearn.ensemble import IsolationForest
from pyod.models.hbos import HBOS
import numpy as np
from src.utils.logger_utils import log_execution


class BaselineModels:
    def __init__(self, X_data):
        """
        Inicializa o BaselineModels com os dados de entrada.
        Args:
            X_data (np.ndarray ou dask.dataframe.DataFrame): Dados normalizados.
        """
        # Se for Dask DataFrame, converte para numpy para modelagem
        if hasattr(X_data, "compute"):
            self.X = (
                X_data.compute().values
                if hasattr(X_data, "values")
                else X_data.compute()
            )
        else:
            self.X = X_data

    @log_execution
    def train_iso(self, n_estimators=100, contamination="auto"):
        """
        Treina o modelo Isolation Forest para detecção de anomalias.
        Args:
            n_estimators (int): Número de árvores.
            contamination (str ou float): Proporção de anomalias.
        Returns:
            tuple: (labels, scores, modelo treinado)
        """
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            n_jobs=-1,
            random_state=42,
        )
        preds = model.fit_predict(self.X)
        scores = model.score_samples(self.X)
        # 1: Normal, -1: Anomalia -> Converter para 0: Normal, 1: Anomalia
        labels = np.where(preds == -1, 1, 0)
        return labels, scores, model

    @log_execution
    def train_hbos(self, n_bins=10, contamination="auto"):
        """
        Treina o modelo Histogram-Based Outlier Score (HBOS) para detecção de anomalias.
        Args:
            n_bins (int): Número de bins do histograma.
            contamination (float): Proporção de anomalias.
        Returns:
            tuple: (labels, scores, modelo treinado)
        """
        results = {}
        for n_bins in [10, 20, 30]:
            model = HBOS(n_bins=n_bins, contamination=contamination)
            model.fit(self.X)
            scores = model.decision_scores_
            labels = model.labels_
            results[n_bins] = {"labels": labels, "scores": scores, "model": model}
        return results
