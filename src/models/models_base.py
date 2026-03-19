# src/models_base.py

from sklearn.ensemble import IsolationForest
from pyod.models.hbos import HBOS
import numpy as np
from src.utils.logger_utils import log_execution


class BaselineModels:
    def __init__(self, X_data, random_state=None):
        """
        Inicializa o BaselineModels com os dados de entrada.
        Args:
            X_data (np.ndarray ou pandas.DataFrame): Dados normalizados.
            random_state (int): Seed para reprodutibilidade.
        """
        self.X = X_data if isinstance(X_data, np.ndarray) else X_data.values
        self.random_state = 42 if random_state is None else random_state

    @log_execution
    def train_iso(self, n_estimators=100, contamination='auto'):
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
            random_state=self.random_state,
        )
        model.fit(self.X)
        preds = model.predict(self.X)
        scores = model.score_samples(self.X)
        labels = np.where(preds == -1, 1, 0)
        return labels, scores, model

    @log_execution
    def train_hbos(self, n_bins=10, contamination=0.1):
        """
        Treina o modelo HBOS para detecção de anomalias.
        Args:
            n_bins (int): Número de bins do histograma.
            contamination (float): Proporção de anomalias.
        Returns:
            tuple: (labels, scores, modelo treinado)
        """
        model = HBOS(n_bins=n_bins, contamination=contamination)
        model.fit(self.X)
        scores = model.decision_scores_
        labels = model.labels_
        return labels, scores, model
