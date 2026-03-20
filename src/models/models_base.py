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
    def train_iso(self, n_estimators=100, contamination="auto"):
        """
        Treina Isolation Forest no conjunto X e retorna o modelo treinado.

        O scoring (score_samples) e responsabilidade do caller, que deve
        aplicar o modelo ao conjunto completo (treino + val + test).
        Retornar labels/scores aqui seria redundante quando o caller descarta.

        Args:
            n_estimators (int): Numero de arvores.
            contamination (str ou float): Proporcao de anomalias.
        Returns:
            IsolationForest treinado.
        """
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            n_jobs=-1,
            random_state=self.random_state,
        )
        model.fit(self.X)
        return model

    @log_execution
    def train_hbos(self, n_bins=10, contamination=0.1):
        """
        Treina HBOS no conjunto X e retorna o modelo treinado.

        O scoring e responsabilidade do caller.

        Args:
            n_bins (int): Numero de bins do histograma.
            contamination (float): Proporcao de anomalias.
        Returns:
            HBOS treinado.
        """
        model = HBOS(n_bins=n_bins, contamination=contamination)
        model.fit(self.X)
        return model
