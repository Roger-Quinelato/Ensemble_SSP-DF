# src/models_base.py

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from .logger_utils import log_execution

class BaselineModels:
    def __init__(self, X_data):
        """
        Inicializa o BaselineModels com os dados de entrada.
        Args:
            X_data (np.ndarray ou dask.dataframe.DataFrame): Dados normalizados.
        """
        # Se for Dask DataFrame, converte para numpy para modelagem
        if hasattr(X_data, 'compute'):
            self.X = X_data.compute().values if hasattr(X_data, 'values') else X_data.compute()
        else:
            self.X = X_data
        
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
            random_state=42
        )
        preds = model.fit_predict(self.X)
        scores = model.score_samples(self.X)
        # 1: Normal, -1: Anomalia -> Converter para 0: Normal, 1: Anomalia
        labels = np.where(preds == -1, 1, 0)
        return labels, scores, model

    @log_execution
    def train_lof(self, k_neighbors=20, contamination='auto', strategy='standard', mask_inliers=None):
        """
        Treina o modelo Local Outlier Factor (LOF) para detecção de anomalias.
        Args:
            k_neighbors (int): Número de vizinhos.
            contamination (str ou float): Proporção de anomalias.
            strategy (str): 'standard' ou 'novelty'.
            mask_inliers (np.ndarray ou None): Máscara de inliers para treino novelty.
        Returns:
            tuple: (labels, scores)
        """
        """
        strategy: 'standard' (treina em tudo) ou 'novelty' (treina só nos inliers do ISO)
        """
        model = LocalOutlierFactor(
            n_neighbors=k_neighbors,
            contamination=contamination,
            novelty=(strategy == 'novelty'),
            n_jobs=-1
        )
        
        if strategy == 'novelty':
            if mask_inliers is None:
                raise ValueError("Para LOF Novelty, mask_inliers é obrigatório.")
            
            # Treina APENAS nos dados considerados limpos (inliers)
            X_train = self.X[mask_inliers]
            model.fit(X_train)
            
            # Prediz em TUDO
            preds = model.predict(self.X)
            scores = model.score_samples(self.X)
            
        else: # Standard
            preds = model.fit_predict(self.X)
            scores = model.negative_outlier_factor_
            
        labels = np.where(preds == -1, 1, 0)
        return labels, scores