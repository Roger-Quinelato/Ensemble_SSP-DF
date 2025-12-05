# src/models_deep.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from .logger_utils import log_execution

class LSTMPipeline:
    def __init__(self, X_data, vehicle_ids, timestamps, original_indices, window_size=5, max_gap_seconds=600):
        """
        Inicializa o pipeline LSTM para detecção de anomalias sequenciais.
        Suporta Dask DataFrame.
        """
        import dask.dataframe as dd
        if hasattr(X_data, 'compute'):
            self.X = X_data.compute().values if hasattr(X_data, 'values') else X_data.compute()
        else:
            self.X = X_data
        self.vehicle_ids = np.array(vehicle_ids)
        self.timestamps = pd.to_datetime(timestamps).values
        self.original_indices = np.array(original_indices)
        self.window_size = window_size
        self.max_gap_seconds = max_gap_seconds
        self.model = None

    def create_sequences_with_index(self):
        """
        Gera sequências temporais respeitando o mesmo veículo e continuidade temporal.
        Returns:
            tuple: (np.ndarray de sequências, np.ndarray de índices originais ancorados no fim da janela)
        """
        X_seq_list = []
        valid_indices_list = []
        unique_vehicles = np.unique(self.vehicle_ids)
        max_gap_ns = np.timedelta64(self.max_gap_seconds, 's')
        for vehicle in unique_vehicles:
            idx_vehicle = np.where(self.vehicle_ids == vehicle)[0]
            if len(idx_vehicle) <= self.window_size:
                continue
            vehicle_data = self.X[idx_vehicle]
            vehicle_times = self.timestamps[idx_vehicle]
            vehicle_indices = self.original_indices[idx_vehicle]
            for i in range(len(vehicle_data) - self.window_size + 1):
                window_times = vehicle_times[i : i + self.window_size]
                # Verificação rigorosa: nenhum gap consecutivo pode exceder max_gap_ns
                gaps = window_times[1:] - window_times[:-1]
                if np.any(gaps > max_gap_ns):
                    continue
                t_start = window_times[0]
                t_end = window_times[-1]
                # (opcional) manter também a verificação global
                if (t_end - t_start) > max_gap_ns:
                    continue
                seq = vehicle_data[i : i + self.window_size]
                X_seq_list.append(seq)
                original_idx = vehicle_indices[i + self.window_size - 1]
                valid_indices_list.append(original_idx)
        if len(X_seq_list) == 0:
            return np.array([]), np.array([])
        return np.array(X_seq_list), np.array(valid_indices_list)

    @log_execution
    def train_evaluate(self, strategy_name, mask_train=None, epochs=10):
        """
        Treina e avalia um modelo LSTM Autoencoder para detecção de anomalias sequenciais.
        Args:
            strategy_name (str): Nome da estratégia/variação.
            mask_train (np.ndarray or None): Máscara booleana para treino.
            epochs (int): Número de épocas de treino.
        Returns:
            tuple: (np.ndarray de MSE, np.ndarray de índices originais, modelo treinado)
        """
        print(f"   ↳ [LSTM] Gerando sequências (Gap Max: {self.max_gap_seconds}s)...")
        X_seq_all, indices_all = self.create_sequences_with_index()
        if len(X_seq_all) == 0:
            print(f"   ⚠️ Nenhuma sequência válida encontrada para {strategy_name} (Verifique gaps).")
            return None, None, None
        # 2. Filtragem por Máscara (Treino Semi-supervisionado)
        if mask_train is not None:
            mask_series = pd.Series(mask_train, index=self.original_indices)
            train_mask = mask_series.loc[indices_all].values.astype(bool)
            X_train = X_seq_all[train_mask]
            if len(X_train) == 0:
                print(f"   ⚠️ Treino vazio após filtro de máscara.")
                return None, None, None
        else:
            X_train = X_seq_all
        # 3. Modelo e Treino
        n_features = self.X.shape[1]
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(self.window_size, n_features), return_sequences=True),
            LSTM(16, activation='relu', return_sequences=False),
            RepeatVector(self.window_size),
            LSTM(16, activation='relu', return_sequences=True),
            LSTM(32, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='loss', patience=3, mode='min', restore_best_weights=True)
        print(f"   ↳ Treinando {strategy_name} com {len(X_train)} sequências...")
        model.fit(X_train, X_train, epochs=epochs, batch_size=64, callbacks=[es], verbose=0)
        # 4. Inferência
        X_pred = model.predict(X_seq_all, verbose=0)
        mse_sequences = np.mean(np.power(X_seq_all - X_pred, 2), axis=(1, 2))
        # Retorna: O Score (MSE), Os Índices (Onde salvar), O Modelo
        return mse_sequences, indices_all, model