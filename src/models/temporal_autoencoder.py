"""
Modulo unificado de Autoencoder Temporal para deteccao de anomalias sequenciais.
Suporta GRU (default, mais eficiente) e LSTM (para compatibilidade).
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from src.utils.logger_utils import log_execution
from src.utils.logger_utils import logger


class TemporalAutoencoder:
    """
    Pipeline unificado para deteccao de anomalias com Autoencoder Temporal.

    Suporta:
    - GRU (default): 25% menos parametros, mais rapido, ideal para window <= 20
    - LSTM: para janelas longas ou compatibilidade com pipeline legado
    """

    SUPPORTED_ARCHS = ("gru", "lstm")

    def __init__(
        self,
        X_data,
        vehicle_ids,
        timestamps,
        original_indices,
        window_size=5,
        max_gap_seconds=600,
        arch_type="gru",
        arch_config=None,
    ):
        """
        Args:
            X_data: Dados de entrada (numpy array).
            vehicle_ids: IDs dos veiculos.
            timestamps: Timestamps dos registros.
            original_indices: Indices originais do DataFrame.
            window_size: Tamanho da janela temporal (default: 5).
            max_gap_seconds: Gap maximo entre registros consecutivos (default: 600s).
            arch_type: "gru" (default) ou "lstm". GRU recomendado para window <= 20.
        """
        if arch_type not in self.SUPPORTED_ARCHS:
            raise ValueError(
                f"arch_type deve ser {self.SUPPORTED_ARCHS}, recebeu '{arch_type}'"
            )

        self.X = X_data if isinstance(X_data, np.ndarray) else np.array(X_data)
        self.vehicle_ids = np.array(vehicle_ids)
        self.timestamps = pd.to_datetime(timestamps).values
        self.original_indices = np.array(original_indices)
        self.window_size = window_size
        self.max_gap_seconds = max_gap_seconds
        self.arch_type = arch_type
        # Configuração da arquitetura (com defaults)
        default_config = {
            "encoder_units": [2**5, 2**4],
            "decoder_units": [2**4, 2**5],
            "dropout": 0.2,
            "optimizer": "adam",
            "loss": "mse",
        }
        if arch_config is not None:
            default_config.update(arch_config)
        self.arch_config = default_config
        self.model = None
        self._cached_sequences = None
        self._cached_indices = None

        self._LayerClass = keras.layers.GRU if arch_type == "gru" else keras.layers.LSTM
        layer_name = arch_type.upper()

        logger.info("=" * 60)
        logger.info(f"{layer_name} AUTOENCODER - INICIALIZACAO")
        logger.info("=" * 60)
        logger.info(f"Arquitetura: {layer_name}")
        logger.info(f"Dados: {self.X.shape[0]:,} registros x {self.X.shape[1]} features")
        logger.info(f"Veiculos unicos: {len(np.unique(self.vehicle_ids)):,}")
        logger.info(f"Window size: {window_size} timesteps")
        logger.info(f"Max gap: {max_gap_seconds}s ({max_gap_seconds/60:.1f} min)")
        logger.info("=" * 60)

    def create_sequences_with_index(self):
        """
        Gera sequencias temporais respeitando veiculo e continuidade temporal.
        Returns:
            tuple: (np.ndarray[N, window, features], np.ndarray[N] de indices)
        """
        x_seq_list = []
        valid_indices_list = []
        unique_vehicles = np.unique(self.vehicle_ids)
        max_gap_ns = np.timedelta64(self.max_gap_seconds, "s")

        for vehicle in unique_vehicles:
            idx_vehicle = np.where(self.vehicle_ids == vehicle)[0]
            if len(idx_vehicle) < self.window_size:
                continue
            vehicle_data = self.X[idx_vehicle]
            vehicle_times = self.timestamps[idx_vehicle]
            vehicle_indices = self.original_indices[idx_vehicle]
            for i in range(len(vehicle_data) - self.window_size + 1):
                window_times = vehicle_times[i: i + self.window_size]
                gaps = window_times[1:] - window_times[:-1]
                if np.any(gaps > max_gap_ns):
                    continue
                seq = vehicle_data[i: i + self.window_size]
                x_seq_list.append(seq)
                original_idx = vehicle_indices[i + self.window_size - 1]
                valid_indices_list.append(original_idx)

        if len(x_seq_list) == 0:
            return np.array([]), np.array([])
        return np.array(x_seq_list), np.array(valid_indices_list)

    def _get_or_create_sequences(self):
        """Retorna sequencias cacheadas. Gera apenas na primeira chamada."""
        if self._cached_sequences is None:
            logger.info("   -> [CACHE] Gerando sequencias pela primeira vez...")
            self._cached_sequences, self._cached_indices = self.create_sequences_with_index()
            if len(self._cached_sequences) > 0:
                logger.info(
                    f"   -> [CACHE] {len(self._cached_sequences):,} sequencias cacheadas"
                )
        else:
            logger.info(
                f"   -> [CACHE] Reutilizando {len(self._cached_sequences):,} sequencias"
            )
        return self._cached_sequences, self._cached_indices

    def _build_model(self, n_features):
        """Constrói o autoencoder com a arquitetura configurável."""
        layer = self._LayerClass
        arch = self.arch_type.upper()
        cfg = self.arch_config

        enc_units = cfg["encoder_units"]
        dec_units = cfg["decoder_units"]
        dropout = cfg["dropout"]

        layers = []

        # Encoder
        for i, units in enumerate(enc_units):
            is_last = i == len(enc_units) - 1
            layer_kwargs = {
                "activation": "tanh",
                "return_sequences": not is_last,
                "name": f"encoder_{arch}{i+1}" + ("_bottleneck" if is_last else ""),
            }
            if i == 0:
                layer_kwargs["input_shape"] = (self.window_size, n_features)
            layers.append(layer(units, **layer_kwargs))
            if i < len(enc_units) - 1:
                layers.append(keras.layers.Dropout(dropout, name=f"encoder_dropout_{i+1}"))

        # Bridge
        layers.append(keras.layers.RepeatVector(self.window_size, name="repeat"))

        # Decoder
        for i, units in enumerate(dec_units):
            layers.append(
                layer(
                    units,
                    activation="tanh",
                    return_sequences=True,
                    name=f"decoder_{arch}{i+1}",
                )
            )
            if i < len(dec_units) - 1:
                layers.append(keras.layers.Dropout(dropout, name=f"decoder_dropout_{i+1}"))

        # Output
        layers.append(
            keras.layers.TimeDistributed(
                keras.layers.Dense(n_features, activation=None),
                name="output",
            )
        )

        model = keras.Sequential(layers)
        model.compile(optimizer=cfg["optimizer"], loss=cfg["loss"])
        return model

    @log_execution
    def train_evaluate(self, strategy_name, mask_train=None, epochs=10, batch_size=64):
        """
        Treina e avalia o autoencoder temporal.

        Args:
            strategy_name: Nome da estrategia.
            mask_train: Mascara booleana para treino semi-supervisionado.
            epochs: Epocas de treino.
            batch_size: Tamanho do batch.
        Returns:
            tuple: (MSE scores, indices, modelo) ou (None, None, None)
        """
        x_seq_all, indices_all = self._get_or_create_sequences()
        if len(x_seq_all) == 0:
            logger.warning(f"Nenhuma sequencia valida para {strategy_name}")
            return None, None, None

        if mask_train is not None:
            mask_series = pd.Series(mask_train, index=self.original_indices)
            train_mask = mask_series.loc[indices_all].values.astype(bool)
            x_train = x_seq_all[train_mask]
            if len(x_train) == 0:
                logger.warning("Treino vazio apos filtro de mascara.")
                return None, None, None
        else:
            x_train = x_seq_all

        n_features = self.X.shape[1]
        self.model = self._build_model(n_features)

        es = keras.callbacks.EarlyStopping(
            monitor="loss", patience=3, mode="min", restore_best_weights=True
        )
        logger.info(
            f"   -> Treinando {strategy_name} ({self.arch_type.upper()}) com {len(x_train)} sequencias..."
        )
        self.model.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0,
        )

        x_pred = self.model.predict(x_seq_all, verbose=0)
        mse_sequences = np.mean(np.power(x_seq_all - x_pred, 2), axis=(1, 2))

        return mse_sequences, indices_all, self.model
