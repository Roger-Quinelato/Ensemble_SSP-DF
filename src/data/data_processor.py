# src/data/data_processor.py

import pandas as pd
import numpy as np
import yaml
import dask.dataframe as dd
import dask.array as da
import holidays
from src.utils.logger_utils import log_execution

# Tenta carregar config de forma segura
try:
    with open("config_mapeamento.yaml", "r") as config_file:
        CONFIG = yaml.safe_load(config_file)
except Exception:
    CONFIG = {}


class DataProcessor:
    """
    Classe responsável pelo processamento e engenharia de dados.
    """

    def __init__(self, config):
        self.config = config
        self.features_to_use = []
        self.scaler = None

    @property
    def map_cols(self):
        return self.config["mapeamento_colunas"]

    @log_execution
    def load_and_standardize(self, filepath):
        """
        Carrega e padroniza o DataFrame de entrada.
        """
        if filepath.endswith(".parquet"):
            df = dd.read_parquet(filepath)
        else:
            df = dd.read_csv(filepath, assume_missing=True)

        df = df.rename(columns=self.map_cols)

        # Garantir Tipagem
        df["timestamp"] = dd.to_datetime(df["timestamp"])

        # Corrigido: usar nome padronizado das colunas, não o mapeamento original
        for col in ["latitude", "longitude"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df = df.dropna(subset=["timestamp", "placa"])

        return df

    def _haversine_vectorized(self, lat1, lon1, lat2, lon2):
        """Calcula a distância Haversine em metros."""
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        )
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @log_execution
    def feature_engineering(self, df):
        """
        Cria features numéricas e aplica One-Hot Encoding (Get Dummies).
        """
        self.features_to_use = []  # Reinicia a lista

        # 1. Features Temporais
        df["hora"] = df["timestamp"].dt.hour
        df["dia_sem"] = df["timestamp"].dt.dayofweek

        # Ciclicidade de hora
        if hasattr(df, "map_partitions"):
            df["hora_sin"] = df["hora"].map_partitions(
                lambda x: np.sin(2 * np.pi * x / 24), meta=("hora", "float64")
            )
            df["hora_cos"] = df["hora"].map_partitions(
                lambda x: np.cos(2 * np.pi * x / 24), meta=("hora", "float64")
            )
        else:
            df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
            df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)

        self.features_to_use.extend(["hora_sin", "hora_cos", "dia_sem"])

        # 2. Features de Contexto (Feriados)
        br_holidays = holidays.BR(state="DF")

        def is_holiday_func(ts_series):
            return ts_series.dt.date.apply(lambda x: 1 if x in br_holidays else 0)

        if hasattr(df, "map_partitions"):
            df["eh_feriado"] = df["timestamp"].map_partitions(
                is_holiday_func, meta=("timestamp", "int64")
            )
        else:
            df["eh_feriado"] = df["timestamp"].dt.date.apply(
                lambda d: 1 if d in br_holidays else 0
            )

        self.features_to_use.append("eh_feriado")

        # 3. Features Espaciais (Velocidade/Aceleração)
        if "latitude" in df.columns and "longitude" in df.columns:
            # Sort para garantir ordem temporal
            df = df.sort_values(["placa", "timestamp"])
            
            # Shift com meta especificado para evitar warnings do Dask
            df["lat_prev"] = df.groupby("placa")["latitude"].shift(1, meta=("latitude", "float64"))
            df["lon_prev"] = df.groupby("placa")["longitude"].shift(1, meta=("longitude", "float64"))
            df["time_prev"] = df.groupby("placa")["timestamp"].shift(1, meta=("timestamp", "datetime64[ns]"))

            # Calcular distância
            dist = self._haversine_vectorized(
                df["latitude"], df["longitude"], df["lat_prev"], df["lon_prev"]
            )

            if hasattr(dist, "fillna"):
                df["dist_m"] = dist.fillna(0)
            else:
                df["dist_m"] = da.from_array(np.array(dist), chunks='auto').fillna(0)

            # Calcular tempo decorrido
            df["delta_time_s"] = (
                (df["timestamp"] - df["time_prev"]).dt.total_seconds().fillna(0)
            )
            df["delta_time_h"] = df["delta_time_s"] / 3600

            # Velocidade em km/h
            df["velocidade_kmh"] = (df["dist_m"] / 1000) / df["delta_time_h"].replace(
                0, np.nan
            )
            df["velocidade_kmh"] = df["velocidade_kmh"].fillna(0)
            
            self.features_to_use.extend(["velocidade_kmh", "dist_m"])

            # Aceleração
            df["vel_prev"] = df.groupby("placa")["velocidade_kmh"].shift(1, meta=("velocidade_kmh", "float64"))
            df["aceleracao"] = (
                (df["velocidade_kmh"] - df["vel_prev"]) * 1000 / 3600
            ) / df["delta_time_s"].replace(0, np.nan)
            df["aceleracao"] = df["aceleracao"].fillna(0)

            self.features_to_use.append("aceleracao")

            # Limpar colunas temporárias
            df = df.drop(columns=["lat_prev", "lon_prev", "time_prev", "vel_prev", 
                                   "delta_time_s", "delta_time_h"], errors="ignore")

        # 4. Região Administrativa (COM GET_DUMMIES)
        if "regiao_adm" in df.columns:
            # Para Dask: precisa categorizar E especificar as categorias conhecidas
            if hasattr(df, "map_partitions"):
                # Primeiro, compute as categorias únicas (necessário para Dask)
                categorias_unicas = df["regiao_adm"].unique().compute()
                
                # Converte para categorical com categorias conhecidas
                df["regiao_adm"] = df["regiao_adm"].astype(
                    pd.CategoricalDtype(categories=categorias_unicas)
                )
                
                # Agora get_dummies funciona
                df = dd.get_dummies(df, columns=["regiao_adm"], prefix="RA")
            else:
                # Para Pandas (mais simples)
                df = pd.get_dummies(df, columns=["regiao_adm"], prefix="RA")

            # Identificar novas colunas criadas
            new_cols = [c for c in df.columns if c.startswith("RA_")]
            self.features_to_use.extend(new_cols)

        # Drop NaNs apenas nas colunas essenciais
        if self.features_to_use:
            # Verificar quais features existem no DataFrame
            existing_features = [f for f in self.features_to_use if f in df.columns]
            if existing_features:
                df = df.dropna(subset=existing_features)

        return df