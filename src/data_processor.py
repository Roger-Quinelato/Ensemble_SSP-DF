# src/data_processor.py

import pandas as pd
import numpy as np
import holidays
import yaml
from .logger_utils import log_execution

with open('config_mapeamento.yaml', 'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

class DataProcessor:
    def __init__(self, config):
        """
        Inicializa o DataProcessor com o dicionário de configuração.
        Args:
            config (dict): Dicionário de configuração carregado do YAML.
        """
        self.config = config
        self.features_to_use = []
        self.scaler = None

    @property
    def map_cols(self):
        """
        Retorna o dicionário de mapeamento de colunas do config.
        """
        return self.config['mapeamento_colunas']

    @log_execution
    def load_and_standardize(self, filepath):
        """
        Carrega e padroniza o DataFrame de entrada usando Dask.
        Args:
            filepath (str): Caminho do arquivo CSV ou Parquet.
        Returns:
            dask.dataframe.DataFrame: DataFrame padronizado.
        """
        import dask.dataframe as dd
        # Detecção automática de formato
        if filepath.endswith('.parquet'):
            df = dd.read_parquet(filepath)
        else:
            df = dd.read_csv(filepath, assume_missing=True)
        # Renomear colunas
        df = df.rename(columns=self.map_cols)
        # Garantir Tipagem
        df['timestamp'] = dd.to_datetime(df[self.map_cols['timestamp']])
        # Conversão Lat/Long segura
        for col in [self.map_cols['latitude'], self.map_cols['longitude']]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df = df.dropna(subset=[self.map_cols['timestamp'], self.map_cols['placa']])
        df = df.sort_values(by=[self.map_cols['placa'], self.map_cols['timestamp']])
        return df

    def _haversine_vectorized(self, lat1, lon1, lat2, lon2):
        """
        Calcula a distância Haversine entre pares de coordenadas.
        Args:
            lat1, lon1, lat2, lon2: Arrays de latitude/longitude.
        Returns:
            np.ndarray: Distâncias em km.
        """
        R = 6371  # Raio da Terra em km
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lon2 - lon1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @log_execution
    def feature_engineering(self, df):
        """
        Realiza a engenharia de features temporais, espaciais e contextuais no DataFrame.
        Args:
            df (dask.dataframe.DataFrame): DataFrame de entrada.
        Returns:
            dask.dataframe.DataFrame: DataFrame com novas features.
        """
        import dask.dataframe as dd
        import dask
        # 1. Features Temporais (Seno/Cosseno para ciclicidade)
        df['hora'] = df['timestamp'].dt.hour
        df['dia_sem'] = df['timestamp'].dt.dayofweek
        df['hora_sin'] = df['hora'].map_partitions(lambda x: np.sin(2 * np.pi * x / 24))
        df['hora_cos'] = df['hora'].map_partitions(lambda x: np.cos(2 * np.pi * x / 24))
        self.features_to_use.extend(['hora_sin', 'hora_cos', 'dia_sem'])

        # 2. Features de Contexto (Feriados)
        br_holidays = holidays.BR(state='DF')
        def feriado_func(x):
            def is_holiday(d):
                try:
                    return 1 if pd.to_datetime(d, errors='coerce') in br_holidays else 0
                except Exception:
                    return 0
            return x.apply(is_holiday)
        df['eh_feriado'] = df['timestamp'].dt.date.map_partitions(feriado_func, meta=('eh_feriado', 'int64'))
        self.features_to_use.append('eh_feriado')

        # 3. Features Espaciais (Velocidade/Aceleração)
        if self.map_cols['latitude'] in df.columns and self.map_cols['longitude'] in df.columns:
            # Shift para pegar o ponto anterior do MESMO veículo
            df['lat_prev'] = df.groupby(self.map_cols['placa'])[self.map_cols['latitude']].shift(1)
            df['lon_prev'] = df.groupby(self.map_cols['placa'])[self.map_cols['longitude']].shift(1)
            df['time_prev'] = df.groupby(self.map_cols['placa'])[self.map_cols['timestamp']].shift(1)

            # Distância
            df['dist_km'] = dd.from_array(self._haversine_vectorized(
                df['latitude'].values,
                df['longitude'].values,
                df['lat_prev'].values,
                df['lon_prev'].values
            )).fillna(0)

            # Tempo (em horas)
            df['delta_time_h'] = (df['timestamp'] - df['time_prev']).dt.total_seconds() / 3600
            df['delta_time_h'] = df['delta_time_h'].replace(0, np.nan) # Evitar div por zero
            # Velocidade (km/h)
            df['velocidade_calc'] = df['dist_km'] / df['delta_time_h']
            df['velocidade_calc'] = df['velocidade_calc'].fillna(0)
            # Limpeza de ruído extremo (GPS teleport)
            df = df[df['velocidade_calc'] < 1200]
            self.features_to_use.extend(['velocidade_calc', 'dist_km'])
            # Aceleração (Delta V / Delta T)
            df['vel_prev'] = df.groupby(self.map_cols['placa'])['velocidade_calc'].shift(1)
            df['aceleracao'] = (df['velocidade_calc'] - df['vel_prev']) / df['delta_time_h']
            df['aceleracao'] = df['aceleracao'].fillna(0)
            self.features_to_use.append('aceleracao')

        # Drop NaNs residuais nas features usadas
        df = df.dropna(subset=self.features_to_use)
        return df