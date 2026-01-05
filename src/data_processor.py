# src/data_processor.py

import pandas as pd
import numpy as np
import holidays
import yaml
from .logger_utils import log_execution

with open('config_mapeamento.yaml', 'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

class DataProcessor:
    """
    Classe responsável pelo processamento e engenharia de dados para o pipeline de detecção de anomalias.
    Inclui carregamento, padronização, criação de features temporais, espaciais e contextuais.
    """
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
        Returns:
            dict: Mapeamento de colunas.
        """
        return self.config['mapeamento_colunas']

    @log_execution
    def load_and_standardize(self, filepath):
        """
        Carrega e padroniza o DataFrame de entrada usando Dask.
        Args:
            filepath (str): Caminho do arquivo CSV ou Parquet.
        Returns:
            pd.DataFrame: DataFrame padronizado e ordenado.
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
        # Se for Dask DataFrame, compute antes de ordenar
        if hasattr(df, 'compute'):
            try:
                df = df.compute().sort_values(by=[self.map_cols['placa'], self.map_cols['timestamp']])
            except Exception:
                df = df.sort_values(by=[self.map_cols['placa'], self.map_cols['timestamp']])
        else:
            df = df.sort_values(by=[self.map_cols['placa'], self.map_cols['timestamp']])
        return df

    def _haversine_vectorized(self, lat1, lon1, lat2, lon2):
        """
        Calcula a distância Haversine entre pares de coordenadas.
        Args:
            lat1, lon1, lat2, lon2: Arrays de latitude/longitude.
        Returns:
            np.ndarray: Distâncias em metros.
        """
        R = 6371000  # Raio da Terra em metros
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @log_execution
    def feature_engineering(self, df):
        """
        Realiza a engenharia de features temporais, espaciais e contextuais no DataFrame.
        Inclui hora, dia da semana, feriado, velocidade (m/s), aceleração (m/s²) e prepara para Região Administrativa.
        Args:
            df (pd.DataFrame): DataFrame de entrada.
        Returns:
            pd.DataFrame: DataFrame com novas features.
        """
        import dask.dataframe as dd
        import dask
        # 1. Features Temporais (Seno/Cosseno para ciclicidade)
        df['hora'] = df['timestamp'].dt.hour
        df['dia_sem'] = df['timestamp'].dt.dayofweek
        if hasattr(df['hora'], 'map_partitions'):
            df['hora_sin'] = df['hora'].map_partitions(lambda x: np.sin(2 * np.pi * x / 24))
            df['hora_cos'] = df['hora'].map_partitions(lambda x: np.cos(2 * np.pi * x / 24))
        else:
            df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
            df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
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
        if hasattr(df['timestamp'].dt.date, 'map_partitions'):
            df['eh_feriado'] = df['timestamp'].dt.date.map_partitions(feriado_func, meta=('eh_feriado', 'int64'))
        else:
            df['eh_feriado'] = df['timestamp'].dt.date.apply(lambda d: 1 if pd.to_datetime(d, errors='coerce') in br_holidays else 0)
        self.features_to_use.append('eh_feriado')

        # 3. Features Espaciais (Velocidade/Aceleração em m/s)
        if self.map_cols['latitude'] in df.columns and self.map_cols['longitude'] in df.columns:
            # Shift para pegar o ponto anterior do MESMO veículo
            df['lat_prev'] = df.groupby(self.map_cols['placa'])[self.map_cols['latitude']].shift(1)
            df['lon_prev'] = df.groupby(self.map_cols['placa'])[self.map_cols['longitude']].shift(1)
            df['time_prev'] = df.groupby(self.map_cols['placa'])[self.map_cols['timestamp']].shift(1)

            # Distância em metros
            df['dist_m'] = dd.from_array(self._haversine_vectorized(
                df['latitude'].values,
                df['longitude'].values,
                df['lat_prev'].values,
                df['lon_prev'].values
            )).fillna(0)

            # Tempo (em segundos)
            df['delta_time_s'] = (df['timestamp'] - df['time_prev']).dt.total_seconds()
            df['delta_time_s'] = df['delta_time_s'].replace(0, np.nan) # Evitar div por zero
            # Velocidade (m/s)
            df['velocidade_ms'] = df['dist_m'] / df['delta_time_s']
            df['velocidade_ms'] = df['velocidade_ms'].fillna(0)
            # Limpeza de ruído extremo (GPS teleport)
            df = df[df['velocidade_ms'] < 333]  # 1200 km/h = 333 m/s
            self.features_to_use.extend(['velocidade_ms', 'dist_m'])
            # Aceleração (Delta V / Delta T)
            df['vel_prev'] = df.groupby(self.map_cols['placa'])['velocidade_ms'].shift(1)
            df['aceleracao'] = (df['velocidade_ms'] - df['vel_prev']) / df['delta_time_s']
            df['aceleracao'] = df['aceleracao'].fillna(0)
            self.features_to_use.append('aceleracao')

        # 4. Região Administrativa (placeholder para futura integração)
        if 'regiao_adm' in df.columns:
            self.features_to_use.append('regiao_adm')
        # Drop NaNs residuais nas features usadas
        df = df.dropna(subset=self.features_to_use)
        return df