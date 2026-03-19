# src/data/data_processor.py

import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import StandardScaler
import joblib as _joblib
from src.data.schema import validate_input
from src.utils.logger_utils import log_execution
from src.utils.logger_utils import logger

class DataProcessor:
    """
    Classe responsável pelo processamento e engenharia de dados.
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
        Carrega e padroniza o DataFrame de entrada.
        """
        if filepath.endswith(".parquet"):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, low_memory=False)

        df = df.rename(columns=self.map_cols)

        # Garantir tipagem quando a coluna existir
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Corrigido: usar nome padronizado das colunas, não o mapeamento original
        for col in ["latitude", "longitude"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        if {"timestamp", "placa"}.issubset(df.columns):
            df = df.dropna(subset=["timestamp", "placa"])

        # Validar schema de entrada
        try:
            df = validate_input(df)
            logger.info("✅ Schema de entrada validado com sucesso")
        except Exception as e:
            logger.error(f"❌ Falha na validação de schema: {e}")
            logger.error("Verifique se os dados de entrada seguem o formato esperado.")
            raise

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
        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)

        self.features_to_use.extend(["hora_sin", "hora_cos", "dia_sem"])

        # 2. Features de Contexto (Feriados)
        br_holidays = holidays.BR(state="DF")

        df["eh_feriado"] = df["timestamp"].dt.date.apply(
            lambda d: 1 if d in br_holidays else 0
        )

        self.features_to_use.append("eh_feriado")

        # 3. Features Espaciais (Velocidade/Aceleração)
        if "latitude" in df.columns and "longitude" in df.columns:
            # Sort para garantir ordem temporal
            df = df.sort_values(["placa", "timestamp"])

            df["lat_prev"] = df.groupby("placa")["latitude"].shift(1)
            df["lon_prev"] = df.groupby("placa")["longitude"].shift(1)
            df["time_prev"] = df.groupby("placa")["timestamp"].shift(1)

            # Calcular distância Haversine
            dist = self._haversine_vectorized(
                df["latitude"].values,
                df["longitude"].values,
                df["lat_prev"].fillna(df["latitude"]).values,
                df["lon_prev"].fillna(df["longitude"]).values
            )
            df["dist_m"] = pd.Series(dist, index=df.index).fillna(0)

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
            df["vel_prev"] = df.groupby("placa")["velocidade_kmh"].shift(1)
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

    def fit_scaler(self, df, output_path="outputs/models_saved/scaler.joblib"):
        """
        Ajusta o StandardScaler APENAS nos dados de treino e serializa para disco.
        DEVE ser chamado APENAS com dados de treino (split temporal).

        Args:
            df (pd.DataFrame): DataFrame de TREINO.
            output_path (str): Caminho para salvar o scaler.
        Returns:
            pd.DataFrame: DataFrame de treino com features normalizadas.
        """
        if not self.features_to_use:
            raise ValueError(
                "features_to_use está vazio. Execute feature_engineering() primeiro."
            )

        existing_features = [f for f in self.features_to_use if f in df.columns]

        self.scaler = StandardScaler()
        df[existing_features] = self.scaler.fit_transform(df[existing_features])

        # Serializar o scaler para garantir consistência na inferência
        _joblib.dump(self.scaler, output_path)
        logger.info(
            f"   ✅ Scaler ajustado em {len(existing_features)} features e salvo em {output_path}"
        )
        logger.info(
            f"   📊 Médias: {dict(zip(existing_features, self.scaler.mean_.round(4)))}"
        )
        logger.info(
            f"   📊 Desvios: {dict(zip(existing_features, self.scaler.scale_.round(4)))}"
        )

        return df

    def transform_scaler(self, df, scaler_path=None):
        """
        Aplica o scaler já ajustado aos dados (teste ou novos dados).
        Se scaler não estiver carregado, tenta ler do disco.

        Args:
            df (pd.DataFrame): DataFrame para transformar.
            scaler_path (str, optional): Caminho do scaler serializado.
        Returns:
            pd.DataFrame: DataFrame com features normalizadas.
        """
        if self.scaler is None:
            if scaler_path is None:
                scaler_path = "outputs/models_saved/scaler.joblib"
            self.scaler = _joblib.load(scaler_path)
            logger.info(f"   📂 Scaler carregado de {scaler_path}")

        existing_features = [f for f in self.features_to_use if f in df.columns]
        df[existing_features] = self.scaler.transform(df[existing_features])

        return df
