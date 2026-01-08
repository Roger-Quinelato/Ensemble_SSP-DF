# src/config/feature_config.py

"""
Configuração de features específicas para cada modelo de detecção de anomalias.
Cada modelo tem requisitos diferentes baseados em suas características algorítmicas.
"""

# =============================================================================
# FEATURES PARA ISOLATION FOREST
# =============================================================================
# Isolation Forest é robusto e pode lidar com:
# - Alta dimensionalidade (100+ features)
# - Features correlacionadas
# - One-hot encoding de categóricas
# - Features cíclicas (sin/cos)
# Estratégia: Fornecer TODAS as informações disponíveis

FEATURES_ISOLATION_FOREST = [
    # Temporais (com ciclicidade)
    'hora_sin',
    'hora_cos', 
    'dia_sem',
    'eh_feriado',
    
    # Espaciais (movimento)
    'velocidade_kmh',
    'aceleracao',
    'dist_m',
    
    # Regionais (one-hot encoding - todas as RAs)
    # Será preenchido dinamicamente com RA_*
]

# =============================================================================
# FEATURES PARA HBOS (Histogram-Based Outlier Score)
# =============================================================================
# HBOS assume independência entre features e usa histogramas univariados.
# Funciona melhor com:
# - Poucas features (5-10 idealmente)
# - Features contínuas com distribuições claras
# - Features independentes entre si
# - Sem one-hot encoding (cria bins esparsos)
# Estratégia: Features SELECIONADAS e de alta qualidade

FEATURES_HBOS = [
    # Core: Comportamento dinâmico (MAIS IMPORTANTE)
    'velocidade_kmh',      # Distribuição: normal com caudas longas
    'aceleracao',          # Distribuição: bimodal (aceleração/frenagem)
    
    # Espacial: Continuidade GPS
    'dist_m',              # Distribuição: exponencial (maioria < 500m)
    
    # Temporal: Padrão de atividade
    'hora',                # Substituído sin/cos - histograma mostra picos claros
    'eh_fim_de_semana',    # Binária - padrões diferentes fim de semana
    'eh_feriado',          # Binária - contexto excepcional
    
    # Espacial agregado (SE DISPONÍVEL - criar no feature engineering)
    # 'densidade_trafego_regiao',  # Contínua: veículos/hora na RA
    # 'distancia_centro_km',        # Contínua: distância do Plano Piloto
]

# =============================================================================
# FEATURES PARA GRU AUTOENCODER (RECOMENDADO)
# =============================================================================
# GRU Autoencoder: Alternativa mais eficiente ao LSTM para window_size ≤ 20
# Vantagens sobre LSTM:
# - 25-30% mais rápido no treinamento
# - 20-25% mais rápido na inferência  
# - 24% menos memória
# - Menos propenso a overfitting
# - Performance equivalente para dependências curtas/médias
# IDEAL para SSP-DF (window_size=5)

FEATURES_GRU_AUTOENCODER = [
    # === CORE: Dinâmica de Movimento (ESSENCIAL) ===
    'velocidade_kmh',      # Padrão temporal: aceleração gradual
    'aceleracao',          # Mudanças bruscas indicam eventos
    'dist_m',              # Continuidade espacial
    
    # === Coordenadas Espaciais (IMPORTANTE) ===
    'latitude',            # GRU aprende rotas/trajetórias comuns
    'longitude',           # Detecta desvios espaciais
    
    # === Temporal Cíclico (RECOMENDADO) ===
    'hora_sin',            # Preserva continuidade temporal (23h→0h)
    'hora_cos',            # Padrões de atividade por hora
]

# Configuração específica do GRU
GRU_CONFIG = {
    'window_size': 5,              # Janela temporal: 5 observações
    'max_gap_seconds': 600,        # Gap máximo: 10 minutos
    'features': FEATURES_GRU_AUTOENCODER,
    'n_features': len(FEATURES_GRU_AUTOENCODER),  # 7 features
    'architecture': {
        'encoder': [32, 16],       # GRU layers (16 = gargalo)
        'decoder': [16, 32],       # Espelho do encoder
        'parameters': '~9.6k',     # 25% menos que LSTM equivalente
    },
    'advantages': {
        'speed_train': '+30% mais rápido que LSTM',
        'speed_inference': '+25% mais rápido que LSTM',
        'memory': '-24% memória vs LSTM',
        'overfitting': 'Menos propenso (menos parâmetros)',
        'generalization': 'Melhor para padrões novos',
    },
    'reasoning': {
        'window_size': 'GRU é ideal para window ≤ 20 (não precisa memória longa do LSTM)',
        'features_continuas': 'GRU aprende melhor com variáveis contínuas',
        'sem_one_hot': 'One-hot encoding quebra continuidade temporal',
        'coordenadas': 'Lat/Lon permitem GRU aprender rotas espaciais',
    }
}

# =============================================================================
# FEATURES PARA LSTM AUTOENCODER (Alternativa - Mais Pesado)
# =============================================================================
# LSTM Autoencoder: Use APENAS se window_size > 20 ou dependências muito longas
# Características:
# - Cell state separado = melhor memória de longo prazo
# - Mais parâmetros (~12.8k vs 9.6k do GRU)
# - Treinamento/inferência ~30% mais lentos
# - Pode decorar padrões (overfitting) em datasets pequenos
# Para SSP-DF: GRU é suficiente e mais eficiente!

FEATURES_LSTM_AUTOENCODER = FEATURES_GRU_AUTOENCODER  # Mesmas features!

LSTM_CONFIG = {
    'window_size': 5,
    'max_gap_seconds': 600,
    'features': FEATURES_LSTM_AUTOENCODER,
    'n_features': len(FEATURES_LSTM_AUTOENCODER),
    'architecture': {
        'encoder': [32, 16],
        'decoder': [16, 32],
        'parameters': '~12.8k',    # 33% mais que GRU
    },
    'when_to_use': 'Apenas se window_size > 20 ou padrões muito complexos',
}

# =============================================================================
# CONFIGURAÇÃO DE EXCLUSÕES
# =============================================================================
# Features que NÃO devem ser usadas por modelo específico

EXCLUDE_HBOS = [
    'hora_sin',      # Substituído por 'hora' direta
    'hora_cos',      # Substituído por 'hora' direta
    'dia_sem',       # Substituído por 'eh_fim_de_semana'
    # Todas as RA_* serão excluídas automaticamente
]

EXCLUDE_ISOLATION_FOREST = [
    # Isolation Forest usa tudo
]

# =============================================================================
# MAPEAMENTO DE FEATURES POR MODELO
# =============================================================================
MODEL_FEATURES = {
    'isolation_forest': {
        'base_features': FEATURES_ISOLATION_FOREST,
        'include_one_hot': True,      # Incluir RA_*
        'exclude': EXCLUDE_ISOLATION_FOREST
    },
    'hbos': {
        'base_features': FEATURES_HBOS,
        'include_one_hot': False,     # NÃO incluir RA_*
        'exclude': EXCLUDE_HBOS
    },
    'gru': {  # RECOMENDADO para SSP-DF
        'base_features': FEATURES_GRU_AUTOENCODER,
        'include_one_hot': False,
        'exclude': []
    },
    'lstm': {  # Use apenas se window_size > 20
        'base_features': FEATURES_LSTM_AUTOENCODER,
        'include_one_hot': False,
        'exclude': []
    },
    'autoencoder': {  # Alias - usa GRU por padrão
        'base_features': FEATURES_GRU_AUTOENCODER,
        'include_one_hot': False,
        'exclude': []
    }
}

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_features_for_model(model_name, available_columns):
    """
    Retorna lista de features apropriadas para o modelo.
    
    Args:
        model_name (str): Nome do modelo ('isolation_forest', 'hbos', etc.)
        available_columns (list): Colunas disponíveis no DataFrame
        
    Returns:
        list: Features filtradas para o modelo
    """
    if model_name not in MODEL_FEATURES:
        raise ValueError(f"Modelo '{model_name}' não configurado. Opções: {list(MODEL_FEATURES.keys())}")
    
    config = MODEL_FEATURES[model_name]
    features = config['base_features'].copy()
    
    # Adicionar one-hot encoding se permitido
    if config['include_one_hot']:
        ra_features = [col for col in available_columns if col.startswith('RA_')]
        features.extend(ra_features)
    
    # Remover features excluídas
    features = [f for f in features if f not in config['exclude']]
    
    # Filtrar apenas features que existem no DataFrame
    features = [f for f in features if f in available_columns]
    
    return features


def get_feature_importance_order(model_name):
    """
    Retorna ordem de importância das features para o modelo.
    Útil para análise e interpretação.
    """
    importance = {
        'isolation_forest': [
            'velocidade_kmh', 'aceleracao', 'dist_m', 
            'hora_sin', 'hora_cos', 'eh_feriado', 'dia_sem'
        ],
        'hbos': [
            'velocidade_kmh', 'aceleracao', 'dist_m',
            'hora', 'eh_fim_de_semana', 'eh_feriado'
        ],
        'gru': [
            'velocidade_kmh', 'aceleracao', 'latitude', 'longitude',
            'dist_m', 'hora_sin', 'hora_cos'
        ],
        'lstm': [
            'velocidade_kmh', 'aceleracao', 'latitude', 'longitude',
            'dist_m', 'hora_sin', 'hora_cos'
        ]
    }
    return importance.get(model_name, [])