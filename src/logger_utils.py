# src/logger_utils.py

import time
import functools
import logging
import os

# Configuração básica de Logging
os.makedirs('outputs/logs', exist_ok=True)
logging.basicConfig(
    filename='outputs/logs/execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_execution(func):
    """Decorador para cronometrar funções e capturar erros."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func_name = func.__name__
        try:
            logging.info(f"Iniciando {func_name}...")
            print(f"⏳ [START] {func_name}")
            result = func(*args, **kwargs)
            end = time.time()
            duration = end - start
            logging.info(f"Finalizado {func_name}. Tempo: {duration:.2f}s")
            print(f"✅ [END] {func_name} ({duration:.2f}s)")
            return result
        except Exception as e:
            logging.error(f"Erro em {func_name}: {str(e)}")
            print(f"❌ [ERROR] Falha em {func_name}: {e}")
            raise e
    return wrapper