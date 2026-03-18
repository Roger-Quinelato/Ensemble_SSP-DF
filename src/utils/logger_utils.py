import functools
import logging
import os
import time

# Criar diretorio de logs
os.makedirs("outputs/logs", exist_ok=True)


def setup_logger(name="sspdf", log_file="outputs/logs/execution.log", level=logging.INFO):
    """
    Configura logger com output DUAL: arquivo + console.
    Chamado uma vez na inicializacao do pipeline.
    """
    logger = logging.getLogger(name)

    # Evitar duplicar handlers se chamado multiplas vezes
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler: Arquivo
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler: Console (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # Console usa formato mais curto
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# Logger global do projeto
logger = setup_logger()


def log_execution(func):
    """Decorador para cronometrar funcoes e capturar erros."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func_name = func.__name__
        try:
            logger.info(f"⏳ [START] {func_name}")
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"✅ [END] {func_name} ({duration:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"❌ [ERROR] Falha em {func_name}: {e}")
            raise

    return wrapper
