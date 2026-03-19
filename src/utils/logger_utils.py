import functools
import logging
import os
import time


# Diretorio de logs: configuravel via variavel de ambiente ou default
_LOG_DIR = os.environ.get("SSPDF_LOG_DIR", "outputs/logs")
_LOG_FILE = os.path.join(_LOG_DIR, "execution.log")


def _ensure_log_dir():
    """Cria diretorio de logs sob demanda (nao no import)."""
    os.makedirs(_LOG_DIR, exist_ok=True)


def setup_logger(name="sspdf", log_file=None, level=logging.INFO):
    """
    Configura logger com output DUAL: arquivo + console.
    O diretorio de logs e criado apenas quando o logger e configurado,
    nao no momento do import.

    Args:
        name: Nome do logger.
        log_file: Caminho do arquivo de log. Se None, usa default.
        level: Nivel de logging.
    Returns:
        logging.Logger configurado.
    """
    if log_file is None:
        log_file = _LOG_FILE

    logger = logging.getLogger(name)

    # Evitar duplicar handlers se chamado multiplas vezes
    if logger.handlers:
        return logger

    # Criar diretorio de logs sob demanda
    _ensure_log_dir()

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
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# Logger global do projeto
# NOTA: setup_logger() agora cria o diretorio sob demanda na primeira chamada,
# nao no import. Porem, como este modulo e importado antes do main, o diretorio
# sera criado na inicializacao do logger.
logger = setup_logger()


def log_execution(func):
    """Decorador para cronometrar funcoes e capturar erros."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func_name = func.__name__
        try:
            logger.info(f"[START] {func_name}")
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"[END] {func_name} ({duration:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"[ERROR] Falha em {func_name}: {e}")
            raise

    return wrapper
