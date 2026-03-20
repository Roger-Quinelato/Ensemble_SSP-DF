import functools
import logging
import os
import time


# Diretorio de logs: configuravel via variavel de ambiente ou default
_LOG_DIR = os.environ.get("SSPDF_LOG_DIR", "outputs/logs")
_LOG_FILE = os.path.join(_LOG_DIR, "execution.log")


def _default_log_file(run_id=None):
    """Monta caminho padrao do log considerando run_id."""
    log_dir = os.environ.get("SSPDF_LOG_DIR", "outputs/logs")
    if run_id:
        return os.path.join(log_dir, f"execution_{run_id}.log")
    return os.path.join(log_dir, "execution.log")


def _ensure_log_dir(log_file):
    """Cria diretorio do arquivo de log sob demanda (nao no import)."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)


def _clear_handlers(logger):
    """Remove e fecha handlers para permitir reconfiguracao segura."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def setup_logger(name="sspdf", log_file=None, level=logging.INFO, run_id=None):
    """
    Configura logger com output DUAL: arquivo por run + console.

    Args:
        name: Nome do logger.
        log_file: Caminho do arquivo de log. Se None, usa default baseado em run_id.
        level: Nivel de logging.
        run_id: Identificador da execucao para nomear o arquivo de log.
    Returns:
        logging.Logger configurado.
    """
    if log_file is None:
        log_file = _default_log_file(run_id=run_id)

    default_run_log = _default_log_file(run_id=run_id)
    file_paths = [log_file]
    if os.path.abspath(default_run_log) != os.path.abspath(log_file):
        file_paths.append(default_run_log)

    logger = logging.getLogger(name)

    # Evitar duplicacao; quando solicitado log por run/log_file especifico, reconfigura.
    if logger.handlers:
        if run_id is None and log_file == _LOG_FILE:
            return logger
        _clear_handlers(logger)

    logger.setLevel(level)
    formatter_txt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler: Arquivo(s)
    for path in file_paths:
        _ensure_log_dir(path)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter_txt)
        logger.addHandler(file_handler)

    # Handler: Console (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
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
            logger.info(f"[START] {func_name}")
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"[END] {func_name} ({duration:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"[ERROR] Falha em {func_name}: {e}")
            raise

    return wrapper
