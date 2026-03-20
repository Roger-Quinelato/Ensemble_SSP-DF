"""
MLflow tracking leve e opt-in para o pipeline SSP-DF.

Desativar com: export SSPDF_DISABLE_TRACKING=true
Ou sem MLflow instalado: funciona sem rastreamento (sem erro).
"""

import logging
import os

logger = logging.getLogger("sspdf")
_DISABLED = os.environ.get("SSPDF_DISABLE_TRACKING", "false").lower() in (
    "true",
    "1",
    "yes",
)

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    if not _DISABLED:
        logger.info(
            "MLflow nao instalado - tracking desativado. "
            "Instale com: pip install mlflow"
        )


def tracking_active():
    return _MLFLOW_AVAILABLE and not _DISABLED


def init_experiment(experiment_name="sspdf_anomaly_detection", run_id=None):
    if not tracking_active():
        return None
    try:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_id)
        logger.info(
            f"MLflow run iniciado: {run.info.run_id} | Experiment: {experiment_name}"
        )
        return run
    except Exception as e:
        logger.warning(f"MLflow init falhou (nao critico): {e}")
        return None


def log_params(params: dict):
    if not tracking_active():
        return
    try:
        # MLflow limita chaves a 250 chars e valores a 500 chars
        clean = {str(k)[:250]: str(v)[:500] for k, v in params.items()}
        mlflow.log_params(clean)
    except Exception as e:
        logger.warning(f"MLflow log_params falhou: {e}")


def log_metrics(metrics: dict, step=None):
    if not tracking_active():
        return
    try:
        numeric = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        mlflow.log_metrics(numeric, step=step)
    except Exception as e:
        logger.warning(f"MLflow log_metrics falhou: {e}")


def log_artifact(path: str):
    if not tracking_active() or not os.path.exists(path):
        return
    try:
        mlflow.log_artifact(path)
    except Exception as e:
        logger.warning(f"MLflow log_artifact falhou: {e}")


def end_run(status="FINISHED"):
    if not tracking_active():
        return
    try:
        mlflow.end_run(status=status)
        logger.info(f"MLflow run finalizado com status: {status}")
    except Exception as e:
        logger.warning(f"MLflow end_run falhou: {e}")
