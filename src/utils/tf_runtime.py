"""
Utilitarios para configurar runtime TensorFlow (CPU/GPU) de forma auditavel.
"""

import logging
import os
import random

import numpy as np

logger = logging.getLogger("sspdf")

VALID_TF_DEVICE_MODES = ("auto", "cpu", "gpu")


def configure_tensorflow_runtime(tf_device="auto"):
    """
    Configura o runtime TensorFlow com selecao explicita de dispositivo.

    Args:
        tf_device: "auto", "cpu" ou "gpu".
            - auto: usa GPU se disponivel, senao CPU.
            - cpu: forca CPU (desabilita visibilidade de GPU).
            - gpu: exige GPU; falha se nenhuma GPU estiver disponivel.

    Returns:
        tuple:
            - modulo tensorflow importado
            - dict com metadados de runtime
    """
    requested = (tf_device or "auto").strip().lower()
    if requested not in VALID_TF_DEVICE_MODES:
        raise ValueError(
            f"tf_device invalido: '{tf_device}'. "
            f"Use um de {VALID_TF_DEVICE_MODES}."
        )

    # Forca CPU antes do import do TensorFlow.
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    gpu_count = len(gpus)

    if requested == "gpu" and gpu_count == 0:
        raise RuntimeError(
            "tf_device='gpu' solicitado, mas nenhuma GPU foi detectada pelo TensorFlow. "
            "Verifique driver NVIDIA, CUDA/cuDNN e NVIDIA Container Toolkit no host."
        )

    if gpu_count > 0:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                logger.warning(
                    f"Nao foi possivel habilitar memory_growth para {gpu.name}: {exc}"
                )

    active = "gpu" if gpu_count > 0 and requested != "cpu" else "cpu"
    gpu_names = [gpu.name for gpu in gpus]

    logger.info(
        f"TensorFlow runtime configurado | solicitado={requested} | ativo={active} "
        f"| gpus_detectadas={gpu_count}"
    )
    if gpu_names:
        logger.info(f"GPUs detectadas: {gpu_names}")
    elif requested in ("auto", "gpu"):
        logger.warning("Nenhuma GPU detectada; TensorFlow executara em CPU.")

    return tf, {
        "requested": requested,
        "active": active,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
    }


def setup_deterministic_runtime(seed, tf_device="auto"):
    """
    Configura seed deterministica + runtime TensorFlow em um unico ponto.

    Esta funcao deve ser chamada antes de qualquer import prematuro de
    TensorFlow/Keras para garantir reproducibilidade entre execucoes.

    Args:
        seed: Seed global (int).
        tf_device: "auto", "cpu" ou "gpu".
    Returns:
        tuple:
            - modulo tensorflow
            - dict de runtime com metadados e seed
    """
    seed_int = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_int)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed_int)
    np.random.seed(seed_int)

    tf, runtime = configure_tensorflow_runtime(tf_device=tf_device)
    tf.random.set_seed(seed_int)

    logger.info(
        "Seed/runtime deterministico configurado | "
        f"seed={seed_int} | tf_device={runtime['requested']} | tf_ativo={runtime['active']}"
    )
    runtime = dict(runtime)
    runtime["seed"] = seed_int
    return tf, runtime
