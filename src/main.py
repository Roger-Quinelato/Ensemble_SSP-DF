"""
Ponto de entrada do Pipeline de Deteccao de Anomalias (SSP-DF).
Uso:
    python -m src.main
    python -m src.main --input data/input/producao.csv --epochs 50
    python -m src.main --config meu_config.yaml --seed 123
    python -m src.main --tf-device gpu
"""

import argparse
import datetime
import logging
import os
import random
import sys

import numpy as np

# Suporte a execucao direta: python src/main.py
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("sspdf")
from src.utils.logger_utils import setup_logger
from src.utils.tf_runtime import configure_tensorflow_runtime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de Deteccao de Anomalias Veiculares - SSP-DF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_mapeamento.yaml",
        help="Caminho do arquivo YAML de configuracao",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Caminho dos dados de entrada (CSV ou Parquet). Se nao especificado, usa o padrao.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Diretorio base de output. Sera criado outputs/<run_id>/ dentro deste.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Numero de epocas para treino temporal. "
            "Se nao informado, usa o valor de config_mapeamento.yaml "
            "(parametros.temporal.epochs). "
            "Se informado, tem precedencia sobre o YAML."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed global para reprodutibilidade",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar output detalhado",
    )
    parser.add_argument(
        "--tf-device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help=(
            "Dispositivo TensorFlow: auto (usa GPU se houver), "
            "cpu (forca CPU), gpu (exige GPU visivel)."
        ),
    )
    return parser.parse_args()


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    setup_logger(name="sspdf")
    if args.verbose:
        logging.getLogger("sspdf").setLevel(logging.DEBUG)
    set_global_seed(args.seed)

    # Importar TensorFlow DEPOIS de definir variaveis de ambiente e seeds.
    tf, tf_runtime = configure_tensorflow_runtime(args.tf_device)

    tf.random.set_seed(args.seed)

    from src.pipeline.experiment_runner import run_experiment

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_with_run = os.path.join(args.output_dir, run_id)

    logger.info("Inicializando Pipeline de Deteccao de Anomalias (SSP-DF)...")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"TensorFlow device mode: {args.tf_device}")
    logger.info(
        f"TensorFlow runtime ativo: {tf_runtime['active']} "
        f"(GPUs detectadas: {tf_runtime['gpu_count']})"
    )
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output dir: {output_dir_with_run}")
    if args.input:
        logger.info(f"Input: {args.input}")

    try:
        output_dir_final = run_experiment(
            config_path=args.config,
            input_path=args.input,
            output_dir=output_dir_with_run,
            epochs=args.epochs,
            seed=args.seed,
            run_id=run_id,
        )

        logger.info(f"Pipeline concluido. Outputs em: {output_dir_final}")
        if output_dir_final:
            relatorio = os.path.join(output_dir_final, "relatorio_executivo.html")
            if os.path.exists(relatorio):
                logger.info(f"Relatorio HTML disponivel: {relatorio}")
    except KeyboardInterrupt:
        logger.warning("Execucao interrompida pelo usuario.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Erro fatal na execucao: {e}")
        raise


if __name__ == "__main__":
    main()
