"""
Ponto de entrada do Pipeline de Deteccao de Anomalias (SSP-DF).
Uso:
    python -m src.main
    python -m src.main --input data/input/producao.csv --epochs 50
    python -m src.main --config meu_config.yaml --seed 123
"""

import argparse
import datetime
import os
import random
import sys

import numpy as np

# Suporte a execucao direta: python src/main.py
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger_utils import logger


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
        help="Diretorio base para saida de resultados",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Numero de epocas para treino dos modelos temporais (GRU)",
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
    return parser.parse_args()


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    if args.verbose:
        import logging

        logging.getLogger("sspdf").setLevel(logging.DEBUG)
    set_global_seed(args.seed)

    # Importar TensorFlow DEPOIS de definir variaveis de ambiente e seeds.
    import tensorflow as tf

    tf.random.set_seed(args.seed)

    from src.pipeline.experiment_runner import run_experiment
    from src.utils.organizacao_arquivos import (
        compilar_descricoes,
        mover_arquivos_finais,
        mover_perfil_json,
        gerar_metricas_base,
        gerar_json_carros_por_ra,
        mover_imagens,
    )

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_with_run = os.path.join(args.output_dir, run_id)

    logger.info("Inicializando Pipeline de Deteccao de Anomalias (SSP-DF)...")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output dir: {output_dir_with_run}")
    if args.input:
        logger.info(f"Input: {args.input}")

    try:
        run_experiment(
            config_path=args.config,
            input_path=args.input,
            output_dir=output_dir_with_run,
            epochs=args.epochs,
            seed=args.seed,
            run_id=run_id,
        )

        # Pos-processamento legado fixo em outputs/. Incompativel com subdiretorios versionados.
        if output_dir_with_run == "outputs":
            logger.info("Organizando relatorios...")
            compilar_descricoes()
            mover_arquivos_finais()
            mover_perfil_json()
            gerar_metricas_base()
            gerar_json_carros_por_ra()
            mover_imagens()
            logger.info("Relatorios compilados e arquivos organizados em outputs/reports/")
        else:
            logger.warning(
                "Pos-processamento em src.utils.organizacao_arquivos foi ignorado "
                "porque usa caminhos fixos em 'outputs/'."
            )
    except KeyboardInterrupt:
        logger.warning("Execucao interrompida pelo usuario.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Erro fatal na execucao: {e}")
        raise


if __name__ == "__main__":
    main()
