"""
Ponto de entrada do Pipeline de Deteccao de Anomalias (SSP-DF).
Uso:
    python -m src.main
    python -m src.main --input data/input/producao.csv --epochs 50
    python -m src.main --config meu_config.yaml --seed 123
"""

import argparse
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
        help="Numero de epocas para treino dos modelos temporais (LSTM/GRU)",
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

    logger.info("🚀 Inicializando Pipeline de Detecção de Anomalias (SSP-DF)...")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Output dir: {args.output_dir}")
    if args.input:
        logger.info(f"Input: {args.input}")

    try:
        run_experiment(
            config_path=args.config,
            input_path=args.input,
            output_dir=args.output_dir,
            epochs=args.epochs,
            seed=args.seed,
        )

        # Pos-processamento legado fixo em outputs/.
        if args.output_dir == "outputs":
            logger.info("📁 Organizando relatórios...")
            compilar_descricoes()
            mover_arquivos_finais()
            mover_perfil_json()
            gerar_metricas_base()
            gerar_json_carros_por_ra()
            mover_imagens()
            logger.info(
                "✅ Relatórios compilados e arquivos organizados em outputs/reports/"
            )
        else:
            logger.warning(
                "Pós-processamento em src.utils.organizacao_arquivos foi ignorado "
                "porque ainda usa caminhos fixos em 'outputs/'."
            )
    except KeyboardInterrupt:
        logger.warning("🛑 Execução interrompida pelo usuário.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"❌ Erro fatal na execução: {e}")
        raise


if __name__ == "__main__":
    main()
