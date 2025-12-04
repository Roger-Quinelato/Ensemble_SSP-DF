# src/main.py

import sys
import os

# Adiciona o diretório raiz do projeto ao PYTHONPATH
# Isso permite rodar "python src/main.py" sem erros de importação
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_runner import run_experiment

if __name__ == "__main__":
    print("🚀 Inicializando Pipeline de Detecção de Anomalias (SSP-DF)...")
    try:
        run_experiment()
        # Organiza e compila relatórios após o pipeline
        import src.organizacao_arquivos
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro fatal na execução: {e}")
        raise