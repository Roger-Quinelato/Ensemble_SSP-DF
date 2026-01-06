# src/main.py

from src.pipeline.experiment_runner import run_experiment

if __name__ == "__main__":
    print("🚀 Inicializando Pipeline de Detecção de Anomalias (SSP-DF)...")
    try:
        run_experiment()
        # Organiza e compila relatórios após o pipeline
        from src.utils import organizacao_arquivos  # noqa: F401
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro fatal na execução: {e}")
        raise