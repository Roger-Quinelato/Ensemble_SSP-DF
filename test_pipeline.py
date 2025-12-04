"""
Teste de integração do pipeline principal.
Valida se o fluxo completo executa e gera os principais arquivos de saída.
"""
import os
from src.experiment_runner import run_experiment

def test_pipeline_flow():
    run_experiment()
    assert os.path.exists('outputs/master_table/resultado_final.parquet'), "Arquivo de saída principal não gerado."
    assert os.path.exists('outputs/metrics/perfil_dados.json'), "Arquivo de perfil de dados não gerado."
    assert os.path.exists('outputs/metrics/comparativo_completo.csv'), "Arquivo de comparativo não gerado."

if __name__ == "__main__":
    test_pipeline_flow()
    print("Teste de fluxo do pipeline executado com sucesso!")
