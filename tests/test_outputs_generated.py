
"""
Testes para verificar se arquivos foram gerados nos diretórios de saída do projeto Ensemble_SSP-DF.

Testes:
    - test_reports_generated: Verifica se há arquivos no diretório outputs/reports.
    - test_models_saved_generated: Verifica se há arquivos no diretório outputs/models_saved.

Esses testes ajudam a garantir que o pipeline principal está gerando os relatórios e modelos esperados.
Além disso, exibem o diretório atual e os arquivos encontrados para facilitar o debug em ambientes como Google Colab.
"""

import os
import unittest

class TestOutputFiles(unittest.TestCase):
    def test_reports_generated(self):
        print(f"Diretório atual: {os.getcwd()}")
        reports_dir = os.path.join('outputs', 'reports')
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        files = os.listdir(reports_dir)
        print(f"Arquivos em {reports_dir}: {files}")
        self.assertTrue(len(files) > 0, f'Nenhum arquivo gerado em {reports_dir}')

    def test_models_saved_generated(self):
        print(f"Diretório atual: {os.getcwd()}")
        models_dir = os.path.join('outputs', 'models_saved')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        files = os.listdir(models_dir)
        print(f"Arquivos em {models_dir}: {files}")
        self.assertTrue(len(files) > 0, f'Nenhum arquivo gerado em {models_dir}')

if __name__ == '__main__':
    unittest.main()
