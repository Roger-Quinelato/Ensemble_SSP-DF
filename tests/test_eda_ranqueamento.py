"""
Testes para o script de EDA e ranqueamento (eda_ranqueamento.py).
Garante que os gráficos principais são gerados corretamente.
"""
import os
import pytest
import importlib.util

IMG_DIR = os.path.join('outputs', 'imagens')

def test_eda_ranqueamento_gera_graficos():
    # Executa o script de EDA
    import sys
    import pathlib
    script_path = os.path.join('src', 'eda_ranqueamento.py')
    script_path = str(pathlib.Path(script_path).resolve())
    spec = importlib.util.spec_from_file_location("eda_ranqueamento", script_path)
    eda_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eda_module)
    # Verifica se os principais gráficos foram gerados
    arquivos_esperados = [
        'volume_por_regiao_adm.jpg',
        'volume_veiculos_por_semana_mes.jpg',
        'ranqueamento_anomalias_por_ra.jpg'
    ]
    for arq in arquivos_esperados:
        caminho = os.path.join(IMG_DIR, arq)
        assert os.path.exists(caminho), f'Gráfico não gerado: {caminho}'
