# organização_arquivos.py

import os
import glob
import shutil
import pandas as pd
import json

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')

# Certifica que o diretório de reports existe
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Compilar arquivos de descrição por família
describe_files = glob.glob(os.path.join(BASE_DIR, 'describe_*.csv'))

families = {}
for f in describe_files:
    fname = os.path.basename(f)
    if 'ISO' in fname:
        families.setdefault('ISO', []).append(f)
    elif 'LOF' in fname:
        families.setdefault('LOF', []).append(f)
    elif 'LSTM' in fname:
        families.setdefault('LSTM', []).append(f)
    # Adicione outras famílias se necessário

for fam, files in families.items():
    dfs = [pd.read_csv(f) for f in files]
    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat.to_csv(os.path.join(REPORTS_DIR, f'describe_{fam}_compilado.csv'), index=False)

# 2. Mover arquivos finais para reports
final_files = [
    'iso_metrics.csv', 'iso_results.csv',
    'lof_metrics.csv', 'lof_results.csv',
    'lstm_metrics.csv', 'lstm_results.csv'
]
for fname in final_files:
    src = os.path.join(BASE_DIR, fname)
    dst = os.path.join(REPORTS_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)

# 3. Renomear e mover perfil_dados.json
perfil_json = os.path.join(BASE_DIR, 'perfil_dados.json')
new_json = os.path.join(REPORTS_DIR, 'InformacaoInicial_BaseDados.json')
if os.path.exists(perfil_json):
    shutil.copy2(perfil_json, new_json)

# 4. Gerar métricas da base de dados crua
df_raw = pd.read_csv(os.path.join(INPUT_DIR, 'amostra_ssp.csv'))
metrics = {
    'num_linhas': len(df_raw),
    'num_colunas': len(df_raw.columns),
    'colunas': list(df_raw.columns),
    'tipos': df_raw.dtypes.astype(str).to_dict(),
    'nulos_por_coluna': df_raw.isnull().sum().to_dict(),
    'estatisticas': df_raw.describe(include='all').to_dict()
}

# Salva as métricas em CSV
metrics_df = pd.DataFrame({
    'Coluna': metrics['colunas'],
    'Tipo': [metrics['tipos'][col] for col in metrics['colunas']],
    'Nulos': [metrics['nulos_por_coluna'][col] for col in metrics['colunas']]
})
metrics_df.to_csv(os.path.join(REPORTS_DIR, 'DescricaoInicial_BaseDados.csv'), index=False)

# Salva estatísticas descritivas em outro CSV (opcional)
df_raw.describe(include='all').to_csv(os.path.join(REPORTS_DIR, 'Estatisticas_DescricaoInicial_BaseDados.csv'))

print('Relatórios compilados e arquivos organizados em outputs/reports/')
