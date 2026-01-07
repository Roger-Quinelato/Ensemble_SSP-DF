def gerar_json_carros_por_ra():
    """Gera um JSON com a quantidade de carros únicos por Região Administrativa (RA)."""
    OUTPUT_JSON = os.path.join(REPORTS_DIR, 'quantidade_carros_por_ra.json')
    # Preferir Parquet se existir
    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
    if 'regiao_adm' in df.columns and 'placa' in df.columns:
        carros_por_ra = df.groupby('regiao_adm')['placa'].nunique().to_dict()
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(carros_por_ra, f, ensure_ascii=False, indent=2)
        print(f'✔️ JSON de quantidade de carros por RA salvo em {OUTPUT_JSON}')
    else:
        print('⚠️ Colunas regiao_adm ou placa não encontradas para gerar o JSON.')
"""
Script de organização e compilação de arquivos de outputs e relatórios.
Compila descrições, move arquivos finais, gera métricas e estatísticas iniciais da base de dados.
Inclui suporte a arquivos Parquet e imagens.
"""
import os
import glob
import shutil
import pandas as pd
import json

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'input')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')
IMG_DIR = os.path.join(BASE_DIR, 'outputs', 'imagens')
PARQUET_PATH = os.path.join(INPUT_DIR, 'amostra_ssp.parquet')
CSV_PATH = os.path.join(INPUT_DIR, 'amostra_ssp.csv')

# Certifica que os diretórios de saída existem
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

def compilar_descricoes():
    """Compila arquivos de descrição por família de modelo."""
    describe_files = glob.glob(os.path.join(BASE_DIR, 'describe_*.csv'))
    families = {}
    for f in describe_files:
        fname = os.path.basename(f)
        if 'ISO' in fname:
            families.setdefault('ISO', []).append(f)
        elif 'HBOS' in fname:
            families.setdefault('HBOS', []).append(f)
        elif 'LSTM' in fname:
            families.setdefault('LSTM', []).append(f)
        # Adicione outras famílias se necessário
    for fam, files in families.items():
        dfs = [pd.read_csv(f) for f in files]
        df_concat = pd.concat(dfs, ignore_index=True)
        df_concat.to_csv(os.path.join(REPORTS_DIR, f'describe_{fam}_compilado.csv'), index=False)
    print('✔️ Descrições compiladas por família.')

def mover_arquivos_finais():
    """Move arquivos finais de métricas/resultados para a pasta de reports."""
    final_files = [
        'iso_metrics.csv', 'iso_results.csv',
        'hbos_metrics.csv', 'hbos_results.csv',
        'lstm_metrics.csv', 'lstm_results.csv'
    ]
    for fname in final_files:
        src = os.path.join(BASE_DIR, fname)
        dst = os.path.join(REPORTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print('✔️ Arquivos finais movidos para reports.')

def mover_perfil_json():
    """Renomeia e move o perfil_dados.json para reports."""
    perfil_json = os.path.join(BASE_DIR, 'perfil_dados.json')
    new_json = os.path.join(REPORTS_DIR, 'InformacaoInicial_BaseDados.json')
    if os.path.exists(perfil_json):
        shutil.copy2(perfil_json, new_json)
        print('✔️ perfil_dados.json movido e renomeado.')

def gerar_metricas_base():
    """Gera métricas e estatísticas iniciais da base de dados (CSV e Parquet)."""
    # Preferir Parquet se existir
    if os.path.exists(PARQUET_PATH):
        df_raw = pd.read_parquet(PARQUET_PATH)
    else:
        df_raw = pd.read_csv(CSV_PATH)
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
    # Estatísticas descritivas para colunas categóricas
    categorical_cols = df_raw.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df_raw[categorical_cols].describe(include='all').to_csv(os.path.join(REPORTS_DIR, 'EstatisticasIniciais_Categoricas.csv'))
    # Estatísticas descritivas para colunas numéricas
    numeric_cols = df_raw.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        df_raw[numeric_cols].describe().to_csv(os.path.join(REPORTS_DIR, 'EstatisticasIniciais_Numericas.csv'))
    print('✔️ Métricas e estatísticas iniciais geradas.')

def mover_imagens():
    """Move imagens geradas para a pasta de reports (opcional, mantém cópia)."""
    for img_file in glob.glob(os.path.join(IMG_DIR, '*.jpg')):
        shutil.copy2(img_file, os.path.join(REPORTS_DIR, os.path.basename(img_file)))
    print('✔️ Imagens copiadas para reports.')

if __name__ == "__main__":
    print('🔄 Organizando e compilando relatórios e arquivos...')
    compilar_descricoes()
    mover_arquivos_finais()
    mover_perfil_json()
    gerar_metricas_base()
    gerar_json_carros_por_ra()
    mover_imagens()
    print('✅ Relatórios compilados e arquivos organizados em outputs/reports/')
