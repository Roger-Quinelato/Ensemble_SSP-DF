# src/experiment_runner.py

import pandas as pd
import numpy as np
import yaml
import json
import itertools
from dask_ml.preprocessing import MinMaxScaler
from .data_processor import DataProcessor
from .models_base import BaselineModels
from .models_deep import LSTMPipeline
from .evaluation import ThresholdOptimizer, GroundTruthComparator

def run_experiment():
    """
    Executa o pipeline completo de detecção de anomalias (Experimento End-to-End)
    de forma agnóstica ao schema, gerando uma Master Table com evidências de múltiplos modelos.

    Esta função atua como o orquestrador principal, seguindo as fases:
    1. Ingestão e Adaptação (Mapeamento via Config).
    2. Sanitização Genérica (Tratamento de tipos e nulos).
    3. Feature Engineering Condicional (Só calcula o que os dados permitirem).
    4. Modelagem Pontual (Isolation Forest, LOF Clássico).
    5. Geração de Ground Truths Sintéticos (União e Interseção).
    6. Modelagem Sequencial (LSTM-AE treinado sobre diferentes GTs).
    7. Consolidação da Master Table de Evidências.

    Args:
        raw_data_path (str): Caminho para o arquivo de dados brutos (CSV ou Parquet).
                             O schema deste arquivo é desconhecido a priori.
        config_path (str): Caminho para o arquivo 'config_mapeamento.yaml'.
                           Este arquivo DEVE conter o dicionário de para (De->Para)
                           das colunas da SSP para as colunas internas do modelo.

    Returns:
        pd.DataFrame: Master DataFrame ('benchmark_results') contendo:
            - Colunas originais mapeadas e features calculadas (condicionais).
            - Scores e Labels do Isolation Forest.
            - Scores e Labels do LOF (Clássico).
            - Labels dos Ground Truths Sintéticos (GT_Uniao, GT_Intersecao).
            - Erros de Reconstrução (MAE/MSE) e Labels dos modelos LSTM-AE 
              (variantes Uniao, Intersecao e Sujo).
            - Metadados de identificação (objeto_id, timestamp).

    Raises:
        FileNotFoundError: Se os arquivos de dados ou configuração não existirem.
        KeyError: Se as colunas mínimas obrigatórias (definidas no config) não
                  puderem ser mapeadas.
        ValueError: Se o dataset estiver vazio após a sanitização.

    Notes:
        - O pipeline é resiliente à ausência de colunas opcionais (ex: lat/lon, velocidade).
          Warnings serão logados, mas a execução continuará.
        - O LSTM-AE será treinado apenas com as features numéricas disponíveis após
          o pré-processamento.
        - Os artefatos (modelos treinados, scalers) são salvos no diretório definido
          no config.yaml.
    """
    
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/master_table', exist_ok=True)
    
    with open('config_mapeamento.yaml') as f:
        config = yaml.safe_load(f)
    
    # Atualizar referências de colunas para usar o mapeamento do arquivo de configuração
    # Substituir referências diretas como 'placa', 'timestamp', etc., por config['mapeamento_colunas']['<coluna>']
    map_cols = config['mapeamento_colunas']
    
    proc = DataProcessor(config)
    print("=" * 80)
    print("📂 Carregando e Processando Dados...")
    df = proc.load_and_standardize('data/input/amostra_ssp.csv') 
    df = proc.feature_engineering(df)
    
    # Profiling Rápido e Informações Gerais
    dias = df[map_cols['timestamp']].dt.date.nunique()
    meses = df[map_cols['timestamp']].dt.to_period('M').nunique()
    grouped = df.groupby([map_cols['latitude'], map_cols['longitude']])[map_cols['placa']].count()
    if hasattr(grouped, 'compute'):
        grouped = grouped.compute()
    local_mais_fluxo = grouped.sort_values(ascending=False).reset_index().iloc[0]
    stats = {
        'total_veiculos': int(df[map_cols['placa']].nunique()),
        'periodo': f"{df[map_cols['timestamp']].min()} a {df[map_cols['timestamp']].max()}",
        'vel_media': float(df['velocidade_calc'].mean()),
        'dias_analise': int(dias),
        'meses_analise': int(meses),
        'local_mais_fluxo_latitude': float(local_mais_fluxo[map_cols['latitude']]),
        'local_mais_fluxo_longitude': float(local_mais_fluxo[map_cols['longitude']]),
        'fluxo_veiculos_local': int(local_mais_fluxo[map_cols['placa']])
    }
    with open('outputs/metrics/perfil_dados.json', 'w') as f:
        json.dump(stats, f, indent=4)

    # Preparação ML
    scaler = MinMaxScaler()
    # Se for Dask, converte para pandas para o scaler
    import dask.dataframe as dd
    if isinstance(df, dd.DataFrame):
        df_pd = df.compute()
    else:
        df_pd = df
    X_scaled = scaler.fit_transform(df[proc.features_to_use])
    # Adiciona dimensões categóricas
    df['hora'] = df[map_cols['timestamp']].dt.hour
    df['dia_semana'] = df[map_cols['timestamp']].dt.day_name()
    # Estatísticas por bins de score usando pd.qcut
    for col_score in score_columns_audit:
        df[f'{col_score}_bin'] = pd.qcut(df[col_score], q=10, labels=[f'Decil_{i+1}' for i in range(10)])
        for dim in ['hora', 'dia_semana', f'{col_score}_bin']:
            stats = df.groupby(dim)[col_score].describe()
            filename = f'outputs/metrics/stats_{col_score}_por_{dim}.csv'
            stats.to_csv(filename)
            print(f"   Salvo: {filename}")
    models_base = BaselineModels(X_scaled)
    optimizer = ThresholdOptimizer(config['parametros']['percentis_teste'])
    results_summary = []
    # Dicionários para armazenar as máscaras de "Inliers" (Dados Normais) de cada variação
    iso_masks_registry = {} 
    lof_masks_registry = {}
    score_columns_audit = []

    # ==========================================================================
    # 2. TREINAMENTO DOS MODELOS BASE (GERADORES DE HIPÓTESES)
    # ==========================================================================
    
    # --- LOOP ISO FOREST ---
    print("\n" + "-" * 40)
    print("🌲 TREINANDO VARIAÇÕES ISO FOREST")
    for n_est in [100, 200]:
        tag = f"ISO_n{n_est}"
        print(f"   ↳ {tag}...")
        labels, scores, model = models_base.train_iso(n_estimators=n_est)
        # Salvar modelo Isolation Forest
        import joblib
        import os
        os.makedirs('outputs/models_saved', exist_ok=True)
        joblib.dump(model, f'outputs/models_saved/iso_n{n_est}.joblib')
        
        # Salvar Score
        df[f'{tag}_score'] = -scores
        score_columns_audit.append(f'{tag}_score')
        
        # Calcular Percentis
        df, metrics = optimizer.apply_dynamic_thresholds(df, f'{tag}_score', tag)
        results_summary.extend(metrics)
        
        # REGISTRAR MÁSCARA (Usando p95 como corte de segurança para 'Normalidade')
        # Quem é '0' no label p95 é considerado 'Inlier' (Normal) para treino futuro
        iso_masks_registry[tag] = (df[f'{tag}_p95_label'] == 0)

    # --- LOOP LOF ---
    print("\n" + "-" * 40)
    print("🔍 TREINANDO VARIAÇÕES LOF")
    for k in [10, 20]:
        tag = f"LOF_k{k}_standard"
        print(f"   ↳ {tag}...")
        try:
            labels, scores = models_base.train_lof(k_neighbors=k)
            # Salvar modelo LOF
            import joblib
            import os
            os.makedirs('outputs/models_saved', exist_ok=True)
                joblib.dump(model, f'outputs/models_saved/lof_k{k}_{strat}.joblib')
                
                df[f'{tag}_score'] = -scores
                score_columns_audit.append(f'{tag}_score')
                
                df, metrics = optimizer.apply_dynamic_thresholds(df, f'{tag}_score', tag)
                results_summary.extend(metrics)
                
                # REGISTRAR MÁSCARA
                lof_masks_registry[tag] = (df[f'{tag}_p95_label'] == 0)
                
            except Exception as e:
                print(f"   ❌ Falha em {tag}: {e}")

    # ==========================================================================
    # 3. COMBINATORIAL GROUND TRUTH & LSTM LOOP
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🧠 TREINAMENTO LSTM MULTI-CENÁRIOS (COMBINATORIAL)")
    print("=" * 80)
    
    # Instância única do Pipeline (dados são os mesmos)
    gap_seconds = config.get('configuracoes_gerais', {}).get('gap_segmentation_seconds', 300)
    lstm_pipe = LSTMPipeline(
        X_data=X_scaled,
        vehicle_ids=df[map_cols['placa']].values,
        timestamps=df[map_cols['timestamp']].values,
        original_indices=df.index.values,
        window_size=config['parametros']['lstm_window_size'],
        max_gap_seconds=gap_seconds
    )

    # Vamos combinar cada ISO com cada LOF para criar GTs diferentes
    # Ex: GT1 = ISO_100 + LOF_10 | GT2 = ISO_200 + LOF_20 ...
    
    # Para evitar explosão combinatória (muitas horas de treino), vamos selecionar
    # pares representativos ou fazer o produto cartesiano completo.
    # Aqui faremos o Produto Cartesiano dos modelos que deram certo.
    
    # Loop expandido: todas as combinações possíveis de groundtruths (União e Interseção)
    for iso_name, iso_mask_inlier in iso_masks_registry.items():
        for lof_name, lof_mask_inlier in lof_masks_registry.items():
            gt_base_name = f"GT_({iso_name}+{lof_name})"
            print(f"\n⚡ Cenário: {gt_base_name}")
            # União (Rigorosa)
            mask_train_uniao = iso_mask_inlier & lof_mask_inlier
            lstm_name_uniao = f"LSTM_Uniao_{iso_name}_{lof_name}"
            print(f"   ↳ Treinando {lstm_name_uniao}...")
            mse_uniao, idx_uniao, model_uniao = lstm_pipe.train_evaluate(lstm_name_uniao, mask_train=mask_train_uniao, epochs=5)
            # Salvar modelo LSTM União
            if model_uniao is not None:
                import os
                os.makedirs('outputs/models_saved', exist_ok=True)
                model_uniao.save(f'outputs/models_saved/lstm_uniao_{iso_name}_{lof_name}.h5')
            if mse_uniao is not None and idx_uniao is not None:
                df.loc[idx_uniao, f'{lstm_name_uniao}_score'] = mse_uniao
                score_columns_audit.append(f'{lstm_name_uniao}_score')
                df, metrics = optimizer.apply_dynamic_thresholds(df, f'{lstm_name_uniao}_score', lstm_name_uniao)
                results_summary.extend(metrics)
            # Interseção (Permissiva)
            mask_train_inter = iso_mask_inlier | lof_mask_inlier
            lstm_name_inter = f"LSTM_Inter_{iso_name}_{lof_name}"
            print(f"   ↳ Treinando {lstm_name_inter}...")
            mse_inter, idx_inter, model_inter = lstm_pipe.train_evaluate(lstm_name_inter, mask_train=mask_train_inter, epochs=5)
            # Salvar modelo LSTM Interseção
            if model_inter is not None:
                import os
                os.makedirs('outputs/models_saved', exist_ok=True)
                model_inter.save(f'outputs/models_saved/lstm_inter_{iso_name}_{lof_name}.h5')
            if mse_inter is not None and idx_inter is not None:
                df.loc[idx_inter, f'{lstm_name_inter}_score'] = mse_inter
                score_columns_audit.append(f'{lstm_name_inter}_score')
                df, metrics = optimizer.apply_dynamic_thresholds(df, f'{lstm_name_inter}_score', lstm_name_inter)
                results_summary.extend(metrics)

    # Treino Controle (Sujo - Sem GT)
    print(f"\n⚡ Cenário: Controle (Sem GT)")
    mse_sujo, idx_sujo, model_sujo = lstm_pipe.train_evaluate("LSTM_Sujo", mask_train=None, epochs=5)
    # Salvar modelo LSTM Sujo
    if model_sujo is not None:
        import os
        os.makedirs('outputs/models_saved', exist_ok=True)
        model_sujo.save('outputs/models_saved/lstm_sujo.h5')
    if mse_sujo is not None and idx_sujo is not None:
        df.loc[idx_sujo, 'LSTM_Sujo_score'] = mse_sujo
        score_columns_audit.append('LSTM_Sujo_score')
        df, metrics = optimizer.apply_dynamic_thresholds(df, 'LSTM_Sujo_score', 'LSTM_Sujo')
        results_summary.extend(metrics)

    # ==========================================================================
    # 4. EXPORTAÇÃO
    # ==========================================================================
    print("\n" + "=" * 80)
    print("💾 SALVANDO RESULTADOS")
    
    # Exportação com Dask
    if hasattr(df, 'to_parquet'):
        df.to_parquet('outputs/master_table/resultado_final.parquet', index=False)
    else:
        pd.DataFrame(df).to_parquet('outputs/master_table/resultado_final.parquet', index=False)
    # Exportar métricas agrupadas por família de modelo
    iso_metrics = [m for m in results_summary if m['Model'].startswith('ISO')]
    lof_metrics = [m for m in results_summary if m['Model'].startswith('LOF')]
    lstm_metrics = [m for m in results_summary if m['Model'].startswith('LSTM')]
    if iso_metrics:
        pd.DataFrame(iso_metrics).to_csv('outputs/metrics/iso_metrics.csv', index=False)
    if lof_metrics:
        pd.DataFrame(lof_metrics).to_csv('outputs/metrics/lof_metrics.csv', index=False)
    if lstm_metrics:
        pd.DataFrame(lstm_metrics).to_csv('outputs/metrics/lstm_metrics.csv', index=False)
    print("   ✅ Métricas exportadas: iso_metrics.csv, lof_metrics.csv, lstm_metrics.csv")
    # Segmentação dos resultados por modelo
    id_cols = [map_cols['placa'], map_cols['timestamp'], map_cols['latitude'], map_cols['longitude']]
    df_iso = df[id_cols + [col for col in df.columns if col.startswith('ISO')]]
    df_lof = df[id_cols + [col for col in df.columns if col.startswith('LOF')]]
    df_lstm = df[id_cols + [col for col in df.columns if col.startswith('LSTM')]]
    if hasattr(df_iso, 'to_csv'):
        # Salva CSV de forma compatível com Dask e pandas
        if hasattr(df_iso, 'to_csv') and 'single_file' in df_iso.to_csv.__code__.co_varnames:
            df_iso.to_csv('outputs/metrics/iso_results.csv', single_file=True, index=False)
        else:
            df_iso.to_csv('outputs/metrics/iso_results.csv', index=False)
        # Salva CSV de forma compatível com Dask e pandas
        if hasattr(df_lof, 'to_csv') and 'single_file' in df_lof.to_csv.__code__.co_varnames:
            df_lof.to_csv('outputs/metrics/lof_results.csv', single_file=True, index=False)
        else:
            df_lof.to_csv('outputs/metrics/lof_results.csv', index=False)
        # Salva CSV de forma compatível com Dask e pandas
        if hasattr(df_lstm, 'to_csv') and 'single_file' in df_lstm.to_csv.__code__.co_varnames:
            df_lstm.to_csv('outputs/metrics/lstm_results.csv', single_file=True, index=False)
        else:
            df_lstm.to_csv('outputs/metrics/lstm_results.csv', index=False)
    else:
        pd.DataFrame(df_iso).to_csv('outputs/metrics/iso_results.csv', index=False)
        pd.DataFrame(df_lof).to_csv('outputs/metrics/lof_results.csv', index=False)
        pd.DataFrame(df_lstm).to_csv('outputs/metrics/lstm_results.csv', index=False)
    print("   ✅ Resultados segmentados exportados: iso_results.csv, lof_results.csv, lstm_results.csv")
    # Describe comparativo
    if score_columns_audit:
        describe_df = df[score_columns_audit].describe().T
        if hasattr(describe_df, 'to_csv'):
            # Salva CSV de forma compatível com Dask e pandas
            if hasattr(describe_df, 'to_csv') and 'single_file' in describe_df.to_csv.__code__.co_varnames:
                describe_df.to_csv('outputs/metrics/comparativo_completo.csv', single_file=True)
            else:
                describe_df.to_csv('outputs/metrics/comparativo_completo.csv')
        else:
            pd.DataFrame(describe_df).to_csv('outputs/metrics/comparativo_completo.csv')
        print("   ✅ Comparativo salvo: outputs/metrics/comparativo_completo.csv")
        # Estatísticas agrupadas por variação de parâmetros (exemplo para LOF)
        lof_cols = [col for col in score_columns_audit if col.startswith('LOF')]
        if lof_cols:
            for col in lof_cols:
                parts = col.split('_')
                if len(parts) >= 4:
                    k = parts[2][1:]
                    strat = parts[3]
                    group_stats = df.groupby([f'{col}'])[col].describe()
                    if hasattr(group_stats, 'to_csv'):
                        # Salva CSV de forma compatível com Dask e pandas
                        if hasattr(group_stats, 'to_csv') and 'single_file' in group_stats.to_csv.__code__.co_varnames:
                            group_stats.to_csv(f'outputs/metrics/describe_LOF_{strat}_k{k}.csv', single_file=True)
                        else:
                            group_stats.to_csv(f'outputs/metrics/describe_LOF_{strat}_k{k}.csv')
                    else:
                        pd.DataFrame(group_stats).to_csv(f'outputs/metrics/describe_LOF_{strat}_k{k}.csv')
        # Estatísticas agrupadas para LSTM-AE
        lstm_cols = [col for col in score_columns_audit if col.startswith('LSTM')]
        if lstm_cols:
            for col in lstm_cols:
                group_stats = df.groupby([col])[col].describe()
                if hasattr(group_stats, 'to_csv'):
                    # Salva CSV de forma compatível com Dask e pandas
                    if hasattr(group_stats, 'to_csv') and 'single_file' in group_stats.to_csv.__code__.co_varnames:
                        group_stats.to_csv(f'outputs/metrics/describe_{col}.csv', single_file=True)
                    else:
                        group_stats.to_csv(f'outputs/metrics/describe_{col}.csv')
                else:
                    pd.DataFrame(group_stats).to_csv(f'outputs/metrics/describe_{col}.csv')
    print("\n✅ EXPERIMENTO COMBINATÓRIO FINALIZADO!")

if __name__ == "__main__":
    run_experiment()