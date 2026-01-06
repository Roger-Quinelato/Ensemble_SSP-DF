# src/pipeline/experiment_runner.py

import os
import pandas as pd
import numpy as np
import yaml
import json
import itertools
import joblib
from dask_ml.preprocessing import MinMaxScaler

# Imports ajustados para a nova estrutura de pastas
from src.data.data_processor import DataProcessor
from src.models.models_base import BaselineModels
from src.models.models_deep import LSTMPipeline
from src.utils.evaluation import ThresholdOptimizer, GroundTruthComparator


def run_experiment():
    """
    Executa o pipeline completo de detecção de anomalias (Experimento End-to-End).
    """
    # CRÍTICO: Não coloque 'import os' dentro desta função em lugar nenhum!
    # O único import os deve ser o da linha 1 deste arquivo.
    
    print("📂 Inicializando diretórios...")
    # Como 'os' é importado apenas lá em cima, agora ele é Global e funciona aqui.
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/master_table', exist_ok=True)
    os.makedirs('outputs/models_saved', exist_ok=True)

    # Carregar Configuração
    print("⚙️ Carregando configurações...")
    with open('config_mapeamento.yaml') as f:
        config = yaml.safe_load(f)

    # Atualizar referências de colunas
    map_cols = config['mapeamento_colunas']

    proc = DataProcessor(config)
    print("=" * 80)
    print("📂 Carregando e Processando Dados...")
    
    # Caminho com fallback (tenta CSV, se não achar, tenta Parquet)
    input_path = 'data/input/amostra_ssp.csv'
    if not os.path.exists(input_path):
        input_path = 'data/input/amostra_ssp.parquet'
        
    df = proc.load_and_standardize(input_path)
    df = proc.feature_engineering(df)

    # Profiling Rápido
    dias = df[map_cols['timestamp']].dt.date.nunique()
    meses = df[map_cols['timestamp']].dt.to_period('M').nunique()
    
    # Agrupamento (compatível com Dask/Pandas)
    grouped = df.groupby([map_cols['latitude'], map_cols['longitude']])[map_cols['placa']].count()
    if hasattr(grouped, 'compute'):
        grouped = grouped.compute()
    
    if not grouped.empty:
        local_mais_fluxo = grouped.sort_values(ascending=False).reset_index().iloc[0]
        stats = {
            'total_veiculos': int(df[map_cols['placa']].nunique()),
            'periodo': f"{df[map_cols['timestamp']].min()} a {df[map_cols['timestamp']].max()}",
            'vel_media': float(df['velocidade_calc'].mean()) if 'velocidade_calc' in df.columns else 0.0,
            'dias_analise': int(dias),
            'meses_analise': int(meses),
            'local_mais_fluxo_latitude': float(local_mais_fluxo[map_cols['latitude']]),
            'local_mais_fluxo_longitude': float(local_mais_fluxo[map_cols['longitude']]),
            'fluxo_veiculos_local': int(local_mais_fluxo[map_cols['placa']])
        }
    else:
        stats = {'info': 'Base vazia ou sem agrupamento possível'}

    with open('outputs/metrics/perfil_dados.json', 'w') as f:
        json.dump(stats, f, indent=4)

    # Preparação ML
    scaler = MinMaxScaler()
    import dask.dataframe as dd
    if isinstance(df, dd.DataFrame):
        df_pd = df.compute()
    else:
        df_pd = df
        
    X_scaled = scaler.fit_transform(df[proc.features_to_use])
    
    df['hora'] = df[map_cols['timestamp']].dt.hour
    df['dia_semana'] = df[map_cols['timestamp']].dt.day_name()
    
    score_columns_audit = []
    results_summary = []
    iso_masks_registry = {}
    lof_masks_registry = {}

    models_base = BaselineModels(X_scaled)
    optimizer = ThresholdOptimizer(config['parametros']['percentis_teste'])

    # ==========================================================================
    # 2. TREINAMENTO DOS MODELOS BASE
    # ==========================================================================

    # --- LOOP ISO FOREST ---
    print("\n" + "-" * 40)
    print("🌲 TREINANDO VARIAÇÕES ISO FOREST")
    for n_est in [100, 200]:
        tag = f"ISO_n{n_est}"
        print(f"   ↳ {tag}...")
        labels, scores, model = models_base.train_iso(n_estimators=n_est)
        
        # Salvar modelo (SEM IMPORT AQUI DENTRO)
        joblib.dump(model, f'outputs/models_saved/iso_n{n_est}.joblib')

        df[f'{tag}_score'] = -scores
        score_columns_audit.append(f'{tag}_score')

        df, metrics = optimizer.apply_dynamic_thresholds(df, f'{tag}_score', tag)
        results_summary.extend(metrics)
        iso_masks_registry[tag] = (df[f'{tag}_p95_label'] == 0)

    # --- LOOP LOF ---
    print("\n" + "-" * 40)
    print("🔍 TREINANDO VARIAÇÕES LOF")
    for k in [10, 20]:
        tag = f"LOF_k{k}_standard"
        print(f"   ↳ {tag}...")
        try:
            labels, scores = models_base.train_lof(k_neighbors=k)
            # LOF não tem método predict simples para salvar como pickle seguro
            
            df[f'{tag}_score'] = -scores
            score_columns_audit.append(f'{tag}_score')

            df, metrics = optimizer.apply_dynamic_thresholds(df, f'{tag}_score', tag)
            results_summary.extend(metrics)
            lof_masks_registry[tag] = (df[f'{tag}_p95_label'] == 0)
        except Exception as e:
            print(f"   ❌ Falha em {tag}: {e}")

    # ==========================================================================
    # 3. COMBINATORIAL GROUND TRUTH & LSTM LOOP
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🧠 TREINAMENTO LSTM MULTI-CENÁRIOS")
    print("=" * 80)

    gap_seconds = config.get('configuracoes_gerais', {}).get('gap_segmentation_seconds', 300)
    lstm_pipe = LSTMPipeline(
        X_data=X_scaled,
        vehicle_ids=df[map_cols['placa']].values,
        timestamps=df[map_cols['timestamp']].values,
        original_indices=df.index.values,
        window_size=config['parametros']['lstm_window_size'],
        max_gap_seconds=gap_seconds
    )

    for iso_name, iso_mask_inlier in iso_masks_registry.items():
        for lof_name, lof_mask_inlier in lof_masks_registry.items():
            gt_base_name = f"GT_({iso_name}+{lof_name})"
            print(f"\n⚡ Cenário: {gt_base_name}")
            
            # União (Rigorosa)
            mask_train_uniao = iso_mask_inlier & lof_mask_inlier
            lstm_name_uniao = f"LSTM_Uniao_{iso_name}_{lof_name}"
            print(f"   ↳ Treinando {lstm_name_uniao}...")
            
            mse_uniao, idx_uniao, model_uniao = lstm_pipe.train_evaluate(
                lstm_name_uniao, mask_train=mask_train_uniao, epochs=5)
            
            if model_uniao is not None:
                # SEM IMPORT OS AQUI
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
            
            mse_inter, idx_inter, model_inter = lstm_pipe.train_evaluate(
                lstm_name_inter, mask_train=mask_train_inter, epochs=5)
            
            if model_inter is not None:
                model_inter.save(f'outputs/models_saved/lstm_inter_{iso_name}_{lof_name}.h5')
            
            if mse_inter is not None and idx_inter is not None:
                df.loc[idx_inter, f'{lstm_name_inter}_score'] = mse_inter
                score_columns_audit.append(f'{lstm_name_inter}_score')
                df, metrics = optimizer.apply_dynamic_thresholds(df, f'{lstm_name_inter}_score', lstm_name_inter)
                results_summary.extend(metrics)

    # Treino Controle (Sujo - Sem GT)
    print(f"\n⚡ Cenário: Controle (Sem GT)")
    mse_sujo, idx_sujo, model_sujo = lstm_pipe.train_evaluate("LSTM_Sujo", mask_train=None, epochs=5)
    
    if model_sujo is not None:
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

    if hasattr(df, 'to_parquet'):
        df.to_parquet('outputs/master_table/resultado_final.parquet', index=False)
    else:
        pd.DataFrame(df).to_parquet('outputs/master_table/resultado_final.parquet', index=False)

    iso_metrics = [m for m in results_summary if m['Model'].startswith('ISO')]
    lof_metrics = [m for m in results_summary if m['Model'].startswith('LOF')]
    lstm_metrics = [m for m in results_summary if m['Model'].startswith('LSTM')]
    
    if iso_metrics: pd.DataFrame(iso_metrics).to_csv('outputs/metrics/iso_metrics.csv', index=False)
    if lof_metrics: pd.DataFrame(lof_metrics).to_csv('outputs/metrics/lof_metrics.csv', index=False)
    if lstm_metrics: pd.DataFrame(lstm_metrics).to_csv('outputs/metrics/lstm_metrics.csv', index=False)
    
    print("   ✅ Métricas exportadas.")

    # Segmentação
    id_cols = [map_cols['placa'], map_cols['timestamp'], map_cols['latitude'], map_cols['longitude']]
    # Garante que as colunas existem antes de filtrar
    cols_present = [c for c in id_cols if c in df.columns]
    
    df_iso = df[cols_present + [col for col in df.columns if col.startswith('ISO')]]
    df_lof = df[cols_present + [col for col in df.columns if col.startswith('LOF')]]
    df_lstm = df[cols_present + [col for col in df.columns if col.startswith('LSTM')]]
    
    # Função auxiliar para salvar
    def safe_save(dframe, fname):
        if hasattr(dframe, 'to_csv') and 'single_file' in dframe.to_csv.__code__.co_varnames:
            dframe.to_csv(fname, single_file=True, index=False)
        else:
            pd.DataFrame(dframe).to_csv(fname, index=False)

    safe_save(df_iso, 'outputs/metrics/iso_results.csv')
    safe_save(df_lof, 'outputs/metrics/lof_results.csv')
    safe_save(df_lstm, 'outputs/metrics/lstm_results.csv')

    print("   ✅ Resultados segmentados exportados.")
    print("\n✅ EXPERIMENTO FINALIZADO!")

if __name__ == "__main__":
    run_experiment()