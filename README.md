# Ensemble_SSP-DF

## Objetivo
Pipeline de detecção de anomalias veiculares para apoio operacional da SSP-DF.

O sistema combina três famílias de modelos:
- Isolation Forest: detecção tabular global.
- HBOS: detecção tabular por histogramas univariados.
- GRU Autoencoder: detecção temporal por sequência (com LSTM apenas como fallback configurável).

A decisão final é feita por ensemble por família (ISO, HBOS, Temporal), com rastreabilidade por `run_id`, manifesto de artefatos e hash de integridade.

## Fluxo do Pipeline
O treinamento completo (`python -m src.main`) executa 5 etapas:

1. Ingestão e validação de schema.
2. Feature engineering por partição temporal (treino/val/test), com alinhamento de colunas sem leakage.
3. Treino dos modelos base (ISO + HBOS).
4. Treino dos modelos temporais (GRU) nos cenários Union/Inter/Baseline.
5. Exportação de artefatos: thresholds, manifesto, parquet final e relatório HTML.

## Modelos
| Família | Implementação | Escopo | Observação |
|---|---|---|---|
| ISO Forest | `sklearn.ensemble.IsolationForest` | Todos os registros | Usa `random_state` fixo para reprodutibilidade. |
| HBOS | `pyod.models.hbos.HBOS` | Todos os registros | Features independentes definidas em `config/feature_config.py`. |
| Temporal | `src.models.temporal_autoencoder.TemporalAutoencoder` | Registros com sequência válida | Padrão: GRU. LSTM é fallback configurável (`parametros.temporal.arch_type`). |

Dois scalers são usados por design:
- `scaler.joblib`: features tabulares de ISO/HBOS.
- `gru_scaler.joblib`: features temporais do GRU (inclui lat/lon).

## Instalação
### pip
```bash
pip install -r requirements.txt
```

### conda
```bash
conda env create -f environment.yml
conda activate sspdf-anomalias
```

## Execução de Treinamento
Comando base:
```bash
python -m src.main
```

Exemplos:
```bash
python -m src.main --input data/input/amostra_ssp.csv
python -m src.main --epochs 1 --seed 42
python -m src.main --output-dir outputs --config config_mapeamento.yaml
python -m src.main --input <caminho_arquivo.csv> --epochs 10 --seed 123 --output-dir <diretorio_saida>
```

Parâmetros CLI (`src/main.py`):
| Parâmetro | Default | Descrição |
|---|---|---|
| `--config` | `config_mapeamento.yaml` | YAML de configuração. |
| `--input` | `None` | CSV/Parquet de entrada. Se `None`, tenta `data/input/amostra_ssp.csv` e depois `.parquet`. |
| `--output-dir` | `outputs` | Diretório base; a execução cria `outputs/<run_id>/...`. |
| `--epochs` | `None` | Se ausente, usa `parametros.temporal.epochs` do YAML. |
| `--seed` | `42` | Seed global de reprodutibilidade. |
| `--verbose` | `False` | Ativa logs em nível debug. |

## Outputs Gerados
Estrutura por execução:
```text
outputs/<run_id>/
  models_saved/             # modelos, scalers, thresholds, manifesto
  metrics/                  # métricas e logs
  master_table/             # resultado_final.parquet
  relatorio_executivo.html
outputs/runs_index.csv      # índice de execuções
```

Artefatos principais:
| Arquivo | Local | Função |
|---|---|---|
| `resultado_final.parquet` | `outputs/<run_id>/master_table/` | Tabela consolidada com scores, labels e decisão final. |
| `models_manifest.json` | `outputs/<run_id>/models_saved/` | Inventário de artefatos com SHA256 e metadados git/run. |
| `thresholds_p90.json`, `thresholds_p95.json`, `thresholds_p99.json` | `outputs/<run_id>/models_saved/` | Thresholds serializados para inferência. |
| `perfil_dados.json` | `outputs/<run_id>/metrics/` | Perfil e metadados da execução. |
| `concordancia_modelos.csv` | `outputs/<run_id>/metrics/` | Concordância entre labels dos modelos. |
| `run_summary.json` | `outputs/<run_id>/metrics/` | Resumo estruturado da run. |
| `execution.log` | `outputs/<run_id>/metrics/` | Log textual da execução. |
| `vehicle_risk_ranking.csv` | `outputs/<run_id>/metrics/` | Ranking de risco por veículo. |
| `vehicle_coverage_report.csv` | `outputs/<run_id>/metrics/` | Cobertura de avaliação por veículo. |

## Inferência em Dados Novos
Comando real (`src/pipeline/inference.py`):
```bash
python -m src.pipeline.inference \
  --models-dir outputs/<run_id>/models_saved \
  --input <novos_dados.csv> \
  --output outputs/inferencia/
```

Parâmetros úteis:
- `--percentile` (default `95`)
- `--config` (default `config_mapeamento.yaml`)
- `--allow-legacy-manifest` (permite manifesto sem SHA256)

Modos de operação:
- Modo normal: thresholds (`thresholds_p<percentil>.json`) carregados do treino.
- Modo degradado: thresholds ausentes -> recalibra nos dados novos com warning (não lança exceção).

Temporal na inferência:
- Se `gru_scaler.joblib` ausente, a família temporal é pulada com warning.
- A inferência continua com ISO/HBOS (sem abortar).

## Contrato de Artefatos
Obrigatórios para inferência:
- `models_manifest.json`
- `scaler.joblib`
- `iso_*.joblib`
- `hbos_*.joblib`
- `thresholds_p95.json`

Condicionais (família temporal):
- `gru_scaler.joblib`
- `temporal_*.h5`

Operacionais (pós-processamento e auditoria):
- `concordancia_modelos.csv`
- `vehicle_risk_ranking.csv`
- `run_summary.json`

## Schema de Entrada
Validação implementada em `src/data/schema.py` (Pandera).

Colunas mínimas esperadas (após mapeamento):
- `placa`
- `timestamp`
- `latitude`
- `longitude`

Regras principais:
- `latitude` em `[-16.5, -15.0]`
- `longitude` em `[-48.5, -47.0]`
- `timestamp` entre `2020-01-01` e `2030-12-31`
- Dataset com pelo menos 1 registro (suporta micro-batch)

Formatos de entrada suportados:
- `.csv`
- `.parquet`

## Reprodutibilidade
Com `--seed 42`, o pipeline fixa:
- `PYTHONHASHSEED`
- `TF_DETERMINISTIC_OPS`
- `random.seed(...)`
- `numpy.random.seed(...)`
- `tf.random.set_seed(...)`

Limitação prática:
- Em GPU, operações podem não ser bit-perfect entre execuções.

## Rastreabilidade
A rastreabilidade da run é registrada em:
- `outputs/runs_index.csv`
- `outputs/<run_id>/metrics/run_summary.json`
- `outputs/<run_id>/models_saved/models_manifest.json`

`models_manifest.json` inclui:
- `model_version`
- hashes SHA256 dos artefatos
- metadados git (`commit_hash`, branch, dirty flag, timestamp)

Se o treino ocorrer com working tree sujo (`is_dirty=true`), o pipeline gera warning explícito.

## Testes
Executar suíte:
```bash
pytest tests/ -v
```

Testes herméticos (sem depender de run prévia):
- `tests/test_regression_c1_c2.py`
- `tests/test_schema.py`
- `tests/test_git_utils.py`
- `tests/test_artifact_integrity.py`

Testes que dependem de artefatos de run prévia:
- `tests/test_integration_train_infer.py`
- `tests/test_stability.py`
- `tests/test_outputs_generated.py`
- `tests/test_manifest_portability.py`

## Docker
Build:
```bash
docker build -t sspdf-anomalias .
```

Execução (treino):
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  sspdf-anomalias
```

Se `docker-compose.yml` existir:
```bash
docker compose run train
docker compose run infer
```

## Limitações Conhecidas
- Em GPU, determinismo bit-a-bit não é garantido para todas as operações.
- O sistema é não supervisionado: sem ground truth rotulado, as métricas operacionais são de consistência/concordância, não de performance supervisionada (precision/recall/F1 reais).
