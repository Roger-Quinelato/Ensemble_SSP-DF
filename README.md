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

## Guia de Hiperparâmetros
O pipeline foi desenhado para que a maior parte dos ajustes operacionais seja feita no YAML (`config_mapeamento.yaml`), e nao no codigo. O parametro mais sensivel no dia a dia costuma ser `epochs`, mas ele nao deve ser alterado isoladamente: o efeito real depende da janela temporal, do volume de dados por veiculo e da cobertura de sequencias validas.

### Como `--epochs` funciona na pratica
- Se `--epochs` for informado na CLI, ele tem precedencia sobre o YAML.
- Se `--epochs` nao for informado, o pipeline usa `parametros.temporal.epochs` do arquivo de configuracao.
- No estado atual do projeto, o valor padrao operacional no YAML e `5`.

Exemplos reais:
```bash
python -m src.main --epochs 1
python -m src.main --epochs 10 --seed 42
python -m src.main --config config_mapeamento.yaml
```

Leitura operacional:
- `epochs=1`: smoke test, validacao rapida de fluxo e geracao de artefatos.
- `epochs=5`: baseline atual do projeto; bom equilibrio entre custo e estabilidade.
- `epochs>5`: util quando ha mais dados e cobertura temporal suficiente, mas aumenta tempo de treino e pode ampliar risco de overfitting em bases pequenas.

### Parametros centrais do YAML
Valores atuais em `config_mapeamento.yaml`:
- `parametros.split_ratios`: `train=0.6`, `validation=0.2`, `test=0.2`
- `parametros.percentis_teste`: `[90, 95, 99]`
- `parametros.isolation_forest.n_estimators`: `[100, 200]`
- `parametros.hbos.n_bins`: `[10, 20]`
- `parametros.temporal.arch_type`: `gru`
- `parametros.temporal.window_size`: `3`
- `parametros.temporal.epochs`: `5`
- `parametros.temporal.batch_size`: `64`
- `parametros.temporal.dropout`: `0.2`
- `configuracoes_gerais.gap_segmentation_seconds`: `1800`

### O que cada parametro muda
| Parametro | Efeito principal | Quando ajustar |
|---|---|---|
| `epochs` | Numero de passagens de treino do modelo temporal | Ajuste fino de convergencia do GRU |
| `window_size` | Tamanho da sequencia temporal | Quando o padrao anomalo depende de janelas mais curtas ou mais longas |
| `gap_segmentation_seconds` | Quebra sequencias com gaps longos | Quando a frequencia de GPS muda entre ambientes |
| `n_estimators` (ISO) | Robustez e custo computacional do Isolation Forest | Quando quiser comparar estabilidade de variantes ISO |
| `n_bins` (HBOS) | Granularidade dos histogramas do HBOS | Quando quiser comparar sensibilidade local do HBOS |
| `percentis_teste` | Thresholds operacionais de alerta | Quando precisar calibrar severidade de triagem |
| `seed` | Reprodutibilidade da execucao | Deve permanecer fixo em producao para auditoria |

### Recomendacoes de uso
- Para validar instalacao, CI local ou estrutura de outputs: usar `--epochs 1`.
- Para treino operacional reprodutivel: manter `--seed 42` e registrar sempre o `run_id`.
- Para comparar configuracoes: alterar um grupo de parametros por vez e comparar `runs_index.csv`, `model_selection_val.csv` e `models_manifest.json`.
- Para bases pequenas ou com baixa cobertura temporal: nao aumentar `epochs` antes de verificar quantas sequencias validas o GRU realmente conseguiu formar.

### O que observar no log
Durante o treino, os logs relevantes para diagnostico sao:
- `--epochs nao informado. Usando valor do YAML: ...`
- `--epochs=<N> (explicito via CLI). Valor do YAML (...) ignorado.`
- `Sequencias criadas: ...`
- `Sequencias de treino STRICT (todos elementos no treino): ...`
- `Thresholds calibrados no TREINO para ...`

Essas mensagens ajudam a responder, em auditoria, tres perguntas importantes:
- qual configuracao efetivamente foi usada;
- quanto dado temporal foi realmente treinavel;
- em que distribuicao os thresholds foram calibrados.

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
| `relatorio_executivo.html` | `outputs/<run_id>/` | Relatório visual consolidado para leitura executiva e auditoria. |

O artefato mais adequado para apresentação institucional costuma ser `outputs/<run_id>/relatorio_executivo.html`, porque ele resume KPIs, ranking de veículos, concordância, cobertura e limitações metodológicas em um único documento.

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

## Relatório HTML Executivo
Ao final do treinamento, o pipeline tenta gerar automaticamente:

```text
outputs/<run_id>/relatorio_executivo.html
```

Como abrir:
- navegador local (Chrome, Edge ou equivalente);
- sem depender de notebook, planilha ou ferramenta externa para leitura inicial.

O que o relatório contém:
- KPIs consolidados da run;
- total de alertas e taxa de alerta;
- ranking de risco por veículo;
- cobertura de avaliação, incluindo cobertura temporal;
- distribuições de score por família de modelo;
- matriz de concordância entre modelos;
- seção metodológica com disclaimers explícitos;
- referência ao `run_id` da execução.

Como interpretar corretamente:
- `ensemble_alert = 1` significa que o registro foi considerado anômalo pelo ensemble configurado; nao significa confirmação de irregularidade.
- ausência de alerta nao equivale a normalidade comprovada.
- cobertura temporal inferior a 100% significa que parte dos registros nao formou sequência suficiente para o GRU.
- concordância entre modelos mede consistência interna, nao performance supervisionada.

Limitação interpretativa:
- o relatório é um artefato de triagem analítica e apoio à decisão;
- ele não substitui validação supervisionada, anotação humana ou confirmação operacional em campo;
- sem ground truth rotulado, os gráficos e métricas devem ser lidos como evidência estatística e de rastreabilidade, não como prova de acurácia final.

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
> Nota de validacao: o Docker nao foi validado localmente neste ambiente por indisponibilidade de engine (`docker` nao instalado/ativo no host de validacao).

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
