# Ensemble_SSP-DF

## Objetivo
Este repositório implementa um pipeline de detecção de anomalias veiculares para apoio analítico e operacional da SSP-DF.

O sistema combina três famílias de modelos:
- Isolation Forest: detecção tabular global.
- HBOS: detecção tabular por histogramas univariados.
- GRU Autoencoder: detecção temporal por sequência (com LSTM apenas como fallback configurável).

A decisão final é realizada por ensemble por família (ISO, HBOS, Temporal), com rastreabilidade por `run_id`, manifesto de artefatos e hash de integridade.

## Fluxo do Pipeline
O treinamento completo (`python -m src.main`) executa 5 etapas:

1. Ingestão e validação de schema.
2. Feature engineering por partição temporal (treino/val/test), com alinhamento de colunas sem leakage.
3. Treino dos modelos base (ISO + HBOS).
4. Treino dos modelos temporais (GRU) nos cenários Union/Inter/Baseline.
5. Exportação de artefatos: thresholds, manifesto, parquet final e relatório HTML.

## Diagrama da Arquitetura Operacional
O diagrama abaixo representa o fluxo efetivamente executado pelo projeto, incluindo treino, persistência de artefatos, inferência e camada de auditoria:

```mermaid
flowchart TD
    A["`src/main.py`
    CLI: --config, --input, --output-dir, --epochs, --seed"] --> B["`run_experiment()`
    `src/pipeline/experiment_runner.py`"]

    B --> C["Carga padronizada
    `DataProcessor.load_and_standardize()`
    + `validate_input()`"]
    C --> D["Split temporal 60/20/20
    treino / validacao / teste"]
    D --> E["Feature engineering por particao
    + alinhamento de colunas `RA_*`
    + `scaler.joblib`"]
    E --> F["Preparacao por familia
    ISO / HBOS / GRU
    + `gru_scaler.joblib`"]

    F --> G["Modelos base
    `BaselineModels`
    `iso_n*.joblib`
    `hbos_bins*.joblib`"]
    G --> H["Thresholds por treino
    p90 / p95 / p99"]
    G --> I["Mascaras de inlier
    para cenarios temporais"]

    I --> J["Modelos temporais
    `TemporalAutoencoder`
    cenarios Union / Inter / Baseline
    `temporal_*.h5`"]
    J --> H

    H --> K["`export_results()`
    ensemble por familia
    ranking por veiculo
    concordancia
    estabilidade em validacao"]
    K --> L["`outputs/<run_id>/models_saved`
    manifesto + hashes SHA256
    scalers + thresholds + modelos"]
    K --> M["`outputs/<run_id>/metrics`
    `execution.log`
    `run_summary.json`
    `perfil_dados.json`
    `concordancia_modelos.csv`
    `vehicle_risk_ranking.csv`
    `vehicle_coverage_report.csv`
    `model_selection_val.csv`"]
    K --> N["`outputs/<run_id>/master_table`
    `resultado_final.parquet`"]
    K --> O["`src/outputs/report_generator.py`
    `relatorio_executivo.html`"]
    K --> P["`outputs/runs_index.csv`"]

    L --> Q["`src/pipeline/inference.py`
    carrega manifesto, scalers, thresholds e modelos"]
    Q --> R["Inferencia oficial
    usa thresholds do treino
    valida hash quando disponivel"]
    Q --> S["Compatibilidade / degradado
    sem manifesto: descobre por nome
    sem thresholds: recalibra nos dados novos
    sem `gru_scaler`: pula familia temporal"]
    R --> T["`outputs_inference/`
    `inference_result.parquet`
    `metrics/alertas_ensemble.csv`
    `metrics/vehicle_risk_ranking.csv`"]
    S --> T

    M --> U["Suite de testes
    contrato de artefatos
    regressao semantica
    smoke de relatorio
    treino -> inferencia"]
    L --> U
    N --> U
    O --> U
```

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

## Guia Operacional por Modalidade (passo a passo)
Esta seção consolida o procedimento de execução ponta a ponta em ambiente local: treinamento, validação de artefatos e inferência.

### Pré-condição única (todas as modalidades)
Mantenha o arquivo de entrada no caminho oficial `data/input/amostra_ssp.csv`.
Para treinamento local da SSP-DF, recomenda-se substituir apenas o conteúdo do arquivo, preservando o mesmo nome para manter compatibilidade operacional.

### 1) Treino completo e o que observar no log
Durante o treinamento completo, o log deve registrar, em sequência:
1. `ETAPA 1`: carga, schema e split temporal.
2. `ETAPA 2`: features por família e gravação de `gru_scaler.joblib`.
3. `ETAPA 3`: treino ISO/HBOS com thresholds `p90/p95/p99`.
4. `ETAPA 4`: treino temporal `Union/Inter/Baseline`.
5. `ETAPA 5`: exportação final, manifesto, parquet e relatório HTML.

Indicadores de execução válida:
- `THRESHOLDS SERIALIZADOS: [90, 95, 99]`
- `Manifesto de modelos salvo`
- `Relatorio HTML gerado`
- `EXPERIMENTO FINALIZADO`

### 2) Comandos por modalidade
#### Bash (Linux/macOS)
```bash
python -m src.main \
  --input data/input/amostra_ssp.csv \
  --epochs 50 \
  --seed 42 \
  --output-dir outputs
```

#### PowerShell (Windows)
```powershell
python -m src.main `
  --input data/input/amostra_ssp.csv `
  --epochs 50 `
  --seed 42 `
  --output-dir outputs
```

#### CMD (Windows)
```cmd
python -m src.main --input data/input/amostra_ssp.csv --epochs 50 --seed 42 --output-dir outputs
```

### 3) Análise de execução dos modelos (como interpretar)
Após a execução, realize a análise técnico-operacional com base em `outputs/<run_id>/metrics/execution.log`:
- **ISO/HBOS**: confirmar calibração de thresholds no treino e aplicação no conjunto consolidado.
- **Temporal (GRU)**: cobertura temporal abaixo de 100% pode ocorrer quando não há sequência válida por veículo.
- **Ensemble final**: validar a seção `CAMADA DE DECISAO FINAL DO ENSEMBLE` e a taxa final de alertas.
- **Seleção de configuração**: analisar `model_selection_val.csv` para estabilidade entre treino e validação.

Artefatos prioritários para auditoria:
- `outputs/<run_id>/master_table/resultado_final.parquet`
- `outputs/<run_id>/models_saved/models_manifest.json`
- `outputs/<run_id>/models_saved/thresholds_p95.json`
- `outputs/<run_id>/relatorio_executivo.html`
- `outputs/<run_id>/metrics/run_summary.json`

### 4) Inferência (mesmo run treinado)
#### Bash (Linux/macOS)
```bash
python -m src.pipeline.inference \
  --models-dir outputs/<run_id>/models_saved \
  --input data/input/amostra_ssp.csv \
  --output outputs/inferencia_<run_id> \
  --percentile 95
```

#### PowerShell (Windows)
```powershell
python -m src.pipeline.inference `
  --models-dir outputs/<run_id>/models_saved `
  --input data/input/amostra_ssp.csv `
  --output outputs/inferencia_<run_id> `
  --percentile 95
```

#### CMD (Windows)
```cmd
python -m src.pipeline.inference --models-dir outputs\<run_id>\models_saved --input data/input/amostra_ssp.csv --output outputs/inferencia_<run_id> --percentile 95
```

## Guia de Hiperparâmetros
O pipeline foi desenhado para que a maior parte dos ajustes operacionais seja feita no YAML (`config_mapeamento.yaml`), e não diretamente no código. O parâmetro mais sensível no uso diário tende a ser `epochs`, mas sua alteração deve considerar conjuntamente janela temporal, volume de dados por veículo e cobertura de sequências válidas.

### Como `--epochs` funciona na pratica
- Se `--epochs` for informado na CLI, ele tem precedência sobre o YAML.
- Se `--epochs` não for informado, o pipeline usa `parametros.temporal.epochs` do arquivo de configuração.
- No estado atual do projeto, o valor operacional padrão no YAML é `5`.

Exemplos reais:
```bash
python -m src.main --epochs 1
python -m src.main --epochs 10 --seed 42
python -m src.main --config config_mapeamento.yaml
```

Leitura técnico-operacional:
- `epochs=1`: smoke test e validação rápida de fluxo/artefatos.
- `epochs=5`: baseline atual do projeto, com equilíbrio entre custo computacional e estabilidade.
- `epochs>5`: recomendado quando há maior volume de dados e cobertura temporal suficiente; aumenta tempo de processamento e pode elevar risco de overfitting em bases pequenas.

### Parâmetros centrais do YAML
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
| Parâmetro | Efeito principal | Quando ajustar |
|---|---|---|
| `epochs` | Número de passagens de treino do modelo temporal | Ajuste fino de convergência do GRU |
| `window_size` | Tamanho da sequência temporal | Quando o padrão anômalo depende de janelas mais curtas ou mais longas |
| `gap_segmentation_seconds` | Quebra sequências com gaps longos | Quando a frequência de GPS muda entre ambientes |
| `n_estimators` (ISO) | Robustez e custo computacional do Isolation Forest | Quando quiser comparar estabilidade de variantes ISO |
| `n_bins` (HBOS) | Granularidade dos histogramas do HBOS | Quando quiser comparar sensibilidade local do HBOS |
| `percentis_teste` | Thresholds operacionais de alerta | Quando precisar calibrar severidade de triagem |
| `seed` | Reprodutibilidade da execução | Deve permanecer fixo em produção para auditoria |

### Recomendações de uso
- Para validar instalação, CI local ou estrutura de outputs: usar `--epochs 1`.
- Para treino operacional reprodutível: manter `--seed 42` e registrar sempre o `run_id`.
- Para comparar configurações: alterar um grupo de parâmetros por vez e comparar `runs_index.csv`, `model_selection_val.csv` e `models_manifest.json`.
- Para bases pequenas ou com baixa cobertura temporal: não aumentar `epochs` antes de verificar quantas sequências válidas o GRU conseguiu formar.

### O que observar no log
Durante o treino, os logs relevantes para diagnóstico são:
- `--epochs nao informado. Usando valor do YAML: ...`
- `--epochs=<N> (explicito via CLI). Valor do YAML (...) ignorado.`
- `Sequencias criadas: ...`
- `Sequencias de treino STRICT (todos elementos no treino): ...`
- `Thresholds calibrados no TREINO para ...`

Essas mensagens apoiam, em auditoria, três perguntas centrais:
- qual configuração efetivamente foi utilizada;
- quanto dado temporal foi efetivamente treinável;
- em qual distribuição os thresholds foram calibrados.

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

Para apresentação institucional, o artefato prioritário é `outputs/<run_id>/relatorio_executivo.html`, pois consolida KPIs, ranking de veículos, concordância, cobertura e limitações metodológicas em um único documento.

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
- Modo compatibilidade: se `models_manifest.json` estiver ausente, a inferência tenta descobrir `iso_*.joblib`, `hbos_*.joblib`, `temporal_*.h5` e scalers diretamente por nome de arquivo.

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
Obrigatórios no fluxo oficial de inferência rastreável:
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

Compatibilidade e fallback ainda aceitos pelo código:
- ausência de `models_manifest.json`: descoberta dos artefatos por convenção de nome.
- ausência de `thresholds_p95.json`: recalibração nos dados novos (modo degradado, não recomendado para produção).
- ausência de `gru_scaler.joblib`: inferência segue apenas com ISO/HBOS.

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
- `tests/test_data_processor.py`
- `tests/test_ensemble_decision.py`
- `tests/test_evaluation.py`
- `tests/test_model_selection.py`
- `tests/test_models_base.py`

Testes que dependem de artefatos de run prévia:
- `tests/test_integration_train_infer.py`
- `tests/test_stability.py`
- `tests/test_pipeline_flow.py`
- `tests/test_outputs_generated.py`
- `tests/test_manifest_portability.py`
- `tests/test_threshold_serialization.py`
- `tests/test_report_smoke.py`

Testes específicos e opcionais:
- `tests/test_models_deep.py`: cobre o `TemporalAutoencoder` e fluxo mínimo da família temporal.
- `tests/test_inference.py` e `tests/test_inference_smoke.py`: validam a inferência real e seus fallbacks.
- `tests/test_epoch_selection.py`: teste lento, voltado a exploração de épocas; não deve ser tratado como contrato mínimo de CI.

## Docker
Build:
```bash
docker build -t sspdf-anomalias .
```

Execução (treino) com `docker run`:
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  sspdf-anomalias \
  python -m src.main --input /app/data/input/amostra_ssp.csv --epochs 50 --seed 42 --output-dir /app/outputs/docker_train
```

PowerShell equivalente:
```powershell
docker run --rm `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\outputs:/app/outputs" `
  sspdf-anomalias `
  python -m src.main --input /app/data/input/amostra_ssp.csv --epochs 50 --seed 42 --output-dir /app/outputs/docker_train
```

`docker compose` (treino e inferência):
```bash
docker compose run --rm train --input /app/data/input/amostra_ssp.csv --epochs 50 --seed 42 --output-dir /app/outputs/docker_train
docker compose run --rm infer
```

Se quiser fixar variáveis do serviço `infer`, copie `.env.example` para `.env` e ajuste:
- `RUN_ID`
- `INPUT_FILE`
- `INFER_OUTPUT_DIR`

## Limitações Conhecidas
- Em GPU, determinismo bit-a-bit não é garantido para todas as operações.
- O sistema é não supervisionado: sem ground truth rotulado, as métricas operacionais são de consistência/concordância, não de performance supervisionada (precision/recall/F1 reais).
