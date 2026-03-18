# SSP-DF Anomalias - Pipeline de Detecção

## 🚨 Detecção de Anomalias em Dados de Câmeras de Segurança

Pipeline desenvolvido para a SSP-DF (Secretaria de Segurança Pública do Distrito Federal)
para treinamento e análise de anomalias em dados de câmeras de segurança.

Respeita a LGPD e permite controle interno dos dados.

---

## 🧠 Motivação

- Detectar padrões suspeitos e anomalias em grandes volumes de dados de mobilidade urbana.
- Pipeline robusto, auditável e modular, para uso institucional.
- Ensemble de modelos: detecção estatística + aprendizado profundo temporal.

---

## 🔄 Fluxo do Pipeline

1. **Ingestão e Mapeamento:** carregamento dos dados brutos e adaptação do schema via `config_mapeamento.yaml`.
2. **Sanitização e Feature Engineering:** criação de features temporais (hora cíclica, dia_sem, feriados), espaciais (velocidade, distância, aceleração) e contextuais (região administrativa).
3. **Modelagem Base:**
   - **Isolation Forest:** detecção global de outliers (`n_estimators`: 100, 200).
   - **HBOS:** detecção univariada por histograma (`n_bins`: 10, 20).
4. **Ground Truth Sintético:** combinação de máscaras ISO × HBOS (cenários rigoroso/permissivo).
5. **Modelagem Sequencial:** autoencoder temporal com GRU (default) ou LSTM via factory (`TemporalAutoencoder`).
6. **Exportação:** métricas, scores, tabela consolidada e relatórios segmentados.

---

## 🧩 Arquitetura de Modelos

| Modelo | Tipo | Propósito | Parâmetros |
|---|---|---|---|
| Isolation Forest | Ensemble de árvores | Detecção global de outliers | `n_estimators=[100,200]`, `contamination=auto` |
| HBOS | Histograma univariado | Detecção por distribuição de features | `n_bins=[10,20]`, `contamination=0.1` |
| Temporal Autoencoder (GRU/LSTM) | Rede recorrente | Detecção temporal em sequências | `window_size` via config, `epochs` via CLI |

---

## ⚡ Instalação

```bash
git clone <url-do-repositorio>
cd Ensemble_SSP-DF
pip install -r requirements.txt
```

Opcional (Conda):

```bash
conda env create -f environment.yml
conda activate sspdf-anomalias
```

---

## ▶️ Execução

```bash
# recomendado (modo módulo)
python -m src.main

# também suportado
python src/main.py

# parâmetros customizados
python -m src.main --input data/input/dados_producao.csv --epochs 50
python -m src.main --config meu_config.yaml --seed 123 --output-dir resultados
python -m src.main --help
```

---

## 🧪 Parâmetros CLI

- `--config`: caminho do YAML de configuração.
- `--input`: caminho de dados de entrada (`.csv`/`.parquet`).
- `--output-dir`: diretório base de saída.
- `--epochs`: épocas de treino dos modelos temporais.
- `--seed`: seed global.
- `--verbose`: flag de verbosidade.

---

## 📁 Estrutura de Pastas

```text
Ensemble_SSP-DF/
├── src/
│   ├── main.py
│   ├── data/
│   │   └── data_processor.py
│   ├── models/
│   │   ├── models_base.py
│   │   ├── temporal_autoencoder.py
│   │   ├── models_deep.py
│   │   └── models_deep2.py
│   ├── pipeline/
│   │   └── experiment_runner.py
│   └── utils/
│       ├── evaluation.py
│       ├── logger_utils.py
│       └── organizacao_arquivos.py
├── config/
│   └── feature_config.py
├── data/input/
├── outputs/
│   ├── metrics/
│   ├── models_saved/
│   ├── master_table/
│   ├── logs/
│   └── reports/
├── tests/
├── config_mapeamento.yaml
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

---

## 📊 Arquivos Gerados

| Arquivo | Descrição |
|---|---|
| `outputs/metrics/iso_metrics.csv` | Métricas de thresholds do Isolation Forest |
| `outputs/metrics/hbos_metrics.csv` | Métricas de thresholds do HBOS |
| `outputs/metrics/lstm_metrics.csv` | Métricas de thresholds dos modelos temporais |
| `outputs/metrics/comparativo_completo.csv` | Comparativo de modelos vs GT de referência |
| `outputs/metrics/perfil_dados.json` | Perfil estatístico da base |
| `outputs/metrics/iso_results.csv` | Scores/labels ISO |
| `outputs/metrics/hbos_results.csv` | Scores/labels HBOS |
| `outputs/metrics/lstm_results.csv` | Scores/labels temporais |
| `outputs/master_table/resultado_final.parquet` | Tabela consolidada final |
| `outputs/models_saved/*.joblib` | Modelos ISO/HBOS + scaler |
| `outputs/models_saved/*.h5` | Modelos temporais |
| `outputs/logs/execution.log` | Log estruturado (arquivo + console) |

---

## 🐳 Docker

```bash
docker build -t sspdf-anomalias .
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs sspdf-anomalias
```

**LGPD:** não inclua dados sensíveis na imagem. Monte dados via volume.

---

## 🧪 Testes

```bash
pytest tests/ -v
```

---

## ⚠️ Limitações Conhecidas

- Pós-processamento em `organizacao_arquivos.py` ainda depende de caminhos fixos em `outputs/`.
- Não há validação formal de schema de entrada (recomendado: Pandera/Great Expectations).
- Tracking de experimentos (MLflow) não está integrado.
- Reprodutibilidade em GPU pode variar (determinismo de TF não é 100% bit-perfect em GPU).
