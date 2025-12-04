# SSP-DF Anomalias - Pipeline de Detecção

## 🚨 Detecção de Anomalias em Dados de Câmeras de Segurança

Este repositório foi desenvolvido para a SSP-DF realizar o treinamento e análise de anomalias em dados de câmeras de segurança, respeitando a LGPD e permitindo total controle interno dos dados.

---

## 🧠 Motivação

- Detectar padrões suspeitos e anomalias em grandes volumes de dados de mobilidade urbana.
- Pipeline robusto, escalável e modular, pronto para uso em ambiente institucional.
- Totalmente adaptado para processamento distribuído com Dask.

---

## 🔄 Fluxo do Pipeline

1. **Ingestão e Mapeamento:** Carregamento dos dados brutos e adaptação do schema via `config_mapeamento.yaml`.
2. **Sanitização e Feature Engineering:** Criação de features temporais, espaciais e contextuais.
3. **Modelagem Base:** Treinamento de Isolation Forest e LOF (standard/novelty).
4. **Ground Truths Sintéticos:** Combinação de máscaras para cenários de normalidade.
5. **Modelagem Sequencial:** LSTM Autoencoder para detecção de anomalias temporais.
6. **Exportação de Resultados:** Relatórios detalhados e métricas segmentadas por família de modelo.

---

## ✨ Principais Funcionalidades

- Processamento escalável com Dask para grandes volumes.
- Modularidade: fácil adaptação para novos cenários/modelos.
- Exportação automática de relatórios e métricas.
- Testes unitários para todos os módulos principais.
- Logging detalhado para auditoria e rastreabilidade.

---

## ⚡ Instalação e Ambiente

1. **Clone o repositório:**
   ```bash
   git clone <url-do-repositorio>
   cd Ensemble_SSP-DF
   ```
2. **Crie o ambiente Conda:**
   ```bash
   conda env create -f environment.yml
   conda activate sspdf-anomalias
   ```
   Ou instale via pip (terminal de comando):
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure o arquivo de mapeamento:**
   # Está previamente configurada nos padrões da SSP-DF, mas estará mais simples caso haja adição de features.
   - Edite `config_mapeamento.yaml` conforme o schema dos dados da SSP-DF.

---

## ▶️ Execução Rápida

```bash
python src/main.py
```

- Os resultados e relatórios serão gerados em `outputs/reports/` e `outputs/metrics/`.

# reports tem os compilados
---

## 📁 Estrutura de Pastas

```
Ensemble_SSP-DF/
├── src/                # Código principal do pipeline
├── data/input/         # Dados brutos para treinamento
├── outputs/            # Relatórios, métricas e logs
├── config_mapeamento.yaml
├── requirements.txt
├── environment.yml
├── tests/              # Testes unitários
```

---

## 📊 Relatórios Gerados

- **iso_metrics.csv, lof_metrics.csv, lstm_metrics.csv:** Métricas dos modelos.
- **iso_results.csv, lof_results.csv, lstm_results.csv:** Resultados segmentados.
- **describe_*:** Estatísticas detalhadas por variação de modelo.
- **InformacaoInicial_BaseDados.json:** Perfil da base de dados.
- **DescricaoInicial_BaseDados.csv:** Métricas da base crua.

---

## 🧪 Testes

- Execute todos os testes com:
  ```bash
  pytest tests/
  ```
- Testes unitários garantem robustez e confiabilidade do pipeline.

---

## 🚀 Escalabilidade

- O pipeline utiliza Dask para processar milhões de registros de forma distribuída.
- Recomenda-se rodar em máquinas com múltiplos núcleos ou clusters para máximo desempenho.

---

## 👥 Contato e Créditos

- Desenvolvido por Equipe de ML - CIIA/CIN
- Para dúvidas e suporte, entre em contato com o responsável técnico do projeto: Roger Quinelato (rogerdiasquinelato@gmail.com).

---

## 🏆 Pronto para uso institucional!

Este projeto foi pensado para ser facilmente adaptado, auditado e expandido conforme as necessidades da SSP-DF. Todos os passos são rastreáveis e documentados para garantir transparência e segurança.
