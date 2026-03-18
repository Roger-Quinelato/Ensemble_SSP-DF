FROM python:3.10-slim

WORKDIR /app

# Variaveis de ambiente
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Cache de dependencias: copiar requirements primeiro
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar codigo-fonte
COPY config_mapeamento.yaml .
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

# Criar diretorios de saida
RUN mkdir -p data/input outputs/metrics outputs/models_saved \
    outputs/master_table outputs/logs outputs/reports

# Ponto de entrada correto
CMD ["python", "src/main.py"]
