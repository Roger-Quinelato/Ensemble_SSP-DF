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

# Usuario nao-root para execucao
RUN useradd -m -u 1000 sspdf

# Copiar codigo-fonte
COPY config_mapeamento.yaml .
COPY src/ src/
COPY config/ config/

# Criar diretorios de saida
RUN mkdir -p data/input

# Ponto de entrada correto
USER sspdf
CMD ["python", "-m", "src.main"]
