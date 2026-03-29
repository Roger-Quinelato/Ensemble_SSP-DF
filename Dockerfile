FROM python:3.10-slim

WORKDIR /app

# Variaveis de ambiente
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Toolchain pinada para builds mais reprodutiveis
ARG PIP_VERSION=24.3.1
ARG SETUPTOOLS_VERSION=75.6.0
ARG WHEEL_VERSION=0.45.1

# Cache de dependencias: copiar requirements primeiro
COPY requirements.txt .
RUN python -m pip install --upgrade \
    pip==${PIP_VERSION} \
    setuptools==${SETUPTOOLS_VERSION} \
    wheel==${WHEEL_VERSION} && \
    python -m pip install -r requirements.txt

# Usuario nao-root para execucao
RUN useradd -m -u 1000 sspdf

# Copiar codigo-fonte e YAMLs de configuracao da raiz
COPY config_mapeamento*.yaml ./
COPY src/ src/
COPY config/ config/

# Preparar diretorios de escrita para execucao com usuario nao-root
RUN mkdir -p /app/data/input /app/outputs && \
    chown -R sspdf:sspdf /app

USER sspdf
HEALTHCHECK --interval=90s --timeout=5s --retries=3 CMD python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('src.main') else 1)"
CMD ["python", "-m", "src.main"]
