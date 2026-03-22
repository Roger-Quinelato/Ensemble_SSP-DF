# Marca o diretorio 'utils' como um pacote Python.
__version__ = "1.0.0"
__author__ = "Equipe de ML - CIIA/CIN"

from src.utils.artifact_utils import (
    sha256_file,
    verify_artifact,
    verify_artifact_strict,
)
from src.utils.git_utils import format_model_version, get_git_info
from src.utils.tf_runtime import configure_tensorflow_runtime
