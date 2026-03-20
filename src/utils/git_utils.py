"""
Utilitarios para obter informacoes do repositorio git no momento do treinamento.
"""
import logging
import subprocess

logger = logging.getLogger("sspdf")


def get_git_info() -> dict:
    """
    Obtem informacoes do repositorio git atual.

    Returns:
        Dict com:
          - commit_hash: SHA do commit HEAD (ou "unknown" se nao for repositorio git)
          - commit_short: primeiros 8 caracteres do hash
          - branch: nome da branch atual
          - is_dirty: True se ha mudancas nao commitadas
          - commit_message: primeira linha do commit message
          - commit_timestamp: timestamp ISO do commit
        Nunca levanta excecao - retorna valores "unknown" se git nao disponivel.
    """

    def _run(cmd):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=".",
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

    commit_hash = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    is_dirty_str = _run(["git", "status", "--porcelain"])
    msg = _run(["git", "log", "-1", "--format=%s"])
    timestamp = _run(["git", "log", "-1", "--format=%cI"])

    if not commit_hash:
        logger.warning(
            "git nao disponivel ou diretorio nao e repositorio git. "
            "model_version nao tera commit hash."
        )
        return {
            "commit_hash": "unknown",
            "commit_short": "unknown",
            "branch": "unknown",
            "is_dirty": None,
            "commit_message": "unknown",
            "commit_timestamp": "unknown",
        }

    is_dirty = bool(is_dirty_str)
    if is_dirty:
        logger.warning(
            "AVISO: ha mudancas nao commitadas no repositorio (working tree dirty). "
            f"O modelo foi treinado com codigo modificado que NAO esta no commit {commit_hash[:8]}. "
            "Para auditabilidade completa, commitar todas as mudancas antes do treinamento."
        )

    return {
        "commit_hash": commit_hash,
        "commit_short": commit_hash[:8],
        "branch": branch or "unknown",
        "is_dirty": is_dirty,
        "dirty_warning": "TREINADO COM CODIGO NAO COMMITADO" if is_dirty else None,
        "commit_message": msg or "unknown",
        "commit_timestamp": timestamp or "unknown",
    }


def format_model_version(git_info: dict, run_id: str) -> str:
    """
    Formata uma string de versao do modelo para exibicao.
    """
    version = f"run_{run_id}_commit_{git_info['commit_short']}"
    if git_info.get("is_dirty"):
        version += "_DIRTY"
    return version
