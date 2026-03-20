"""
Utilitarios para verificacao de integridade de artefatos.
"""

import hashlib
import os


def sha256_file(path: str, chunk_size: int = 65536) -> str:
    """
    Calcula SHA256 de um arquivo de forma eficiente (leitura em chunks).

    Args:
        path: Caminho absoluto do arquivo.
        chunk_size: Tamanho do chunk de leitura em bytes (default: 64KB).
    Returns:
        String hexadecimal do hash SHA256.
    Raises:
        FileNotFoundError: Se o arquivo nao existir.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_artifact(path: str, expected_hash: str) -> bool:
    """
    Verifica integridade de artefato comparando hash SHA256.

    Args:
        path: Caminho do arquivo a verificar.
        expected_hash: Hash SHA256 esperado (do manifesto).
    Returns:
        True se o hash bater; False se houver discrepancia.
    """
    if not os.path.exists(path):
        return False
    actual_hash = sha256_file(path)
    return actual_hash == expected_hash


def verify_artifact_strict(path: str, expected_hash: str, model_name: str = "") -> None:
    """
    Verifica integridade levantando excecao se hash nao bater.

    Args:
        path: Caminho do arquivo.
        expected_hash: Hash esperado do manifesto.
        model_name: Nome do modelo (para mensagem de erro).
    Raises:
        FileNotFoundError: Se arquivo nao existir.
        ValueError: Se hash nao bater.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arquivo de modelo nao encontrado: {path}\n"
            f"Modelo: {model_name}"
        )
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(
            f"FALHA DE INTEGRIDADE: {model_name}\n"
            f"   Arquivo: {path}\n"
            f"   Hash esperado:  {expected_hash}\n"
            f"   Hash calculado: {actual}\n"
            "O arquivo pode ter sido corrompido ou substituido."
        )
