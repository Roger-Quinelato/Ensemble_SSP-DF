"""
Utilitarios de hardening para paths recebidos via CLI.
"""

import os
from pathlib import Path


def has_parent_ref(path_value):
    """
    Retorna True quando o path contem segmento '..' explicito.
    """
    if path_value is None:
        return False
    return ".." in Path(str(path_value)).parts


def normalize_cli_path(
    path_value,
    arg_name,
    *,
    must_exist=False,
    expect_dir=None,
    block_relative_parent=False,
):
    """
    Normaliza e valida path vindo de argumento CLI.

    Args:
        path_value: valor bruto recebido da CLI.
        arg_name: nome do argumento (para mensagens de erro).
        must_exist: exige existencia do path.
        expect_dir:
            - True: path deve ser diretorio (quando existir).
            - False: path deve ser arquivo (quando existir).
            - None: sem validacao de tipo.
        block_relative_parent:
            quando True, bloqueia uso de '..' em paths relativos.
    Returns:
        str: caminho absoluto normalizado.
    """
    if path_value is None:
        return None

    raw = str(path_value).strip()
    if not raw:
        raise ValueError(f"{arg_name}: caminho vazio nao e permitido.")
    if "\x00" in raw:
        raise ValueError(f"{arg_name}: caminho invalido (byte nulo).")

    if block_relative_parent and (not os.path.isabs(raw)) and has_parent_ref(raw):
        raise ValueError(
            f"{arg_name}: path traversal relativo com '..' nao e permitido: {path_value}"
        )

    normalized = os.path.abspath(os.path.normpath(os.path.expanduser(raw)))

    if must_exist and not os.path.exists(normalized):
        raise FileNotFoundError(f"{arg_name}: caminho nao encontrado: {normalized}")

    if expect_dir is True and os.path.exists(normalized) and not os.path.isdir(normalized):
        raise ValueError(f"{arg_name}: diretorio esperado, mas foi recebido arquivo: {normalized}")
    if expect_dir is False and os.path.exists(normalized) and not os.path.isfile(normalized):
        raise ValueError(f"{arg_name}: arquivo esperado, mas foi recebido diretorio: {normalized}")

    return normalized
