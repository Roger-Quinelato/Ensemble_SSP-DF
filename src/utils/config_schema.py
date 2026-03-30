"""
Validacao formal do YAML de configuracao do pipeline SSP-DF.
"""

from __future__ import annotations

from copy import deepcopy
from src.utils.model_selection import normalize_temporal_strategy


_TOP_LEVEL_ALLOWED_KEYS = {
    "random_state",
    "mapeamento_colunas",
    "parametros",
    "configuracoes_gerais",
}
_MAP_REQUIRED_KEYS = ("placa", "timestamp", "latitude", "longitude")
_MAP_ALLOWED_KEYS = set(_MAP_REQUIRED_KEYS) | {"RA"}
_SPLIT_DEFAULTS = {"train": 0.6, "validation": 0.2, "test": 0.2}
_PARAM_BLOCKS_EXPECTED_DICT = ("split_ratios", "isolation_forest", "hbos", "temporal")


def _ensure(condition, message):
    if not condition:
        raise ValueError(message)


def _coerce_int(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} invalido: bool nao e permitido.")
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} invalido: esperado inteiro.") from exc


def _coerce_float(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} invalido: bool nao e permitido.")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} invalido: esperado numero.") from exc


def validate_pipeline_config(config, context="pipeline"):
    """
    Valida e normaliza configuracao YAML.

    Args:
        config: Objeto retornado por yaml.safe_load().
        context: rotulo usado na mensagem de erro (ex.: "treino" ou "inferencia").
    Returns:
        dict: configuracao validada/normalizada.
    Raises:
        ValueError: quando a configuracao e invalida.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuracao YAML invalida para {context}: esperado objeto/dict na raiz."
        )

    cfg = deepcopy(config)

    unknown_top = sorted(set(cfg.keys()) - _TOP_LEVEL_ALLOWED_KEYS)
    _ensure(
        not unknown_top,
        f"Configuracao YAML invalida para {context}: chaves de topo desconhecidas: {unknown_top}.",
    )

    if "random_state" in cfg:
        cfg["random_state"] = _coerce_int(cfg["random_state"], "random_state")

    map_cols = cfg.get("mapeamento_colunas")
    _ensure(
        isinstance(map_cols, dict),
        f"Configuracao YAML invalida para {context}: 'mapeamento_colunas' deve ser objeto.",
    )
    missing_map = [k for k in _MAP_REQUIRED_KEYS if k not in map_cols]
    _ensure(
        not missing_map,
        f"Configuracao YAML invalida para {context}: faltam chaves em mapeamento_colunas: {missing_map}.",
    )
    unknown_map = sorted(set(map_cols.keys()) - _MAP_ALLOWED_KEYS)
    _ensure(
        not unknown_map,
        f"Configuracao YAML invalida para {context}: chaves desconhecidas em mapeamento_colunas: {unknown_map}.",
    )
    for key in _MAP_REQUIRED_KEYS:
        value = map_cols.get(key)
        _ensure(
            isinstance(value, str) and value.strip(),
            f"Configuracao YAML invalida para {context}: mapeamento_colunas.{key} deve ser string nao vazia.",
        )
    if "RA" in map_cols:
        _ensure(
            isinstance(map_cols["RA"], str) and map_cols["RA"].strip(),
            f"Configuracao YAML invalida para {context}: mapeamento_colunas.RA deve ser string nao vazia.",
        )

    params = cfg.get("parametros")
    _ensure(
        isinstance(params, dict),
        f"Configuracao YAML invalida para {context}: 'parametros' deve ser objeto.",
    )
    for block_name in _PARAM_BLOCKS_EXPECTED_DICT:
        if block_name in params and not isinstance(params[block_name], dict):
            raise ValueError(
                f"Configuracao YAML invalida para {context}: parametros.{block_name} deve ser objeto."
            )
    temporal_cfg = params.setdefault("temporal", {})
    _ensure(
        isinstance(temporal_cfg, dict),
        f"Configuracao YAML invalida para {context}: parametros.temporal deve ser objeto.",
    )
    temporal_cfg["temporal_strategy"] = normalize_temporal_strategy(
        temporal_cfg.get("temporal_strategy", "all")
    )

    split = params.get("split_ratios")
    if split is None:
        split = dict(_SPLIT_DEFAULTS)
    else:
        _ensure(
            isinstance(split, dict),
            f"Configuracao YAML invalida para {context}: parametros.split_ratios deve ser objeto.",
        )
        split = dict(split)

    if "val" in split and "validation" not in split:
        split["validation"] = split["val"]
    if "val" in split and "validation" in split and split["val"] != split["validation"]:
        raise ValueError(
            f"Configuracao YAML invalida para {context}: split_ratios.val difere de split_ratios.validation."
        )

    allowed_split = {"train", "validation", "test", "val"}
    unknown_split = sorted(set(split.keys()) - allowed_split)
    _ensure(
        not unknown_split,
        f"Configuracao YAML invalida para {context}: chaves desconhecidas em split_ratios: {unknown_split}.",
    )

    normalized_split = {}
    for key in ("train", "validation", "test"):
        raw_value = split.get(key, _SPLIT_DEFAULTS[key])
        value = _coerce_float(raw_value, f"parametros.split_ratios.{key}")
        _ensure(
            0.0 <= value <= 1.0,
            f"Configuracao YAML invalida para {context}: parametros.split_ratios.{key} fora de [0, 1].",
        )
        normalized_split[key] = value

    split_sum = (
        normalized_split["train"]
        + normalized_split["validation"]
        + normalized_split["test"]
    )
    _ensure(
        split_sum <= 1.0 + 1e-9,
        f"Configuracao YAML invalida para {context}: soma de split_ratios > 1 ({split_sum:.4f}).",
    )
    params["split_ratios"] = normalized_split

    raw_percentiles = params.get("percentis_teste")
    if raw_percentiles is None:
        raw_percentiles = [95]
    _ensure(
        isinstance(raw_percentiles, list) and len(raw_percentiles) > 0,
        f"Configuracao YAML invalida para {context}: parametros.percentis_teste deve ser lista nao vazia.",
    )
    normalized_percentiles = []
    for idx, value in enumerate(raw_percentiles):
        p = _coerce_int(value, f"parametros.percentis_teste[{idx}]")
        _ensure(
            0 <= p <= 100,
            f"Configuracao YAML invalida para {context}: percentile fora de [0, 100] em parametros.percentis_teste[{idx}]={p}.",
        )
        normalized_percentiles.append(p)
    params["percentis_teste"] = normalized_percentiles

    general_cfg = cfg.get("configuracoes_gerais")
    if general_cfg is not None and not isinstance(general_cfg, dict):
        raise ValueError(
            f"Configuracao YAML invalida para {context}: configuracoes_gerais deve ser objeto."
        )

    return cfg


def validate_training_config(config):
    """Alias semantico para validacao de treino."""
    return validate_pipeline_config(config, context="treino")


def validate_inference_config(config):
    """Alias semantico para validacao de inferencia."""
    return validate_pipeline_config(config, context="inferencia")
