"""
Schema de validacao para dados de entrada do pipeline SSP-DF.
Usa Pandera para validacao declarativa de DataFrames.
"""

import numpy as np
import pandera as pa
from pandera import Check, Column


# =========================================================================
# SCHEMA POS-PADRONIZACAO (apos rename de colunas no load_and_standardize)
# =========================================================================
RAW_INPUT_SCHEMA = pa.DataFrameSchema(
    columns={
        "placa": Column(
            str,
            nullable=False,
            checks=[
                Check.str_length(
                    min_value=7, max_value=8, error="Placa deve ter 7-8 caracteres"
                ),
            ],
            description="Placa do veiculo (formato Mercosul ou antigo)",
        ),
        "timestamp": Column(
            "datetime64[ns]",
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(
                    np.datetime64("2020-01-01"), error="Timestamp anterior a 2020"
                ),
                Check.less_than_or_equal_to(
                    np.datetime64("2030-12-31"), error="Timestamp posterior a 2030"
                ),
            ],
            description="Data/hora do registro",
        ),
        "latitude": Column(
            float,
            nullable=True,
            checks=[
                Check.in_range(
                    -16.5, -15.0, error="Latitude fora do DF (-16.5 a -15.0)"
                ),
            ],
            description="Latitude (DF: ~-15.5 a -16.0)",
        ),
        "longitude": Column(
            float,
            nullable=True,
            checks=[
                Check.in_range(
                    -48.5, -47.0, error="Longitude fora do DF (-48.5 a -47.0)"
                ),
            ],
            description="Longitude (DF: ~-47.5 a -48.0)",
        ),
    },
    strict=False,
    checks=[
        # Mínimo de 1 registro — sem restrição de tamanho de amostra.
        # Inferência deve aceitar micro-batches (até 1 registro).
        # Treinamento com amostras pequenas é válido em desenvolvimento.
        pa.Check(lambda df: len(df) >= 1, error="Dataset deve ter ao menos 1 registro"),
    ],
    description="Schema de entrada do pipeline SSP-DF apos padronizacao de colunas",
)


def validate_input(df, schema=None):
    """
    Valida DataFrame de entrada contra schema definido.

    Args:
        df: DataFrame a validar.
        schema: Schema Pandera. Se None, usa RAW_INPUT_SCHEMA.
    Returns:
        DataFrame validado.
    Raises:
        pandera.errors.SchemaError: se validacao falhar.
    """
    if schema is None:
        schema = RAW_INPUT_SCHEMA
    return schema.validate(df, lazy=True)
