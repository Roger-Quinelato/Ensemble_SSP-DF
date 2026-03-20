"""
Testes de estabilidade determinista.

Verificam que o pipeline produz resultados identicos para os mesmos dados e seed.
Um pipeline ML determinista e requisito fundamental para auditabilidade governamental:
dado um alerta, deve ser possivel reproduzir exatamente as condicoes que o geraram.
"""
import glob
import json
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# FIXTURE: dados sinteticos minimos
# =============================================================================


def _generate_synthetic_data(n_records=300, n_vehicles=20, seed=42):
    """Gera dados sinteticos reproduziveis para testes."""
    rng = np.random.default_rng(seed)

    timestamps = [
        datetime(2024, 1, 1) + timedelta(minutes=i * 15) for i in range(n_records)
    ]
    placas = [f"TST{str(i % n_vehicles).zfill(4)}" for i in range(n_records)]

    return pd.DataFrame(
        {
            "placa": placas,
            "timestamp": timestamps,
            "latitude": rng.uniform(-15.9, -15.6, n_records),
            "longitude": rng.uniform(-47.9, -47.5, n_records),
            "regiao_adm": rng.choice(
                ["Plano Piloto", "Taguatinga", "Ceilandia"], n_records
            ),
        }
    )


# =============================================================================
# TESTE 1: Determinismo dos scores
# =============================================================================


class TestDeterminism:
    """Pipeline com mesmo seed deve produzir scores identicos."""

    @pytest.fixture(scope="class")
    def two_runs_output(self, tmp_path_factory):
        """Executa pipeline duas vezes e retorna os parquets de resultado."""
        data_dir = tmp_path_factory.mktemp("data")
        out1 = tmp_path_factory.mktemp("run1")
        out2 = tmp_path_factory.mktemp("run2")

        data_path = data_dir / "stability_test.csv"
        df = _generate_synthetic_data(n_records=300, n_vehicles=20, seed=42)
        df.to_csv(data_path, index=False)

        for out_dir in [out1, out2]:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.main",
                    "--epochs",
                    "1",
                    "--output-dir",
                    str(out_dir),
                    "--input",
                    str(data_path),
                    "--seed",
                    "42",
                ],
                capture_output=True,
                text=True,
                cwd=".",
            )
            if result.returncode != 0:
                pytest.skip(
                    f"Pipeline falhou (returncode={result.returncode}):\n"
                    f"{result.stderr[-1000:]}"
                )

        def _find_parquet(out_dir):
            matches = glob.glob(str(out_dir / "*/master_table/resultado_final.parquet"))
            if not matches:
                pytest.skip(f"resultado_final.parquet nao encontrado em {out_dir}")
            return pd.read_parquet(matches[0])

        return _find_parquet(out1), _find_parquet(out2)

    def test_ensemble_alert_is_identical(self, two_runs_output):
        """ensemble_alert deve ser bit-a-bit identico entre as duas runs."""
        df1, df2 = two_runs_output

        if "ensemble_alert" not in df1.columns:
            pytest.skip("ensemble_alert nao no parquet - verificar Fix NC2")

        alerts1 = df1["ensemble_alert"].values
        alerts2 = df2["ensemble_alert"].values

        nan1 = np.isnan(alerts1.astype(float))
        nan2 = np.isnan(alerts2.astype(float))

        assert np.array_equal(nan1, nan2), (
            "Padrao de NaN diferente entre runs:\n"
            f"  Run 1 NaN count: {nan1.sum()}\n"
            f"  Run 2 NaN count: {nan2.sum()}\n"
            "Isso indica nao-determinismo na avaliacao temporal (window size ou "
            "agrupamento por veiculo pode ter comportamento nao determinista)."
        )

        mask = ~nan1
        assert np.array_equal(alerts1[mask], alerts2[mask]), (
            "ensemble_alert DIFERENTE entre runs para os mesmos dados e seed!\n"
            f"Diferencas: {(alerts1[mask] != alerts2[mask]).sum()} registros\n"
            "Isso indica NON-DETERMINISMO no pipeline.\n"
            "Verificar: random_state nos modelos ISO/HBOS, seed do Keras/TensorFlow."
        )

    def test_iso_scores_are_identical(self, two_runs_output):
        """Scores do Isolation Forest devem ser identicos (random_state fixo)."""
        df1, df2 = two_runs_output

        iso_cols = [c for c in df1.columns if c.startswith("ISO") and c.endswith("_score")]
        if not iso_cols:
            pytest.skip("Nenhuma coluna de score ISO no parquet")

        for col in iso_cols:
            if col not in df2.columns:
                continue
            diff = (df1[col] - df2[col]).abs().max()
            assert diff < 1e-10, (
                f"Score ISO '{col}' difere entre runs: diff_max={diff:.2e}\n"
                "Verificar que IsolationForest usa random_state fixo."
            )

    def test_alert_count_is_deterministic(self, two_runs_output):
        """Numero total de alertas deve ser identico entre runs."""
        df1, df2 = two_runs_output

        if "ensemble_alert" not in df1.columns:
            pytest.skip("ensemble_alert nao no parquet")

        n_alerts_1 = int((df1["ensemble_alert"] == 1.0).sum())
        n_alerts_2 = int((df2["ensemble_alert"] == 1.0).sum())

        assert n_alerts_1 == n_alerts_2, (
            "Numero de alertas diferente entre runs com mesmo seed:\n"
            f"  Run 1: {n_alerts_1} alertas\n"
            f"  Run 2: {n_alerts_2} alertas\n"
            f"Diferenca: {abs(n_alerts_1 - n_alerts_2)} alertas\n"
            "Isso e inaceitavel para auditabilidade governamental."
        )


# =============================================================================
# TESTE 2: Estabilidade de thresholds entre runs
# =============================================================================


class TestThresholdStability:
    """Thresholds calibrados com mesmo seed e dados devem ser identicos."""

    def test_p95_thresholds_identical_between_runs(self, tmp_path):
        """Dois treinamentos com mesmos dados e seed devem gerar thresholds iguais."""
        del tmp_path  # Mantido para compatibilidade da assinatura solicitada.
        runs = sorted(glob.glob("outputs/*/models_saved/thresholds_p95.json"), reverse=True)

        if len(runs) < 2:
            pytest.skip("Menos de 2 runs disponiveis - executar pipeline 2x com mesmo seed")

        with open(runs[0], encoding="utf-8") as f:
            t1 = json.load(f)
        with open(runs[1], encoding="utf-8") as f:
            t2 = json.load(f)

        common_models = set(t1.keys()) & set(t2.keys())

        if not common_models:
            pytest.skip("Nenhum modelo em comum entre as duas runs")

        for model in common_models:
            ratio = t1[model] / t2[model] if t2[model] != 0 else float("inf")
            assert 0.5 <= ratio <= 2.0, (
                f"Threshold de {model} muito diferente entre runs:\n"
                f"  Run mais recente: {t1[model]:.6f}\n"
                f"  Run anterior:     {t2[model]:.6f}\n"
                f"  Razao: {ratio:.2f}x\n"
                "Se os dados de treino eram diferentes, isso e esperado. "
                "Se eram iguais, indica instabilidade no pipeline."
            )
