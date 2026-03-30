import pandas as pd
import json
import pytest

from src.outputs.report_generator import (
    _build_disclaimer,
    _build_header,
    _build_kpis,
    _build_methodology,
    _resolve_operational_percentile,
    generate_report,
)


pytestmark = [pytest.mark.hermetic]


def test_build_header_handles_na_without_format_error():
    profile = {
        "periodo": "N/A",
        "total_veiculos": "N/A",
        "dias_analise": "N/A",
        "split_temporal": {
            "train_size": "N/A",
            "val_size": "N/A",
            "test_size": "N/A",
        },
    }

    html = _build_header("run_test", profile)
    assert "Run ID: run_test" in html
    assert "Treino: N/A | Validacao: N/A | Teste: N/A" in html


def test_disclaimer_and_kpi_reflect_operational_percentile():
    disclaimer_html = _build_disclaimer(operational_percentile=90)
    assert "p90 = 10%" in disclaimer_html

    df = pd.DataFrame({"ensemble_alert": [1.0, 0.0], "iso_alert": [1.0, 0.0], "hbos_alert": [1.0, 0.0], "temp_alert": [1.0, 0.0]})
    kpi_html = _build_kpis(df, pd.DataFrame(), operational_percentile=90)
    assert "Taxa de Alerta (p90)" in kpi_html


def test_methodology_reflects_operational_percentile_without_p95_text():
    html = _build_methodology(operational_percentile=90)
    assert "percentil 90" in html
    assert "p95" not in html.lower()


def test_report_percentile_resolution_prioritizes_profile_value_over_detected_labels():
    df = pd.DataFrame(
        {
            "ISO_n100_p95_label": [0.0, 1.0],
            "ISO_n100_p90_label": [1.0, 0.0],
        }
    )
    resolved = _resolve_operational_percentile(
        {"operational_percentile": 90},
        df=df,
        fallback=95,
    )
    assert resolved == 90


def test_generate_report_html_is_self_contained_and_has_methodological_guards(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "periodo": "2024-01-01 a 2024-01-02",
        "total_veiculos": 2,
        "dias_analise": 2,
        "split_temporal": {"train_size": 2, "val_size": 1, "test_size": 1},
        "operational_percentile": 95,
    }
    (metrics_dir / "perfil_dados.json").write_text(
        json.dumps(profile),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "placa": ["ABC1234", "DEF5678"],
            "total_registros": [2, 2],
            "registros_avaliados": [2, 2],
            "alertas_ensemble": [1, 0],
            "score_medio": [0.7, 0.2],
            "score_maximo": [0.9, 0.3],
            "taxa_alerta": [0.5, 0.0],
            "ranking_risco": [1, 2],
        }
    ).to_csv(metrics_dir / "vehicle_risk_ranking.csv", index=False)

    pd.DataFrame(
        {
            "config": ["ISO_n100"],
            "train_anomaly_rate": [5.0],
            "val_anomaly_rate": [5.2],
            "stability_delta_pct": [0.2],
            "rank_stability": [1],
        }
    ).to_csv(metrics_dir / "model_selection_val.csv", index=False)

    pd.DataFrame(
        {
            "placa": ["ABC1234", "DEF5678"],
            "max_modelos_temporais": [1, 0],
        }
    ).to_csv(metrics_dir / "vehicle_coverage_report.csv", index=False)

    pd.DataFrame(
        {
            "placa": ["ABC1234", "ABC1234", "DEF5678", "DEF5678"],
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
            "ensemble_alert": [1.0, 0.0, 0.0, 0.0],
            "ensemble_vote_pct": [0.8, 0.2, 0.1, 0.2],
            "iso_alert": [1.0, 0.0, 0.0, 0.0],
            "hbos_alert": [1.0, 0.0, 0.0, 0.0],
            "temp_alert": [1.0, 0.0, 0.0, 0.0],
            "coverage_temporal": [1, 1, 0, 0],
            "ISO_n100_score": [0.9, 0.2, 0.1, 0.3],
        }
    ).to_parquet(tmp_path / "resultado_final.parquet", index=False)

    output_path = tmp_path / "relatorio_executivo.html"
    generated = generate_report(
        metrics_dir=str(metrics_dir),
        parquet_path=str(tmp_path / "resultado_final.parquet"),
        output_path=str(output_path),
        run_id="20260329_000000",
    )

    assert generated == str(output_path)
    html = output_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "Aviso Metodologico - Leitura Obrigatoria" in html
    assert "Limitacao Critica: Ausencia de Ground Truth" in html
    assert "Indicadores Executivos" in html
    assert "Taxa de Alerta (p95)" in html
    assert 'src="https://cdn.plot.ly' not in html
    assert 'src="http://cdn.plot.ly' not in html
