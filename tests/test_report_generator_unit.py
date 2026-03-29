import pandas as pd

from src.outputs.report_generator import (
    _build_disclaimer,
    _build_header,
    _build_kpis,
    _build_methodology,
    _resolve_operational_percentile,
)


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
