def test_report_is_generated(tmp_path):
    """Smoke test: relatorio HTML deve ser gerado apos run do pipeline."""
    import glob
    import os

    import pytest

    _ = tmp_path
    reports = glob.glob("outputs/*/relatorio_executivo.html")
    if not reports:
        pytest.skip("Nenhum relatorio encontrado - execute run_experiment() primeiro")

    latest = max(reports, key=os.path.getmtime)
    size = os.path.getsize(latest)

    assert size > 10_000, (
        f"Relatorio em {latest} tem apenas {size} bytes - parece incompleto"
    )

    with open(latest, encoding="utf-8") as f:
        content = f.read()

    assert "<!DOCTYPE html>" in content
    assert "Relatorio de Anomalias" in content or "Relatório de Anomalias" in content
    assert "ensemble_alert" in content or "Alertas" in content
    assert "Aviso Metodologico" in content or "Aviso Metodológico" in content
    assert "Limitation" in content or "Limitacao" in content or "Limitação" in content
