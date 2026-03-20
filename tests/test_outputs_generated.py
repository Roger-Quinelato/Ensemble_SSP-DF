"""Testes para validar artefatos gerados no ultimo run versionado."""

import os
import re
import unittest


def _latest_run_dir(base="outputs"):
    if not os.path.exists(base):
        return None
    run_pattern = re.compile(r"^\d{8}_\d{6}$")
    runs = sorted(
        [
            d
            for d in os.listdir(base)
            if run_pattern.match(d) and os.path.isdir(os.path.join(base, d))
        ],
        reverse=True,
    )
    if not runs:
        return None
    return os.path.join(base, runs[0])


@unittest.skipUnless(
    _latest_run_dir() is not None,
    "Nenhum run versionado encontrado em outputs/",
)
class TestOutputFiles(unittest.TestCase):
    def test_reports_generated(self):
        run_dir = _latest_run_dir()
        self.assertIsNotNone(run_dir)
        report_path = os.path.join(run_dir, "relatorio_executivo.html")
        self.assertTrue(os.path.exists(report_path), f"Relatorio ausente em {report_path}")

    def test_models_saved_generated(self):
        run_dir = _latest_run_dir()
        self.assertIsNotNone(run_dir)
        models_dir = os.path.join(run_dir, "models_saved")
        self.assertTrue(os.path.isdir(models_dir), f"Diretorio de modelos ausente: {models_dir}")
        files = os.listdir(models_dir)
        self.assertTrue(len(files) > 0, f"Nenhum arquivo gerado em {models_dir}")


if __name__ == "__main__":
    unittest.main()
