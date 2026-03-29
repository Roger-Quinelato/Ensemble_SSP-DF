import logging
import os

import pytest

from src.utils import logger_utils


def test_default_log_file_uses_run_id(monkeypatch):
    monkeypatch.setenv("SSPDF_LOG_DIR", "tmp_logs")
    path = logger_utils._default_log_file(run_id="20260329_101010")
    expected_suffix = os.path.join("tmp_logs", "execution_20260329_101010.log")
    assert path.endswith(expected_suffix)


def test_setup_logger_writes_log_file(tmp_path):
    log_file = tmp_path / "exec.log"
    logger = logger_utils.setup_logger(
        name="sspdf_logger_test_file",
        log_file=str(log_file),
        level=logging.INFO,
        run_id="run_a",
    )
    logger.info("mensagem-teste")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "mensagem-teste" in content


def test_log_execution_logs_start_and_end(caplog):
    caplog.set_level(logging.INFO, logger="sspdf")

    @logger_utils.log_execution
    def _ok():
        return 123

    assert _ok() == 123
    messages = [rec.message for rec in caplog.records]
    assert any("[START] _ok" in msg for msg in messages)
    assert any("[END] _ok" in msg for msg in messages)


def test_log_execution_logs_error(caplog):
    caplog.set_level(logging.ERROR, logger="sspdf")

    @logger_utils.log_execution
    def _boom():
        raise RuntimeError("falha_unitaria")

    with pytest.raises(RuntimeError, match="falha_unitaria"):
        _boom()
    messages = [rec.message for rec in caplog.records]
    assert any("[ERROR] Falha em _boom: falha_unitaria" in msg for msg in messages)
