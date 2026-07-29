import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.logging_config import (
    JSONFormatter,
    LoggerContext,
    get_logger,
    log_api_call,
    setup_logging,
    execution_id_var,
    dataset_id_var,
    api_source_var,
)


def _reset_globals():
    import src.utils.logging_config as lc
    lc._is_initialized = False
    lc._handlers = []


@pytest.fixture(autouse=True)
def reset_logging():
    _reset_globals()
    yield
    _reset_globals()


def test_get_logger_returns_logger(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.logging_config._LOG_DIR", tmp_path)
    monkeypatch.setattr("src.utils.logging_config._APPLICATION_LOG", tmp_path / "application.log")
    monkeypatch.setattr("src.utils.logging_config._API_ERRORS_LOG", tmp_path / "api_errors.log")
    monkeypatch.setattr("src.utils.logging_config._PIPELINE_EXECUTION_LOG", tmp_path / "pipeline_execution.log")
    logger = get_logger("test_get_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_get_logger"


def test_logger_context_sets_vars():
    assert execution_id_var.get() is None
    assert dataset_id_var.get() is None
    assert api_source_var.get() is None

    with LoggerContext(execution_id="exec123", dataset_id="ds456", api_source="openweather"):
        assert execution_id_var.get() == "exec123"
        assert dataset_id_var.get() == "ds456"
        assert api_source_var.get() == "openweather"

    assert execution_id_var.get() is None
    assert dataset_id_var.get() is None
    assert api_source_var.get() is None


def test_log_api_call_logs_fields(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_api_log")
    logger.handlers = []
    logger.propagate = True

    log_api_call(logger, "GET", "http://example.com/data", 200, 150.5, records_count=10, bytes_count=2048, retries=0)

    assert len(caplog.records) >= 1
    record = caplog.records[0]
    assert record.levelname == "INFO"
    assert hasattr(record, "api_method") and record.api_method == "GET"
    assert hasattr(record, "api_url") and record.api_url == "http://example.com/data"
    assert hasattr(record, "http_status") and record.http_status == 200
    assert hasattr(record, "duration_ms") and record.duration_ms == 150.5
    assert hasattr(record, "records_count") and record.records_count == 10


def test_setup_logging_creates_handlers(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.logging_config._LOG_DIR", tmp_path)
    monkeypatch.setattr("src.utils.logging_config._APPLICATION_LOG", tmp_path / "application.log")
    monkeypatch.setattr("src.utils.logging_config._API_ERRORS_LOG", tmp_path / "api_errors.log")
    monkeypatch.setattr("src.utils.logging_config._PIPELINE_EXECUTION_LOG", tmp_path / "pipeline_execution.log")
    setup_logging(log_dir=tmp_path)

    root = logging.getLogger()
    handler_types = [type(h).__name__ for h in root.handlers]
    assert "RotatingFileHandler" in handler_types
    assert "StreamHandler" in handler_types


def test_json_formatter_output():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="hello json",
        args=(),
        exc_info=None,
    )
    out = formatter.format(record)
    parsed = json.loads(out)
    assert parsed["message"] == "hello json"
    assert parsed["level"] == "INFO"
    assert parsed["module"] == "test_logging_config"
    assert "timestamp" in parsed
    assert parsed["execution_id"] is None
    assert parsed["dataset_id"] is None


def test_logging_dir_created(tmp_path, monkeypatch):
    log_dir = tmp_path / "my_logs"
    monkeypatch.setattr("src.utils.logging_config._LOG_DIR", log_dir)
    monkeypatch.setattr("src.utils.logging_config._APPLICATION_LOG", log_dir / "application.log")
    monkeypatch.setattr("src.utils.logging_config._API_ERRORS_LOG", log_dir / "api_errors.log")
    monkeypatch.setattr("src.utils.logging_config._PIPELINE_EXECUTION_LOG", log_dir / "pipeline_execution.log")
    setup_logging(log_dir=log_dir)
    assert log_dir.exists()
    assert log_dir.is_dir()
