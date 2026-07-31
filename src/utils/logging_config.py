import json
import logging
import logging.handlers
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

execution_id_var: ContextVar[str | None] = ContextVar("execution_id", default=None)
dataset_id_var: ContextVar[str | None] = ContextVar("dataset_id", default=None)
api_source_var: ContextVar[str | None] = ContextVar("api_source", default=None)

_LOG_DIR = Path("logs")
_APPLICATION_LOG = _LOG_DIR / "application.log"
_API_ERRORS_LOG = _LOG_DIR / "api_errors.log"
_PIPELINE_EXECUTION_LOG = _LOG_DIR / "pipeline_execution.log"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
            "execution_id": execution_id_var.get(),
            "dataset_id": dataset_id_var.get(),
            "api_source": api_source_var.get(),
            "error_category": getattr(record, "error_category", None),
            "duration_ms": getattr(record, "duration_ms", None),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "value": str(record.exc_info[1]),
                "traceback": "".join(traceback.format_exception(*record.exc_info)).splitlines()
                if record.exc_info
                else None,
            }
        extra_keys = [k for k in record.__dict__ if k not in log_entry and not k.startswith("_")]
        for k in extra_keys:
            try:
                json.dumps(record.__dict__[k])
                log_entry[k] = record.__dict__[k]
            except (TypeError, ValueError):
                log_entry[k] = str(record.__dict__[k])
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class APIErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        source = api_source_var.get()
        return bool(source)


class PipelineExecutionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(execution_id_var.get())


_handlers: list[logging.Handler] = []
_is_initialized: bool = False


def _create_rotating_handler(
    path: Path,
    level: int,
    formatter: logging.Formatter,
    filters: list[logging.Filter] | None = None,
) -> logging.handlers.RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        str(path), maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    if filters:
        for f in filters:
            handler.addFilter(f)
    return handler


def setup_logging(
    log_dir: str | Path | None = None,
    console_level: int = logging.INFO,
    application_level: int = logging.DEBUG,
    api_errors_level: int = logging.WARNING,
    pipeline_level: int = logging.INFO,
) -> None:
    global _is_initialized, _handlers
    if _is_initialized:
        return
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    log_path = Path(log_dir) if log_dir else _LOG_DIR
    json_formatter = JSONFormatter()
    app_handler = _create_rotating_handler(
        path=log_path / "application.log",
        level=application_level,
        formatter=json_formatter,
    )
    api_handler = _create_rotating_handler(
        path=log_path / "api_errors.log",
        level=api_errors_level,
        formatter=json_formatter,
        filters=[APIErrorFilter()],
    )
    pipeline_handler = _create_rotating_handler(
        path=log_path / "pipeline_execution.log",
        level=pipeline_level,
        formatter=json_formatter,
        filters=[PipelineExecutionFilter()],
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s"
    console_handler.setFormatter(logging.Formatter(console_format))
    _handlers = [app_handler, api_handler, pipeline_handler, console_handler]
    root.setLevel(logging.DEBUG)
    for h in _handlers:
        root.addHandler(h)
    _is_initialized = True


def _ensure_initialized() -> None:
    if not _is_initialized:
        setup_logging()


def get_logger(name: str) -> logging.Logger:
    _ensure_initialized()
    return logging.getLogger(name)


class LoggerContext:
    def __init__(
        self,
        execution_id: str | None = None,
        dataset_id: str | None = None,
        api_source: str | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.dataset_id = dataset_id
        self.api_source = api_source
        self._execution_id_token: Any = None
        self._dataset_id_token: Any = None
        self._api_source_token: Any = None

    def __enter__(self) -> "LoggerContext":
        self._execution_id_token = (
            execution_id_var.set(self.execution_id) if self.execution_id else None
        )
        self._dataset_id_token = dataset_id_var.set(self.dataset_id) if self.dataset_id else None
        self._api_source_token = api_source_var.set(self.api_source) if self.api_source else None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._execution_id_token is not None:
            execution_id_var.reset(self._execution_id_token)
        if self._dataset_id_token is not None:
            dataset_id_var.reset(self._dataset_id_token)
        if self._api_source_token is not None:
            api_source_var.reset(self._api_source_token)


def log_api_call(
    logger: logging.Logger,
    method: str,
    url: str,
    status: int,
    duration_ms: float,
    records_count: int = 0,
    bytes_count: int = 0,
    retries: int = 0,
) -> None:
    extra = {
        "api_method": method,
        "api_url": url,
        "http_status": status,
        "duration_ms": duration_ms,
        "records_count": records_count,
        "bytes_count": bytes_count,
        "retries": retries,
        "error_category": "api_error" if status >= 400 else None,
    }
    if status >= 400:
        logger.warning(
            "API call %s %s returned %d in %.1fms (retries=%d)",
            method,
            url,
            status,
            duration_ms,
            retries,
            extra=extra,
        )
    else:
        logger.info(
            "API call %s %s returned %d in %.1fms",
            method,
            url,
            status,
            duration_ms,
            extra=extra,
        )
