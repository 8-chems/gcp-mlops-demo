"""Structured JSON logging. Cloud Run / Cloud Run Jobs / Composer forward stdout to
Cloud Logging, which parses `severity` and `message` and exposes the rest as jsonPayload."""
import json
import logging
import sys
from datetime import datetime, timezone

_LOGGER_NAME = "mlops"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(event: str, severity: str = "INFO", **fields) -> None:
    payload = {
        "severity": severity.upper(),
        "message": event,
        "event": event,
        "time": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    level = getattr(logging, severity.upper(), logging.INFO)
    get_logger().log(level, json.dumps(payload, default=str))
