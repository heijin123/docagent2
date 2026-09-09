"""结构化日志：file + console，级别可配；M5 起支持 JSON 行格式（默认）。

格式（.env LOG_FORMAT）：
- json：单行 JSON（ts/level/logger/message），便于 Filebeat/Loki/Promtail 采集；
- text：人类可读回退（本地 debug）。
console 固定 text（便于人看）；file 跟随 LOG_FORMAT（默认 json）。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(name: str = "agent", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # 已初始化（避免重复 handler）
        return logger

    settings.ensure_dirs()
    logger.setLevel(level)

    text_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        Path(settings.log_dir) / "agent.log", encoding="utf-8"
    )
    file_handler.setFormatter(
        JsonFormatter() if settings.log_format == "json" else text_fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(text_fmt)
    logger.addHandler(console_handler)
    return logger
