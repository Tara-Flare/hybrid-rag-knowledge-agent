"""Agent 请求链路追踪与 SQLite 持久化。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterator
from uuid import uuid4


@dataclass
class TracePhase:
    """单个执行阶段的记录。"""

    name: str
    duration_ms: float
    success: bool
    error_type: str | None = None
    outcome: str | None = None
    token_usage: dict[str, int] | None = None


@dataclass
class RequestTrace:
    """一次 Agent 请求的追踪信息。"""

    session_id: str | None
    knowledge_base_id: str

    request_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    status: str = "running"
    duration_ms: float = 0.0
    error_type: str | None = None

    phases: list[TracePhase] = field(
        default_factory=list
    )

    token_usage: dict[str, int] | None = None

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    _started: float = field(
        default_factory=perf_counter,
        repr=False,
    )

    @contextmanager
    def phase(self, name: str) -> Iterator[TracePhase]:
        """记录阶段耗时、异常及业务结果。"""

        start = perf_counter()

        record = TracePhase(
            name=name,
            duration_ms=0.0,
            success=True,
        )

        try:
            yield record

        except Exception as exc:
            record.success = False
            record.error_type = type(exc).__name__
            raise

        finally:
            record.duration_ms = round(
                (perf_counter() - start) * 1000,
                2,
            )

            self.phases.append(record)

    def finish(
        self,
        error: Exception | None = None,
    ) -> None:
        """结束追踪并记录最终状态。"""

        if self.status != "running":
            raise RuntimeError("RequestTrace 已经结束")

        self.duration_ms = round(
            (perf_counter() - self._started) * 1000,
            2,
        )

        if error is None:
            self.status = "success"
        else:
            self.status = "error"
            self.error_type = type(error).__name__


class RequestTraceStore:
    """将请求追踪记录保存到 SQLite。"""

    def __init__(
        self,
        database_path: str = "data/agent.db",
    ) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_traces (
                    request_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    knowledge_base_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    phases_json TEXT NOT NULL,
                    token_usage_json TEXT,
                    error_type TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_request_traces_session
                ON request_traces(session_id, created_at)
                """
            )

    def save(self, trace: RequestTrace) -> None:
        """保存已经结束的请求追踪。"""

        if trace.status not in {"success", "error"}:
            raise ValueError(
                "只能保存已经结束的 RequestTrace"
            )

        phases_json = json.dumps(
            [asdict(phase) for phase in trace.phases],
            ensure_ascii=False,
        )

        token_usage_json = (
            json.dumps(trace.token_usage)
            if trace.token_usage is not None
            else None
        )

        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO request_traces (
                    request_id,
                    session_id,
                    knowledge_base_id,
                    status,
                    duration_ms,
                    phases_json,
                    token_usage_json,
                    error_type,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.request_id,
                    trace.session_id,
                    trace.knowledge_base_id,
                    trace.status,
                    trace.duration_ms,
                    phases_json,
                    token_usage_json,
                    trace.error_type,
                    trace.created_at,
                ),
            )

    def get(self, request_id: str) -> dict | None:
        """根据 request_id 查询追踪记录。"""

        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                """
                SELECT *
                FROM request_traces
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)

        result["phases"] = json.loads(
            result.pop("phases_json")
        )

        raw_usage = result.pop("token_usage_json")

        result["token_usage"] = (
            json.loads(raw_usage)
            if raw_usage is not None
            else None
        )

        return result
