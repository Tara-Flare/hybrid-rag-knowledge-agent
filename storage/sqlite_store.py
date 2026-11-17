"""桌面智能助手的 SQLite 数据存储模块。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

class SQLiteStore:
    """负责创建数据库并保存工具调用日志。"""

    def __init__(
        self,
        database_path: str = "data/agent.db",
    ) -> None:
        """
        初始化 SQLite 数据库。

        Args:
            database_path: SQLite 数据库文件路径。
        """

        self.database_path = Path(database_path)

        # 确保 data 文件夹存在
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 初始化数据库表
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        """创建并返回数据库连接。"""

        connection = sqlite3.connect(
            self.database_path
        )

        # 查询结果可以通过字段名访问
        connection.row_factory = sqlite3.Row

        return connection

    def _create_tables(self) -> None:
        """创建项目需要的数据表。"""

        sql = """
        CREATE TABLE IF NOT EXISTS tool_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            arguments TEXT NOT NULL,
            result TEXT,
            success INTEGER NOT NULL,
            error_type TEXT,
            duration_ms REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """

        with self._connect() as connection:
            # 防止多个进程同时检查并迁移表结构。
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(sql)

            # Session 与知识库的持久化绑定。
            # 不修改已有 tool_logs 表。
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_bindings (
                    session_id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            # 向后兼容地扩展已有工具日志表。
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(tool_logs)"
                )
            }

            for column in (
                "request_id",
                "session_id",
                "knowledge_base_id",
            ):
                if column not in columns:
                    connection.execute(
                        "ALTER TABLE tool_logs "
                        f"ADD COLUMN {column} TEXT"
                    )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_tool_logs_request_id
                ON tool_logs(request_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_tool_logs_session_kb
                ON tool_logs(
                    session_id,
                    knowledge_base_id
                )
                """
            )

            connection.commit()

    def get_or_bind_session(
        self,
        session_id: str,
        knowledge_base_id: str,
    ) -> str:
        """原子地创建或读取 Session 的知识库绑定。

        返回数据库中实际绑定的 knowledge_base_id。
        如果请求与原绑定不同，不覆盖已有记录。

        调用方必须比较返回值与请求值，
        并在不一致时拒绝请求。
        """

        if not session_id or not knowledge_base_id:
            raise ValueError(
                "session_id 和 knowledge_base_id 不能为空"
            )

        created_at = datetime.now().isoformat(
            timespec="seconds"
        )

        with self._connect() as connection:
            # 获取写事务，串行化不同进程之间的绑定操作。
            connection.execute("BEGIN IMMEDIATE")

            connection.execute(
                """
                INSERT INTO session_bindings (
                    session_id,
                    knowledge_base_id,
                    created_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (
                    session_id,
                    knowledge_base_id,
                    created_at,
                ),
            )

            row = connection.execute(
                """
                SELECT knowledge_base_id
                FROM session_bindings
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                raise RuntimeError(
                    "Session 绑定写入后未能读取记录"
                )

            return str(row["knowledge_base_id"])

    def add_tool_log(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        success: bool,
        duration_ms: float,
        error_type: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        knowledge_base_id: str | None = None,
    ) -> int:
        """
        保存一条工具调用日志。

        Returns:
            新增日志的数据库编号。
        """

        arguments_json = json.dumps(
            arguments,
            ensure_ascii=False,
        )

        created_at = datetime.now().isoformat(
            timespec="seconds"
        )

        sql = """
        INSERT INTO tool_logs (
            tool_name,
            arguments,
            result,
            success,
            error_type,
            duration_ms,
            created_at,
            request_id,
            session_id,
            knowledge_base_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            tool_name,
            arguments_json,
            result,
            int(success),
            error_type,
            duration_ms,
            created_at,
            request_id,
            session_id,
            knowledge_base_id,
        )

        with self._connect() as connection:
            cursor = connection.execute(
                sql,
                values,
            )
            connection.commit()

            return int(cursor.lastrowid)

    def get_recent_tool_logs(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """查询最近的工具调用日志。"""

        if limit <= 0:
            return []

        sql = """
        SELECT
            id,
            tool_name,
            arguments,
            result,
            success,
            error_type,
            duration_ms,
            created_at
        FROM tool_logs
        ORDER BY id DESC
        LIMIT ?
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                (limit,),
            ).fetchall()

        logs: list[dict[str, Any]] = []

        for row in rows:
            log = dict(row)

            # 数据库中的 JSON 字符串恢复为 Python 字典
            try:
                log["arguments"] = json.loads(
                    log["arguments"]
                )
            except json.JSONDecodeError:
                pass

            log["success"] = bool(
                log["success"]
            )

            logs.append(log)

        return logs

    def get_tool_logs_by_request(
        self,
        request_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """查询指定请求产生的工具日志。

        这是内部存储接口，不负责用户身份授权。
        """

        if not request_id or limit <= 0:
            return []

        sql = """
        SELECT
            id,
            tool_name,
            arguments,
            result,
            success,
            error_type,
            duration_ms,
            created_at,
            request_id,
            session_id,
            knowledge_base_id
        FROM tool_logs
        WHERE request_id = ?
        ORDER BY id DESC
        LIMIT ?
        """

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                (request_id, limit),
            ).fetchall()

        logs = []

        for row in rows:
            log = dict(row)

            try:
                log["arguments"] = json.loads(
                    log["arguments"]
                )
            except json.JSONDecodeError:
                pass

            log["success"] = bool(log["success"])
            logs.append(log)

        return logs

    def get_tool_stats(
        self,
    ) -> list[dict[str, Any]]:
        """按工具名称统计调用次数、成功率和平均耗时。"""

        sql = """
        SELECT
            tool_name,
            COUNT(*) AS total_calls,
            SUM(
                CASE
                    WHEN success = 1 THEN 1
                    ELSE 0
                END
            ) AS successful_calls,
            SUM(
                CASE
                    WHEN success = 0 THEN 1
                    ELSE 0
                END
            ) AS failed_calls,
            ROUND(
                100.0 * SUM(
                    CASE
                        WHEN success = 1 THEN 1
                        ELSE 0
                    END
                ) / COUNT(*),
                2
            ) AS success_rate,
            ROUND(
                AVG(duration_ms),
                2
            ) AS average_duration_ms
        FROM tool_logs
        GROUP BY tool_name
        ORDER BY total_calls DESC, tool_name ASC
        """

        with self._connect() as connection:
            rows = connection.execute(sql).fetchall()

        return [
            dict(row)
            for row in rows
        ]