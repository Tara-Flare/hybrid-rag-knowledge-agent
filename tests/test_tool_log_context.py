"""工具日志与 RequestTrace 关联测试。"""

import sqlite3

from agent_core.agent import DesktopAgent
from storage.request_trace import RequestTrace
from storage.sqlite_store import SQLiteStore
from storage.trace_context import bind_request_trace


def make_agent(store):
    agent = DesktopAgent.__new__(DesktopAgent)
    agent.store = store
    agent.tool_functions = {
        "echo": lambda value: f"收到：{value}"
    }
    return agent


def test_legacy_database_migration(tmp_path):
    """迁移必须保留原有工具日志。"""

    db_path = tmp_path / "legacy.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE tool_logs (
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
        )

        conn.execute(
            """
            INSERT INTO tool_logs (
                tool_name, arguments, result,
                success, error_type, duration_ms,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_tool",
                '{"value": 1}',
                "旧记录",
                1,
                None,
                5.0,
                "2026-01-01T00:00:00",
            ),
        )

    store = SQLiteStore(str(db_path))

    # 重复初始化不能重复迁移或丢失数据
    SQLiteStore(str(db_path))

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(tool_logs)"
            )
        }

        row = conn.execute(
            """
            SELECT
                tool_name, request_id,
                session_id, knowledge_base_id
            FROM tool_logs
            WHERE id = 1
            """
        ).fetchone()

    assert {
        "request_id",
        "session_id",
        "knowledge_base_id",
    }.issubset(columns)

    assert row == (
        "legacy_tool",
        None,
        None,
        None,
    )

    assert len(store.get_recent_tool_logs()) == 1


def test_tool_logs_are_linked_to_correct_request(tmp_path):
    """不同请求、不同知识库的日志不得混淆。"""

    store = SQLiteStore(str(tmp_path / "agent.db"))
    agent = make_agent(store)

    trace_a = RequestTrace(
        session_id="session_A",
        knowledge_base_id="tech",
    )

    trace_b = RequestTrace(
        session_id="session_B",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace_a):
        agent._run_tool("echo", {"value": "技术知识库"})

    with bind_request_trace(trace_b):
        agent._run_tool("echo", {"value": "公司知识库"})

    logs_a = store.get_tool_logs_by_request(
        trace_a.request_id
    )

    logs_b = store.get_tool_logs_by_request(
        trace_b.request_id
    )

    assert len(logs_a) == 1
    assert len(logs_b) == 1

    assert logs_a[0]["request_id"] == trace_a.request_id
    assert logs_a[0]["session_id"] == "session_A"
    assert logs_a[0]["knowledge_base_id"] == "tech"
    assert logs_a[0]["arguments"]["value"] == "技术知识库"

    assert logs_b[0]["request_id"] == trace_b.request_id
    assert logs_b[0]["session_id"] == "session_B"
    assert logs_b[0]["knowledge_base_id"] == "company"
    assert logs_b[0]["arguments"]["value"] == "公司知识库"


def test_tool_failure_is_linked_to_trace(tmp_path):
    """ToolResult.fail 也必须正确关联请求。"""

    store = SQLiteStore(str(tmp_path / "agent.db"))
    agent = make_agent(store)

    trace = RequestTrace(
        session_id="session_failed",
        knowledge_base_id="company",
    )

    with bind_request_trace(trace):
        agent._run_tool("nonexistent_tool", {})

    logs = store.get_tool_logs_by_request(
        trace.request_id
    )

    assert len(logs) == 1
    assert logs[0]["success"] is False
    assert logs[0]["error_type"] == "ToolNotFoundError"
    assert logs[0]["request_id"] == trace.request_id


def test_standalone_agent_remains_compatible(tmp_path):
    """不绑定 Trace 的命令行 Agent 仍能记录工具日志。"""

    store = SQLiteStore(str(tmp_path / "agent.db"))
    agent = make_agent(store)

    agent._run_tool(
        "echo",
        {"value": "standalone"},
    )

    logs = store.get_recent_tool_logs()

    assert len(logs) == 1
    assert logs[0]["tool_name"] == "echo"
    assert logs[0]["success"] is True

    # 没有 RequestTrace 的日志不能错误关联任何请求
    assert store.get_tool_logs_by_request(
        "nonexistent_request"
    ) == []
