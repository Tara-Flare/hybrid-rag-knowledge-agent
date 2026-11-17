"""SQLite Session 知识库绑定持久化测试。"""

import sqlite3

from storage.sqlite_store import SQLiteStore


def test_session_binding_created(tmp_path):
    """首次使用 Session 时创建绑定。"""

    db = tmp_path / "test.db"
    store = SQLiteStore(str(db))

    result = store.get_or_bind_session(
        "session_A",
        "tech",
    )

    assert result == "tech"

    # 相同绑定可以重复读取
    assert store.get_or_bind_session(
        "session_A",
        "tech",
    ) == "tech"


def test_session_binding_cannot_be_overwritten(tmp_path):
    """不同知识库不能覆盖已有 Session 绑定。"""

    db = tmp_path / "test.db"
    store = SQLiteStore(str(db))

    store.get_or_bind_session(
        "session_A",
        "tech",
    )

    actual = store.get_or_bind_session(
        "session_A",
        "company",
    )

    # 返回实际绑定，而不是覆盖后的值
    assert actual == "tech"

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            """
            SELECT knowledge_base_id
            FROM session_bindings
            WHERE session_id = ?
            """,
            ("session_A",),
        ).fetchone()

    assert row[0] == "tech"


def test_session_binding_survives_restart(tmp_path):
    """重新创建存储对象后，仍然能读取原绑定。"""

    db = tmp_path / "test.db"

    first_store = SQLiteStore(str(db))

    first_store.get_or_bind_session(
        "session_B",
        "company",
    )

    # 模拟进程重启：创建新的 Store 实例
    second_store = SQLiteStore(str(db))

    actual = second_store.get_or_bind_session(
        "session_B",
        "tech",
    )

    assert actual == "company"


def test_tool_logs_table_preserved(tmp_path):
    """新增 Session 表不能破坏工具日志表。"""

    db = tmp_path / "test.db"
    store = SQLiteStore(str(db))

    store.get_or_bind_session(
        "session_A",
        "tech",
    )

    with sqlite3.connect(db) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table'"
            )
        }

    assert "tool_logs" in tables
    assert "session_bindings" in tables
