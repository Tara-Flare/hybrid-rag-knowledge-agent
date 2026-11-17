"""将旧版技术知识库 Session 绑定迁移到 SQLite。"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path


BASE_DIR = Path("data/sessions")
DB_PATH = Path("data/agent.db")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行迁移；默认只预览",
    )
    args = parser.parse_args()

    if not DB_PATH.is_file():
        raise FileNotFoundError(DB_PATH)

    # 仅迁移真正存在旧版历史文件的 Session。
    candidates = []

    if BASE_DIR.is_dir():
        for directory in sorted(BASE_DIR.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                continue

            session_id = directory.name

            if not re.fullmatch(
                r"[A-Za-z0-9_-]{1,64}",
                session_id,
            ):
                raise ValueError(
                    f"发现非法 Session ID：{session_id}"
                )

            has_legacy_history = any(
                (directory / filename).is_file()
                for filename in (
                    "chat_history.json",
                    "user_profile.json",
                )
            )

            if has_legacy_history:
                candidates.append(session_id)

    connection = sqlite3.connect(DB_PATH, timeout=10)

    try:
        existing = dict(
            connection.execute(
                """
                SELECT session_id, knowledge_base_id
                FROM session_bindings
                """
            ).fetchall()
        )

        # 已绑定为 tech 的无需迁移。
        # 已绑定其他知识库的旧 Session 必须人工检查。
        conflicts = [
            sid for sid in candidates
            if sid in existing and existing[sid] != "tech"
        ]

        if conflicts:
            raise RuntimeError(
                f"发现知识库绑定冲突，停止迁移：{conflicts}"
            )

        pending = [
            sid for sid in candidates
            if sid not in existing
        ]

        print("========== 旧 Session 迁移检查 ==========")
        print("存在旧历史的 Session：", len(candidates))
        print("已有绑定，无需迁移：", len(candidates) - len(pending))
        print("待迁移 Session：", len(pending))

        for sid in pending:
            print("  tech <-", sid)

        if not args.apply:
            print("\n预览完成。添加 --apply 执行迁移。")
            return

        if not pending:
            print("\n没有需要迁移的记录。")
            return

        # 使用 SQLite 在线备份，避免直接复制运行中的数据库。
        backup_path = DB_PATH.with_name(
            "agent.db.before_session_migration_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + ".bak"
        )

        with sqlite3.connect(backup_path) as backup:
            connection.backup(backup)

        print("\n数据库备份：", backup_path)

        # 在一个写事务中执行迁移。
        # 出现冲突时回滚全部本次写入。
        connection.execute("BEGIN IMMEDIATE")

        try:
            now = datetime.now().isoformat(
                timespec="seconds"
            )

            for sid in pending:
                connection.execute(
                    """
                    INSERT INTO session_bindings (
                        session_id,
                        knowledge_base_id,
                        created_at
                    )
                    VALUES (?, 'tech', ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    (sid, now),
                )

                actual = connection.execute(
                    """
                    SELECT knowledge_base_id
                    FROM session_bindings
                    WHERE session_id = ?
                    """,
                    (sid,),
                ).fetchone()

                if actual is None or actual[0] != "tech":
                    raise RuntimeError(
                        f"Session 绑定冲突：{sid}"
                    )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        print(f"\n迁移成功：{len(pending)} 个 Session")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
