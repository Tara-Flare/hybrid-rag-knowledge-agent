"""Session 生命周期、知识库绑定与并发管理。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any, Callable, Iterator

from rag.knowledge_base_config import (
    get_knowledge_base_config,
)
from storage.sqlite_store import SQLiteStore


class SessionKnowledgeBaseConflictError(ValueError):
    """同一个 Session 尝试切换知识库。"""


class SessionBusyError(RuntimeError):
    """同一个 Session 已有请求正在执行。"""


@dataclass
class SessionEntry:
    """单个 Session 的运行时状态。"""

    agent: Any
    knowledge_base_id: str
    last_access: float
    request_lock: Any = field(default_factory=Lock)


class SessionManager:
    """管理 Agent 实例、知识库绑定和会话生命周期。"""

    def __init__(
        self,
        agent_factory: Callable,
        ttl_seconds: float = 1800,
        clock: Callable[[], float] = monotonic,
        binding_store: SQLiteStore | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须大于 0")

        self.agent_factory = agent_factory
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.binding_store = binding_store

        # 保留旧属性，兼容已有代码
        self.agents: dict[str, Any] = {}

        # Session 运行时状态
        self._entries: dict[str, SessionEntry] = {}

        # 记录 Session 绑定的知识库
        # 即使 Agent 因 TTL 被清理，也不允许改绑
        self._bindings: dict[str, str] = {}

        # 只保护 Session 元数据
        # 不在聊天过程中持有此锁
        self.lock = Lock()

    def _cleanup_expired_locked(
        self,
        now: float,
    ) -> None:
        """清理超时且没有正在执行请求的 Agent。"""

        expired = []

        for session_id, entry in self._entries.items():
            inactive = now - entry.last_access

            if (
                inactive >= self.ttl_seconds
                and not entry.request_lock.locked()
            ):
                expired.append(session_id)

        for session_id in expired:
            self._entries.pop(session_id, None)
            self.agents.pop(session_id, None)

        # 注意：
        # 不删除 _bindings 中的知识库绑定。
        # 避免 TTL 后复用同名 Session 切换知识库。

    def _create_agent(
        self,
        session_id: str,
        knowledge_base_id: str,
    ) -> Any:
        """创建 Agent，兼容原有单参数工厂。"""

        if knowledge_base_id == "tech":
            # 兼容旧测试中的 lambda session_id
            return self.agent_factory(session_id)

        # 新工厂需要支持第二个知识库参数
        return self.agent_factory(
            session_id,
            knowledge_base_id,
        )

    def _get_entry_locked(
        self,
        session_id: str,
        knowledge_base_id: str,
    ) -> SessionEntry:
        """调用方必须已经持有 self.lock。"""

        # 不允许使用未知知识库
        get_knowledge_base_config(knowledge_base_id)

        bound_kb = self._bindings.get(session_id)

        if (
            bound_kb is not None
            and bound_kb != knowledge_base_id
        ):
            raise SessionKnowledgeBaseConflictError(
                f"Session {session_id!r} 已绑定知识库 "
                f"{bound_kb!r}，不能切换到 "
                f"{knowledge_base_id!r}"
            )

        # 创建 Agent 之前，检查持久化知识库绑定。
        # SQLite 通过原子事务保证首次绑定不会被覆盖。
        if self.binding_store is not None:
            stored_kb = self.binding_store.get_or_bind_session(
                session_id,
                knowledge_base_id,
            )

            if stored_kb != knowledge_base_id:
                raise SessionKnowledgeBaseConflictError(
                    f"Session {session_id!r} 已持久化绑定 "
                    f"知识库 {stored_kb!r}，不能切换到 "
                    f"{knowledge_base_id!r}"
                )

        now = self.clock()

        self._cleanup_expired_locked(now)

        entry = self._entries.get(session_id)

        if entry is None:
            agent = self._create_agent(
                session_id,
                knowledge_base_id,
            )

            entry = SessionEntry(
                agent=agent,
                knowledge_base_id=knowledge_base_id,
                last_access=now,
            )

            self._entries[session_id] = entry
            self.agents[session_id] = agent
            self._bindings[session_id] = knowledge_base_id

        entry.last_access = now

        return entry

    def get_agent(
        self,
        session_id: str,
        knowledge_base_id: str = "tech",
    ) -> Any:
        """获取 Agent，兼容旧接口。

        注意：聊天执行应使用 use_agent()，
        以获得会话级并发保护。
        """

        with self.lock:
            entry = self._get_entry_locked(
                session_id,
                knowledge_base_id,
            )

            return entry.agent

    @contextmanager
    def use_agent(
        self,
        session_id: str,
        knowledge_base_id: str = "tech",
    ) -> Iterator[Any]:
        """独占使用某个 Session 的 Agent。

        同 Session 并发请求立即失败，
        不同 Session 使用独立请求锁。
        """

        with self.lock:
            entry = self._get_entry_locked(
                session_id,
                knowledge_base_id,
            )

            acquired = entry.request_lock.acquire(
                blocking=False
            )

            if not acquired:
                raise SessionBusyError(
                    f"Session {session_id!r} "
                    "已有请求正在执行"
                )

        try:
            yield entry.agent

        finally:
            with self.lock:
                entry.last_access = self.clock()
                entry.request_lock.release()
