
"""测试不同会话的存储路径是否隔离。"""

from pathlib import Path

import pytest

from agent_core.agent import DesktopAgent


def test_different_sessions_have_different_paths(
    tmp_path: Path,
):
    """不同会话不能共用历史文件。"""

    history_a, profile_a = (
        DesktopAgent._memory_paths(
            "session_A",
            tmp_path,
        )
    )

    history_b, profile_b = (
        DesktopAgent._memory_paths(
            "session_B",
            tmp_path,
        )
    )

    assert history_a != history_b
    assert profile_a != profile_b

    assert history_a.parent.is_dir()
    assert history_b.parent.is_dir()


def test_same_session_has_same_paths(
    tmp_path: Path,
):
    """相同会话应该获得相同存储路径。"""

    paths_1 = DesktopAgent._memory_paths(
        "session_A",
        tmp_path,
    )

    paths_2 = DesktopAgent._memory_paths(
        "session_A",
        tmp_path,
    )

    assert paths_1 == paths_2


def test_invalid_session_id_rejected(
    tmp_path: Path,
):
    """非法会话 ID 应被拒绝。"""

    with pytest.raises(ValueError):
        DesktopAgent._memory_paths(
            "../../other",
            tmp_path,
        )


def test_default_paths_unchanged(
    tmp_path: Path,
):
    """不传 session_id 时兼容旧存储路径。"""

    history, profile = (
        DesktopAgent._memory_paths(
            None,
            tmp_path,
        )
    )

    assert history == (
        tmp_path / "chat_history.json"
    )

    assert profile == (
        tmp_path / "user_profile.json"
    )
