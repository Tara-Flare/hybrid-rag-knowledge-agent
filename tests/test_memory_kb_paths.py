"""知识库记忆路径隔离测试。"""

import pytest

from agent_core.agent import DesktopAgent


def test_memory_paths_isolated(tmp_path):
    """相同 Session 的不同知识库必须使用不同文件。"""

    tech_history, tech_profile = DesktopAgent._memory_paths(
        "user001",
        base_dir=tmp_path,
        knowledge_base_id="tech",
    )

    company_history, company_profile = DesktopAgent._memory_paths(
        "user001",
        base_dir=tmp_path,
        knowledge_base_id="company",
    )

    assert tech_history != company_history
    assert tech_profile != company_profile

    tech_history.write_text(
        "tech_history",
        encoding="utf-8",
    )

    assert not company_history.exists()


def test_legacy_tech_path_unchanged(tmp_path):
    """技术库历史路径必须保持向后兼容。"""

    history, profile = DesktopAgent._memory_paths(
        "user001",
        base_dir=tmp_path,
        knowledge_base_id="tech",
    )

    assert history == (
        tmp_path / "sessions/user001/chat_history.json"
    )

    assert profile == (
        tmp_path / "sessions/user001/user_profile.json"
    )


def test_company_requires_session_id(tmp_path):
    """企业知识库不允许使用全局默认记忆。"""

    with pytest.raises(ValueError):
        DesktopAgent._memory_paths(
            None,
            base_dir=tmp_path,
            knowledge_base_id="company",
        )
