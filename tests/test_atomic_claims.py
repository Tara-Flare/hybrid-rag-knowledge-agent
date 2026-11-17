
"""测试基于 LLM 的原子事实提取器。"""

from unittest.mock import Mock

import pytest

from eval.atomic_claims import extract_atomic_claims


def test_extract_atomic_claims():
    """测试复合句拆分为两条事实。"""

    mock_llm = Mock(
        return_value=(
            '{"claims": ['
            '"Agent 可以调用工具",'
            '"Agent 可以保存长期记忆"'
            ']}'
        )
    )

    answer = (
        "Agent 可以调用工具，并且可以保存长期记忆。\n"
        "来源文件：agent_notes.md；"
        "Chunk ID：agent_notes.md::chunk-0"
    )

    claims = extract_atomic_claims(
        answer,
        mock_llm,
    )

    assert claims == [
        "Agent 可以调用工具",
        "Agent 可以保存长期记忆",
    ]

    mock_llm.assert_called_once()

    # 引用元数据不应进入模型输入
    prompt = mock_llm.call_args.args[0]
    assert "来源文件：" not in prompt


def test_invalid_json():
    """模型返回无效 JSON 时应报错。"""

    mock_llm = Mock(
        return_value="这不是 JSON"
    )

    with pytest.raises(ValueError):
        extract_atomic_claims(
            "Agent 可以调用工具。",
            mock_llm,
        )


def test_empty_answer():
    """空回答不应调用模型。"""

    mock_llm = Mock()

    claims = extract_atomic_claims(
        "",
        mock_llm,
    )

    assert claims == []
    mock_llm.assert_not_called()
