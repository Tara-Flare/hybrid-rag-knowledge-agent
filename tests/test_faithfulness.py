
"""测试事实忠实度计算函数。"""

import pytest

from eval.faithfulness import calculate_faithfulness


def test_all_claims_supported():
    """全部事实都有证据支持。"""

    labels = [
        "supported",
        "supported",
    ]

    assert calculate_faithfulness(labels) == 1.0


def test_partially_supported():
    """部分事实得到证据支持。"""

    labels = [
        "supported",
        "unsupported",
        "uncertain",
        "supported",
    ]

    assert calculate_faithfulness(labels) == 0.5


def test_no_claims():
    """没有事实陈述时不计算忠实度。"""

    assert calculate_faithfulness([]) is None


def test_invalid_label():
    """不合法的事实标签应被拒绝。"""

    with pytest.raises(ValueError):
        calculate_faithfulness(
            ["supported", "invalid"]
        )
