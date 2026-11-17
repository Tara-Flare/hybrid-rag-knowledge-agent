
"""RAG 回答事实忠实度评分。"""


VALID_LABELS = {
    "supported",
    "unsupported",
    "uncertain",
}


def calculate_faithfulness(
    labels: list[str],
) -> float | None:
    """根据事实支持标签计算忠实度。

    Args:
        labels: 每条事实的证据支持情况。

    Returns:
        0.0～1.0 之间的分数。
        没有事实陈述时返回 None。
    """

    # 没有事实陈述时，不能直接给满分
    if not labels:
        return None

    # 检查标签是否合法
    for label in labels:
        if label not in VALID_LABELS:
            raise ValueError(
                f"不支持的事实标签：{label}"
            )

    # 统计有证据支持的事实数量
    supported_count = sum(
        label == "supported"
        for label in labels
    )

    # 计算忠实度
    return supported_count / len(labels)
