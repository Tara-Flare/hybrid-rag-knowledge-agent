
"""从 Agent 回答中提取待核验的候选句。"""

import re


def extract_candidate_claims(
    answer: str,
) -> list[str]:
    """按句子切分回答，并排除引用元数据。

    注意：当前提取的是候选句，
    不保证每个句子只包含一个原子事实。
    """

    if not answer.strip():
        return []

    # 1. 删除来源文件、Chunk ID 等引用元数据行
    lines = []

    for line in answer.splitlines():
        line = line.strip()

        if line.startswith(
            ("来源文件：", "Chunk ID：")
        ):
            continue

        if line:
            lines.append(line)

    # 2. 合并正文
    text = "\n".join(lines)

    # 3. 按句末标点或换行切分
    parts = re.split(
        r"[。！？!?]+|\n+",
        text,
    )

    # 4. 清理空白和引号
    claims = []

    for part in parts:
        claim = part.strip(
            ' \t\r\n“”"'
        )

        if claim:
            claims.append(claim)

    return claims
