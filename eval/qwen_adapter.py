
"""将项目中的 QwenClient 适配为文本生成函数。"""

from agent_core.llm_client import QwenClient


def qwen_generate(prompt: str) -> str:
    """输入 Prompt，返回千问生成的纯文本。"""

    client = QwenClient()

    response = client.create_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个事实提取工具。"
                    "请严格按照要求输出合法 JSON，"
                    "不要输出 Markdown 或额外解释。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    # 提取模型生成的文字
    content = response.choices[0].message.content

    if not isinstance(content, str) or not content.strip():
        raise ValueError("千问返回了空文本或非文本内容")

    return content.strip()
