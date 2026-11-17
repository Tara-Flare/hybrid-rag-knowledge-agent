"""千问大模型调用模块。"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from agent_core.prompts import SYSTEM_PROMPT
from storage.trace_context import record_model_usage

# 1. 加上这一行，让它自动读取根目录下的 .env 文件
load_dotenv()


class QwenClient:
    """负责调用千问模型的客户端。"""

    def __init__(
        self,
        model: str = "qwen-plus", # 如果报错，可以换成 "qwen3.7-plus"
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "没有读取到 DASHSCOPE_API_KEY。\n"
                "请检查项目根目录下是否有 .env 文件，并配置了正确的 Key。"
            )

        self.model = model
        self.system_prompt = system_prompt

        # 2. 恢复为你截图里的专属地址
        self.client = OpenAI(
            api_key=api_key,
            base_url=(
                "https://ws-zoy7x6p4hsyfiruk.cn-beijing.maas.aliyuncs.com"
                "/compatible-mode/v1"
            ),
        )

    def create_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> Any:
        """
        调用千问 Chat Completions 接口。

        Args:
            messages: 完整消息列表。
            tools: 可以提供给模型使用的工具列表。

        Returns:
            千问 API 返回的完整响应对象。
        """
        request_data: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = tool_choice

        try:
            response = self.client.chat.completions.create(**request_data)
        except Exception as exc:
            raise RuntimeError(
                f"调用千问模型失败：{exc}"
            ) from exc

        usage = getattr(response, "usage", None)

        # 统一记录当前请求的 Token 用量。
        # chat() 已委托给 create_completion()，
        # 因此不要在 chat() 中重复记录。
        record_model_usage(usage)

        # 保留了你写的 Token 用量打印
        if usage:
            print(
                "\n[Token用量] "
                f"输入：{usage.prompt_tokens}，"
                f"输出：{usage.completion_tokens}，"
                f"总计：{usage.total_tokens}"
            )

        return response


    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> Any:
        return self.create_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )