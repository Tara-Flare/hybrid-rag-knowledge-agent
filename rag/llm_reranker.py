from __future__ import annotations

import json
import os
from typing import Any, Sequence, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from storage.trace_context import record_model_usage

ResultType = TypeVar("ResultType")

class LLMReranker:
    """
    使用大模型对召回候选 Chunk 进行重排序。

    RRF 负责扩大和融合候选集合；
    LLM Reranker 负责判断哪个 Chunk
    最能够直接、完整地回答用户问题。
    """

    def __init__(
        self,
        model: str | None = None,
        max_chars_per_chunk: int = 800,
    ) -> None:
        load_dotenv()

        api_key = os.getenv(
            "DASHSCOPE_API_KEY"
        )

        base_url = os.getenv(
            "DASHSCOPE_BASE_URL"
        )

        if not api_key:
            raise ValueError(
                "没有找到环境变量 "
                "DASHSCOPE_API_KEY"
            )

        if not base_url:
            raise ValueError(
                "没有找到环境变量 "
                "DASHSCOPE_BASE_URL"
            )

        self.model = (
            model
            or os.getenv("QWEN_MODEL")
            or "qwen-plus"
        )

        self.max_chars_per_chunk = (
            max_chars_per_chunk
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=20.0,
            max_retries=0,
        )

        # 保存最近一次重排序理由，
        # 方便测试和调试。
        self.last_reason = ""

    def _get_attribute(
        self,
        obj: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        """
        同时兼容普通对象和字典。
        """
        if isinstance(obj, dict):
            return obj.get(name, default)

        return getattr(obj, name, default)

    def _build_candidates_text(
        self,
        candidates: Sequence[Any],
    ) -> str:
        """
        把候选 Chunk 整理成发送给大模型的文本。
        """
        blocks: list[str] = []

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            chunk_id = str(
                self._get_attribute(
                    candidate,
                    "chunk_id",
                    "",
                )
            )

            text = str(
                self._get_attribute(
                    candidate,
                    "text",
                    "",
                )
            )

            source = str(
                self._get_attribute(
                    candidate,
                    "source",
                    "unknown",
                )
            )

            # 防止单个 Chunk 过长，造成 Token 浪费。
            text = text[
                :self.max_chars_per_chunk
            ]

            blocks.append(
                f"""候选 {index}
chunk_id: {chunk_id}
source: {source}
text:
{text}
"""
            )

        return "\n".join(blocks)

    def _extract_json(
        self,
        response_text: str,
    ) -> dict[str, Any]:
        """
        从模型输出中提取 JSON。

        即使模型额外输出了 ```json，
        也尽量把 JSON 对象取出来。
        """
        response_text = response_text.strip()

        start = response_text.find("{")
        end = response_text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "模型输出中没有找到 JSON 对象"
            )

        json_text = response_text[
            start : end + 1
        ]

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"模型输出不是有效 JSON："
                f"{response_text}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "模型输出的 JSON 顶层必须是对象"
            )

        return payload

    def rerank(
        self,
        query: str,
        candidates: Sequence[ResultType],
        top_k: int = 3,
    ) -> list[ResultType]:
        """
        对候选 Chunk 重排序。
        """
        if not isinstance(query, str):
            raise TypeError(
                "query 必须是字符串"
            )

        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        if not candidates:
            return []

        candidate_map: dict[str, ResultType] = {}

        original_ids: list[str] = []

        for candidate in candidates:
            chunk_id = str(
                self._get_attribute(
                    candidate,
                    "chunk_id",
                    "",
                )
            ).strip()

            if not chunk_id:
                raise ValueError(
                    "候选结果缺少 chunk_id"
                )

            if chunk_id not in candidate_map:
                candidate_map[chunk_id] = (
                    candidate
                )
                original_ids.append(chunk_id)

        candidates_text = (
            self._build_candidates_text(
                candidates
            )
        )

        system_prompt = """
你是一个严格的 RAG 检索重排序器。

你的任务不是回答用户问题，而是判断候选文本与问题的相关程度，
并将最能够直接、完整回答问题的候选排在前面。

排序时遵守以下规则：

1. 优先选择真正包含问题答案的文本。
2. 仅仅出现相同关键词，不代表能够回答问题。
3. 能够直接解释“是什么、为什么、有什么区别”的文本，
   应排在只提到相关主题的文本前面。
4. 不允许补充候选文本之外的知识。
5. chunk_id 必须原样返回，不得修改。
6. 只输出 JSON，不要输出 Markdown。

输出格式：

{
  "ranking": [
    "第一个 chunk_id",
    "第二个 chunk_id"
  ],
  "reason": "简要说明排序依据"
}
""".strip()

        user_prompt = f"""
用户问题：

{query}

下面是候选文本：

{candidates_text}

请对所有候选文本进行排序。
""".strip()

        response = (
            self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )
        )

        # 记录 Rerank 的独立模型调用及 Token 用量
        record_model_usage(
            getattr(response, "usage", None)
        )

        response_text = (
            response.choices[0]
            .message.content
            or ""
        )

        payload = self._extract_json(
            response_text
        )

        raw_ranking = payload.get(
            "ranking",
            [],
        )

        self.last_reason = str(
            payload.get("reason", "")
        )

        if not isinstance(raw_ranking, list):
            raise ValueError(
                "ranking 必须是列表"
            )

        ordered_ids: list[str] = []
        seen_ids: set[str] = set()

        # 只接受真实存在的候选 ID。
        for raw_chunk_id in raw_ranking:
            chunk_id = str(
                raw_chunk_id
            ).strip()

            if (
                chunk_id in candidate_map
                and chunk_id not in seen_ids
            ):
                ordered_ids.append(chunk_id)
                seen_ids.add(chunk_id)

        # 模型可能遗漏候选项。
        # 被遗漏的结果按照原始 RRF 顺序补到末尾。
        for chunk_id in original_ids:
            if chunk_id not in seen_ids:
                ordered_ids.append(chunk_id)
                seen_ids.add(chunk_id)

        return [
            candidate_map[chunk_id]
            for chunk_id in ordered_ids[
                :top_k
            ]
        ]