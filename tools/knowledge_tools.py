"""本地知识库检索工具说明。"""

from __future__ import annotations

from typing import Any

SEARCH_KNOWLEDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": (
            "搜索用户的本地知识库，返回与问题最相关的"
            "文本片段及其来源。"
            "当用户询问“根据知识库”“根据项目文档”"
            "“根据我的笔记”或问题明显需要查阅本地资料时，"
            "应调用本工具。"
            "不要用本工具处理简单计算、普通闲聊或用户画像查询。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "用于检索知识库的清晰查询。"
                        "应保留问题中的重要术语。"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "最多返回多少个文本块，"
                        "通常使用 3。"
                    ),
                    "minimum": 1,
                    "maximum": 5,
                    "default": 3,
                },
            },
            "required": [
                "query",
            ],
            "additionalProperties": False,
        },
    },
}