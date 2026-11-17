"""长期记忆工具的模型说明。"""

from __future__ import annotations

from typing import Any


SAVE_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "save_memory",
        "description": (
            "保存用户明确提供并且未来仍然有用的长期信息。"
            "例如用户的姓名、长期学习目标、项目名称、"
            "稳定偏好或用户明确要求记住的信息。"
            "不要保存临时问题、一次性计算结果、密码、"
            "API Key 或其他敏感信息。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "信息名称，例如 name、"
                        "learning_goal、current_project。"
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "需要长期保存的信息内容。",
                },
            },
            "required": [
                "key",
                "value",
            ],
            "additionalProperties": False,
        },
    },
}


DELETE_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "delete_memory",
        "description": (
            "删除指定的用户长期信息。"
            "只要用户明确要求忘记、删除或清除某条长期信息，"
            "就必须调用本工具。"
            "即使该信息可能不存在，也必须调用工具确认，"
            "不能仅根据当前上下文直接回答。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "需要删除的长期信息键名，"
                        "例如 name、learning_goal、favorite_movie。"
                    ),
                },
            },
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}