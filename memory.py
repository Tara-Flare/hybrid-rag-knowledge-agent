"""Agent 的本地记忆管理模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonMemory:
    """使用 JSON 文件保存对话历史。"""

    def __init__(
        self,
        file_path: str = "data/chat_history.json",
        max_messages: int = 20,
    ) -> None:
        self.file_path = Path(file_path)
        self.max_messages = max_messages

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self) -> list[dict[str, Any]]:
        """读取历史消息。"""

        if not self.file_path.exists():
            return []

        try:
            with self.file_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                messages = json.load(file)

        except json.JSONDecodeError:
            print(
                "警告：对话历史文件格式错误，"
                "将使用空历史记录。"
            )
            return []

        except OSError as exc:
            print(f"读取对话历史失败：{exc}")
            return []

        if not isinstance(messages, list):
            print(
                "警告：对话历史不是列表，"
                "将使用空历史记录。"
            )
            return []

        return messages

    def save(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        """保存历史消息。"""

        try:
            with self.file_path.open(
                mode="w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    messages,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        except OSError as exc:
            print(f"保存对话历史失败：{exc}")

    def get_recent_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """返回最近若干条消息。"""

        return messages[-self.max_messages :]

    def clear(self) -> None:
        """清空对话历史。"""

        self.save([])


class UserProfileMemory:
    """保存用户的重要长期信息。"""

    def __init__(
        self,
        file_path: str = "data/user_profile.json",
    ) -> None:
        self.file_path = Path(file_path)

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self) -> dict[str, str]:
        """读取用户长期信息。"""

        if not self.file_path.exists():
            return {}

        try:
            with self.file_path.open(
                mode="r",
                encoding="utf-8",
            ) as file:
                profile = json.load(file)

        except json.JSONDecodeError:
            print(
                "警告：用户信息文件格式错误，"
                "将使用空用户信息。"
            )
            return {}

        except OSError as exc:
            print(f"读取用户信息失败：{exc}")
            return {}

        if not isinstance(profile, dict):
            print(
                "警告：用户信息不是字典，"
                "将使用空用户信息。"
            )
            return {}

        # 只保留字符串形式的键和值
        return {
            str(key): str(value)
            for key, value in profile.items()
        }

    def save(
        self,
        profile: dict[str, str],
    ) -> None:
        """保存用户长期信息。"""

        try:
            with self.file_path.open(
                mode="w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    profile,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

        except OSError as exc:
            print(f"保存用户信息失败：{exc}")

    def set(
        self,
        profile: dict[str, str],
        key: str,
        value: str,
    ) -> None:
        """新增或修改一条长期信息。"""

        profile[key] = value
        self.save(profile)

    def delete(
        self,
        profile: dict[str, str],
        key: str,
    ) -> bool:
        """删除一条长期信息。"""

        if key not in profile:
            return False

        del profile[key]
        self.save(profile)

        return True

    def clear(self) -> None:
        """清空全部长期信息。"""

        self.save({})

    @staticmethod
    def format_for_prompt(
        profile: dict[str, str],
    ) -> str:
        """把长期信息转换为适合发送给模型的文本。"""

        if not profile:
            return "目前没有已保存的用户长期信息。"

        lines = [
            f"- {key}: {value}"
            for key, value in profile.items()
        ]

        return "\n".join(lines)