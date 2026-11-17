"""桌面智能助手命令行入口。"""

from __future__ import annotations

from agent_core.agent import DesktopAgent

def show_help() -> None:
    """显示命令帮助。"""

    print(
        "\n可用命令：\n"
        "  /help           查看帮助\n"
        "  /history        查看最近的对话历史\n"
        "  /profile        查看用户长期信息\n"
        "  /forget KEY      删除指定长期信息\n"
        "  /tool-logs       查看最近的工具调用日志\n"
        "  /tool-stats      查看工具调用统计\n"
        "  /clear          清空对话历史\n"
        "  /clear-profile  清空用户长期信息\n"
        "  /exit           退出程序\n"
    )

def show_profile(
    agent: DesktopAgent,
) -> None:
    """显示用户长期信息。"""

    profile = agent.get_profile()

    if not profile:
        print("\n当前没有用户长期信息。")
        return

    print("\n用户长期信息：")

    for key, value in profile.items():
        print(f"- {key}: {value}")

def show_history(
    agent: DesktopAgent,
) -> None:
    """显示最近的对话历史。"""

    history = agent.get_history(limit=10)

    if not history:
        print("\n当前没有历史对话。")
        return

    print("\n最近的对话历史：")

    for message in history:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            role_name = "你"
        elif role == "assistant":
            role_name = "Agent"
        else:
            role_name = str(role)

        print(f"\n{role_name}：{content}")


def show_tool_logs(
    agent: DesktopAgent,
) -> None:
    """显示最近的工具调用日志。"""

    logs = agent.get_tool_logs(limit=10)

    if not logs:
        print("\n当前没有工具调用日志。")
        return

    print("\n最近的工具调用日志：")

    for log in logs:
        status = (
            "成功"
            if log["success"]
            else "失败"
        )

        print(
            "\n"
            f"编号：{log['id']}\n"
            f"时间：{log['created_at']}\n"
            f"工具：{log['tool_name']}\n"
            f"参数：{log['arguments']}\n"
            f"状态：{status}\n"
            f"结果：{log['result']}\n"
            f"异常：{log['error_type']}\n"
            f"耗时：{log['duration_ms']:.2f} ms"
        )


def show_tool_stats(
    agent: DesktopAgent,
) -> None:
    """显示工具调用统计。"""

    stats = agent.get_tool_stats()

    if not stats:
        print("\n当前没有工具调用统计。")
        return

    print("\n工具调用统计：")

    for item in stats:
        print(
            "\n"
            f"工具：{item['tool_name']}\n"
            f"总调用次数：{item['total_calls']}\n"
            f"成功次数：{item['successful_calls']}\n"
            f"失败次数：{item['failed_calls']}\n"
            f"成功率：{item['success_rate']:.2f}%\n"
            f"平均耗时："
            f"{item['average_duration_ms']:.2f} ms"
        )


def main() -> None:
    """启动命令行 Agent。"""

    try:
        agent = DesktopAgent(max_steps=5)

    except Exception as exc:
        print(
            "Agent 初始化失败："
            f"{type(exc).__name__}: {exc}"
        )
        return

    print("=" * 50)
    print("桌面智能助手已启动")
    print("输入 /help 查看命令")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你：").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nAgent：程序已退出。")
            break

        if not user_input:
            print("请输入内容。")
            continue

        command = user_input.lower()

        # 所有本地命令都必须放在 agent.chat() 前面
        if command in {
            "/exit",
            "exit",
            "quit",
        }:
            print("Agent：再见！")
            break

        if command == "/help":
            show_help()
            continue

        if command == "/history":
            show_history(agent)
            continue

        if command == "/profile":
            show_profile(agent)
            continue

        if command == "/clear":
            agent.clear_history()
            print("Agent：对话历史已清空。")
            continue

        if command == "/clear-profile":
            agent.clear_profile()
            print("Agent：用户长期信息已清空。")
            continue

        if command == "/tool-logs":
            show_tool_logs(agent)
            continue

        if command == "/tool-stats":
            show_tool_stats(agent)
            continue

        if command.startswith("/forget "):
            key = user_input[len("/forget ") :].strip()

            if not key:
                print("请输入需要删除的记忆名称。")
                continue

            try:
                result = agent.delete_profile_item(key)

            except ValueError as exc:
                print(f"删除失败：{exc}")

            except Exception as exc:
                print(
                    "删除长期信息时发生错误："
                    f"{type(exc).__name__}: {exc}"
                )

            else:
                print(f"\nAgent：{result}")

            continue

        # 不是本地命令，才发送给大模型
        try:
            answer = agent.chat(user_input)

        except ValueError as exc:
            print(f"输入错误：{exc}")

        except RuntimeError as exc:
            print(f"Agent 运行失败：{exc}")

        except Exception as exc:
            print(
                "程序发生未知错误："
                f"{type(exc).__name__}: {exc}"
            )

        else:
            print(f"\nAgent：{answer}")

if __name__ == "__main__":
    main()