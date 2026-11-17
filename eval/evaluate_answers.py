
"""RAG 最终回答评测：第一版。"""

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from agent_core.agent import DesktopAgent


# 测试数据集
CASES_PATH = Path("eval/answer_cases.jsonl")

# 评测结果文件

RESULTS_PATH = Path(
    "eval/answer_results_with_evidence.jsonl"
)

def main():
    """逐条运行测试问题并记录回答。"""

    cases = []

    # 1. 读取评测数据集
    with CASES_PATH.open(
        "r", encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"本次共有 {len(cases)} 个测试问题")

    # 2. 创建结果文件
    with RESULTS_PATH.open(
        "w", encoding="utf-8"
    ) as output:

        # 3. 逐条调用 Agent
        for case in cases:
            print(f"\n正在评测：{case['id']}")
            print("问题：", case["question"])

            # 每个测试使用全新的会话
            session_id = (
                f"eval_{case['id']}_"
                f"{uuid4().hex[:8]}"
            )

            agent = DesktopAgent(
                session_id=session_id
            )
            # 为当前测试创建独立的证据记录列表
            agent._eval_evidence = []

            start = perf_counter()

            answer = agent.chat(
                case["question"]
            )

            duration_ms = round(
                (perf_counter() - start) * 1000,
                2,
            )

            # 4. 整理评测结果
            result = {
                "id": case["id"],
                "question": case["question"],
                "expected_behavior": case[
                    "expected_behavior"
                ],
                "answer": str(answer),
                "duration_ms": duration_ms,
                # 新增：保存同一次回答所使用的证据
                "evidence": list(agent._eval_evidence),
            }

            # 5. 保存为 JSONL
            output.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                ) + "\n"
            )

            output.flush()

            print("Agent 回答：", answer)
            print("响应时间：", duration_ms, "ms")

    print("\n评测执行完成！")
    print("结果文件：", RESULTS_PATH)


if __name__ == "__main__":
    main()
