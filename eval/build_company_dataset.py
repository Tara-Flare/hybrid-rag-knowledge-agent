"""生成和校验 Company 知识库回答评测集 v1。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "knowledge_bases/company"
OUTPUT_DIR = ROOT / "eval/datasets"

DOCUMENTS = {
    "leave": "leave_policy.md",
    "reimbursement": "reimbursement_policy.md",
    "account": "account_policy.md",
}

# 全部证据引自公司模拟制度的原始文本。
QUOTES = {
    "leave": {
        "l1": "正式员工每个自然年度享有5个工作日的基础年假。",
        "l2": "员工原则上需要至少提前2个工作日提交年假申请。",
        "l3": "申请经过直属主管批准后方可生效。",
        "l4": "员工因身体原因无法工作时，应及时通知直属主管。",
        "l5": "病假结束后应按照公司要求补充相关证明材料。",
        "l6": "普通请假由直属主管审批。",
        "l7": "连续请假超过3个工作日的，还需要人力资源部门确认。",
    },
    "reimbursement": {
        "r1": "员工应在费用发生后的15个自然日内提交报销申请。",
        "r2": "报销申请需要提供费用说明、有效票据以及相应的审批记录。",
        "r3": "材料不完整的申请可能被退回。",
        "r4": "员工因公出差产生的合理交通费用，可以按照公司审批流程申请报销。",
        "r5": "普通费用报销由直属主管审批，再交由财务部门审核。",
        "r6": "未经批准的非必要支出可能无法报销。",
    },
    "account": {
        "a1": "新员工入职后，由人力资源部门提交账号开通申请，信息技术部门负责创建账号。",
        "a2": "资料齐全的情况下，信息技术部门原则上在1个工作日内完成基础办公账号开通。",
        "a3": "员工仅能访问岗位职责所需的系统和数据。",
        "a4": "特殊权限需要直属主管和系统负责人共同审批。",
        "a5": "员工离职时，人力资源部门通知信息技术部门关闭相关账号，并按照内部流程回收设备和访问权限。",
    },
}


def F(document: str, quote_key: str, fact: str) -> dict:
    """创建带原文依据的标准事实。"""
    return {
        "fact": fact,
        "source_file": DOCUMENTS[document],
        "evidence_quote": QUOTES[document][quote_key],
    }


def A(
    scenario: str,
    question: str,
    *facts: dict,
    missing: str | None = None,
) -> dict:
    """应当回答的问题，包括部分可回答场景。"""
    return {
        "scenario": scenario,
        "expected_behavior": "answer",
        "question": question,
        "gold_assertions": list(facts),
        "missing_information": missing,
    }


def R(
    scenario: str,
    question: str,
    missing: str,
) -> dict:
    """知识库缺乏依据，应当拒答的问题。"""
    return {
        "scenario": scenario,
        "expected_behavior": "refuse",
        "question": question,
        "gold_assertions": [],
        "missing_information": missing,
    }


DEV = [
    A(
        "factual",
        "根据公司制度，正式员工每个自然年度有多少个工作日的基础年假？",
        F("leave", "l1", "正式员工每个自然年度享有5个工作日基础年假"),
    ),
    A(
        "factual",
        "根据公司知识库，申请年假原则上至少需要提前几个工作日？",
        F("leave", "l2", "年假原则上至少提前2个工作日申请"),
    ),
    A(
        "approval",
        "年假申请提交后，需要谁批准才能生效？",
        F("leave", "l3", "年假申请经直属主管批准后生效"),
    ),
    A(
        "factual",
        "员工因身体原因无法工作时，首先需要通知谁？",
        F("leave", "l4", "应及时通知直属主管"),
    ),
    A(
        "multi_fact",
        "普通请假由谁审批？连续请假超过3个工作日还需要谁确认？",
        F("leave", "l6", "普通请假由直属主管审批"),
        F("leave", "l7", "连续请假超过3个工作日还需人力资源部门确认"),
    ),
    A(
        "numeric",
        "员工应该在费用发生后多少个自然日内提交报销申请？",
        F("reimbursement", "r1", "应在费用发生后15个自然日内提交报销申请"),
    ),
    A(
        "multi_fact",
        "提交费用报销时需要准备哪些材料？",
        F("reimbursement", "r2", "需要费用说明、有效票据和相应审批记录"),
    ),
    A(
        "factual",
        "如果员工提交的报销材料不完整，可能出现什么情况？",
        F("reimbursement", "r3", "材料不完整的申请可能被退回"),
    ),
    A(
        "approval",
        "普通费用报销先经过谁审批，再由哪个部门审核？",
        F("reimbursement", "r5", "先由直属主管审批，再由财务部门审核"),
    ),
    A(
        "multi_fact",
        "新员工的办公账号由哪个部门申请、哪个部门创建？",
        F("account", "a1", "人力资源部门提出申请，信息技术部门创建账号"),
    ),
    A(
        "conditional",
        "新员工资料齐全时，基础办公账号原则上多久开通？",
        F("account", "a2", "资料齐全时原则上1个工作日内开通基础办公账号"),
    ),
    A(
        "approval",
        "如果员工需要特殊系统权限，要经过哪两方共同审批？",
        F("account", "a4", "特殊权限由直属主管和系统负责人共同审批"),
    ),
    A(
        "cross_document",
        "比较年假提前申请时间与费用报销提交期限，分别是多少？请区分工作日和自然日。",
        F("leave", "l2", "年假原则上至少提前2个工作日申请"),
        F("reimbursement", "r1", "费用发生后15个自然日内提交报销"),
    ),
    R(
        "unanswerable",
        "公司规定每次费用报销的最高金额是多少元？",
        "制度未规定单次报销的具体最高金额",
    ),
    R(
        "unanswerable",
        "员工一个自然年度最多可以申请几次年假？",
        "制度未规定每年申请年假的次数上限",
    ),
    A(
        "partial",
        "年假需要提前多久申请？没有用完的年假能结转到下一年吗？",
        F("leave", "l2", "年假原则上至少提前2个工作日申请"),
        missing="制度没有说明未休年假是否可以结转",
    ),
]


HOLDOUT = [
    A(
        "factual",
        "病假结束以后，员工应当怎样处理相关证明材料？",
        F("leave", "l5", "病假结束后按公司要求补充相关证明材料"),
    ),
    A(
        "paraphrase",
        "按照公司账号管理规定，员工是否能随意访问与本职工作无关的数据？",
        F("account", "a3", "员工仅能访问岗位职责所需的系统和数据"),
    ),
    A(
        "factual",
        "因公出差产生的合理交通费，能否按照制度申请报销？",
        F("reimbursement", "r4", "合理的因公差旅交通费用可按审批流程报销"),
    ),
    A(
        "multi_fact",
        "员工离职时，哪个部门通知关闭账号？还需要处理什么？",
        F("account", "a5", "人力资源部门通知信息技术部门关闭账号，并按流程回收设备和权限"),
    ),
    A(
        "multi_fact",
        "正式员工一年的基础年假是多少天？申请获批后才生效吗？",
        F("leave", "l1", "每个自然年度有5个工作日基础年假"),
        F("leave", "l3", "年假申请需经直属主管批准才生效"),
    ),
    A(
        "conditional",
        "未经批准的非必要支出，公司制度是否保证一定报销？",
        F("reimbursement", "r6", "未经批准的非必要支出可能无法报销"),
    ),
    R(
        "unanswerable",
        "公司要求员工忘记办公账号密码后必须通过哪个网址重置？",
        "账号制度未规定密码重置网址",
    ),
    R(
        "unanswerable",
        "公司规定员工一年最多能休多少天带薪病假？",
        "制度未规定带薪病假天数上限",
    ),
    R(
        "cross_knowledge_base",
        "根据星桥科技公司制度，Agent 的 max_steps 默认配置是多少？",
        "company 知识库未规定 Agent max_steps 配置",
    ),
    A(
        "partial",
        "资料齐全时基础办公账号多久开通？账号的初始密码是什么？",
        F("account", "a2", "资料齐全时基础办公账号原则上1个工作日内开通"),
        missing="制度未规定账号的初始密码",
    ),
]


def build_cases(items: list[dict], split: str) -> list[dict]:
    return [
        {
            "id": f"company_{split}_{index:03d}",
            "knowledge_base_id": "company",
            "split": split,
            "annotation_status": "pending_human_review",
            **item,
        }
        for index, item in enumerate(items, 1)
    ]


def validate_cases(cases: list[dict]) -> None:
    """校验证据真实性、标签与问题唯一性。"""

    source_texts = {
        filename: (SOURCE_DIR / filename).read_text(
            encoding="utf-8"
        )
        for filename in DOCUMENTS.values()
    }

    seen_ids = set()
    seen_questions = set()

    for item in cases:
        case_id = item["id"]
        question = item["question"].strip()
        normalized = "".join(question.split())

        if not question or not normalized:
            raise ValueError(f"{case_id} 的问题为空")

        if case_id in seen_ids or normalized in seen_questions:
            raise ValueError(f"重复的评测问题：{case_id}")

        seen_ids.add(case_id)
        seen_questions.add(normalized)

        behavior = item["expected_behavior"]
        assertions = item["gold_assertions"]
        missing = item["missing_information"]

        if behavior not in {"answer", "refuse"}:
            raise ValueError(f"{case_id} 的行为标签不合法")

        if behavior == "answer" and not assertions:
            raise ValueError(f"{case_id} 缺少标准事实")

        if behavior == "refuse" and (assertions or not missing):
            raise ValueError(f"{case_id} 的拒答标注不合法")

        if item["scenario"] == "partial":
            if behavior != "answer" or not missing:
                raise ValueError(f"{case_id} 的部分回答标注不完整")

        if item["scenario"] == "cross_knowledge_base":
            if behavior != "refuse":
                raise ValueError("跨知识库干扰问题必须标注为拒答")

        for assertion in assertions:
            filename = assertion["source_file"]
            quote = assertion["evidence_quote"]
            fact = assertion["fact"]

            if (
                filename not in source_texts
                or not isinstance(quote, str)
                or not quote
                or quote not in source_texts[filename]
                or not isinstance(fact, str)
                or not fact.strip()
            ):
                raise ValueError(
                    f"{case_id} 存在无法核对的 Gold 证据"
                )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查已经生成的数据文件",
    )
    args = parser.parse_args()

    dev = build_cases(DEV, "dev")
    holdout = build_cases(HOLDOUT, "holdout")
    validate_cases(dev + holdout)

    output_files = {
        "company_dev_v1.jsonl": dev,
        "company_holdout_v1.jsonl": holdout,
    }

    encoded_files = {
        name: (
            "\n".join(
                json.dumps(case, ensure_ascii=False)
                for case in cases
            ) + "\n"
        ).encode("utf-8")
        for name, cases in output_files.items()
    }

    manifest = {
        "version": "company_v1",
        "knowledge_base_id": "company",
        "annotation_status": "pending_human_review",
        "counts": {
            "dev": len(dev),
            "holdout": len(holdout),
        },
        "source_sha256": {
            filename: sha256(
                (SOURCE_DIR / filename).read_bytes()
            )
            for filename in DOCUMENTS.values()
        },
        "dataset_sha256": {
            name: sha256(data)
            for name, data in encoded_files.items()
        },
        "note": (
            "基于模拟制度建立的源文对齐标注草案；"
            "投入正式评测前需要人工复核。"
        ),
    }

    manifest_path = OUTPUT_DIR / "company_v1_manifest.json"
    manifest_data = (
        json.dumps(manifest, ensure_ascii=False, indent=2)
        + "\n"
    ).encode("utf-8")

    if args.check:
        for name, data in encoded_files.items():
            path = OUTPUT_DIR / name
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError(f"数据文件不匹配：{path}")

        if (
            not manifest_path.is_file()
            or manifest_path.read_bytes() != manifest_data
        ):
            raise ValueError("Manifest 不匹配")

        print("数据集、证据原文和 SHA256 校验通过")
        return

    targets = [
        OUTPUT_DIR / name
        for name in encoded_files
    ] + [manifest_path]

    if any(path.exists() for path in targets):
        raise FileExistsError(
            "目标文件已存在，拒绝覆盖。"
            "可运行 --check 校验已有数据。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in encoded_files.items():
        with (OUTPUT_DIR / name).open("xb") as f:
            f.write(data)

    with manifest_path.open("xb") as f:
        f.write(manifest_data)

    print("Company 评测集生成成功")
    print("开发集：", len(dev), "道")
    print("Holdout：", len(holdout), "道")
    print("Manifest：", manifest_path)
    print("标注状态：待人工复核")


if __name__ == "__main__":
    main()
