"""EvidenceChecker 模型输出与异常处理测试。"""

import json

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag.evidence_checker import (
    EvidenceCheckError,
    EvidenceChecker,
)


def fake_checker(payload):
    """构造使用模拟模型响应的检查器。"""

    client = Mock()

    if isinstance(payload, str):
        content = payload

    else:
        content = json.dumps(
            payload,
            ensure_ascii=False,
        )

    client.chat.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content
                )
            )
        ]
    )

    checker = EvidenceChecker(
        client=client
    )

    candidate = SimpleNamespace(
        chunk_id="c1",
        text="测试资料",
    )

    return checker, candidate


def test_sufficient_result():
    """模型返回充分证据的 JSON。"""

    checker, candidate = fake_checker(
        {
            "status": "sufficient",
            "reason": "资料包含定义",
            "supported_chunk_ids": ["c1"],
        }
    )

    result = checker.check(
        "它是什么",
        [candidate],
    )

    assert result.status == "sufficient"

    assert result.supported_chunk_ids == [
        "c1"
    ]

    checker.client.chat.assert_called_once()


def test_insufficient_result():
    """模型能够返回证据不足状态。"""

    checker, candidate = fake_checker(
        {
            "status": "insufficient",
            "reason": "只提到术语",
            "supported_chunk_ids": [],
        }
    )

    result = checker.check(
        "什么意思",
        [candidate],
    )

    assert result.status == "insufficient"


def test_unknown_chunk_id_rejected():
    """模型不能虚构证据来源 ID。"""

    checker, candidate = fake_checker(
        {
            "status": "sufficient",
            "reason": "资料包含定义",
            "supported_chunk_ids": [
                "forged-id"
            ],
        }
    )

    with pytest.raises(EvidenceCheckError):
        checker.check(
            "什么意思",
            [candidate],
        )


def test_malformed_json_rejected():
    """不符合 JSON 规范的模型输出必须报错。"""

    checker, candidate = fake_checker(
        "这段资料可以回答问题"
    )

    with pytest.raises(EvidenceCheckError):
        checker.check(
            "什么意思",
            [candidate],
        )