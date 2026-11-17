
"""API 接口鉴权工具。"""

import os
import secrets

from fastapi import Header, HTTPException


def require_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    """检查请求是否提供正确的 API Key。"""

    expected_key = os.getenv("AGENT_DEV_API_KEY")

    # 没有配置密钥时，禁止访问
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API Key 未配置",
        )

    # 缺少密钥或者密钥不正确
    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="API Key 无效",
        )
