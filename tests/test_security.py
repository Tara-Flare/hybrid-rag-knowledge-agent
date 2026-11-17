from api.security import require_api_key


def test_correct_api_key(monkeypatch):
    """正确的 API Key 应通过验证。"""

    monkeypatch.setenv(
        "AGENT_DEV_API_KEY",
        "test-secret-123",
    )

    # 不应抛出异常
    assert require_api_key(
        x_api_key="test-secret-123"
    ) is None