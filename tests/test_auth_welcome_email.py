import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "api-gateway"))

import auth_routes


class FakeUsersCollection:
    def __init__(self):
        self.find_one = AsyncMock(return_value=None)
        self.insert_one = AsyncMock(return_value=SimpleNamespace(inserted_id="fake-id"))
        self.update_one = AsyncMock(return_value=None)


@pytest.mark.asyncio
async def test_signup_sends_welcome_email():
    fake_collection = FakeUsersCollection()

    with patch.object(auth_routes, "users_collection", fake_collection), patch.object(auth_routes, "send_welcome_email") as send_email:
        response = await auth_routes.signup(
            auth_routes.UserSignup(username="alice", email="alice@example.com", password="secret")
        )

    assert response["status"] == "success"
    send_email.assert_called_once_with("alice@example.com", "alice")
