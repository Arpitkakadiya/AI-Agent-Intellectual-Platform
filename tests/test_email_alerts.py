"""Unit tests for email subscription helpers (mocked DB / SMTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from notifications.email_alerts import EmailAlertsService


@pytest.fixture
def svc() -> EmailAlertsService:
    return EmailAlertsService()


def test_unsubscribe_not_found_returns_status(svc: EmailAlertsService) -> None:
    mock_db = MagicMock()
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = MagicMock(data=[])

    juris = MagicMock()
    juris.select.return_value = juris
    juris.eq.return_value = juris
    juris.limit.return_value = juris
    juris.execute.return_value = MagicMock(data=[{"name": "Test State"}])

    def table(name: str) -> MagicMock:
        return juris if name == "jurisdictions" else chain

    mock_db.table.side_effect = table

    with patch("notifications.email_alerts.get_db", return_value=mock_db):
        with patch.object(svc, "send_unsubscribe_confirmation_email") as send:
            out = svc.unsubscribe(email="a@b.com", jurisdiction_id=1)
    assert out == {"status": "not_found"}
    send.assert_not_called()


def test_unsubscribe_sends_confirmation_when_row_updated(
    svc: EmailAlertsService,
) -> None:
    mock_db = MagicMock()
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.update.return_value = chain
    chain.execute.side_effect = [
        MagicMock(data=[{"id": 1}]),
        MagicMock(data=[]),
    ]

    juris = MagicMock()
    juris.select.return_value = juris
    juris.eq.return_value = juris
    juris.limit.return_value = juris
    juris.execute.return_value = MagicMock(data=[{"name": "Test State"}])

    def table(name: str) -> MagicMock:
        return juris if name == "jurisdictions" else chain

    mock_db.table.side_effect = table

    with patch("notifications.email_alerts.get_db", return_value=mock_db):
        with patch.object(svc, "send_unsubscribe_confirmation_email") as send:
            out = svc.unsubscribe(email="a@b.com", jurisdiction_id=1)
    assert out == {"status": "unsubscribed"}
    send.assert_called_once_with(
        email="a@b.com", jurisdiction_name="Test State"
    )


def test_subscribe_skips_welcome_when_already_active(
    svc: EmailAlertsService,
) -> None:
    mock_db = MagicMock()
    email_table = MagicMock()
    email_table.select.return_value = email_table
    email_table.eq.return_value = email_table
    email_table.limit.return_value = email_table
    email_table.upsert.return_value = email_table
    email_table.execute.side_effect = [
        MagicMock(data=[{"is_active": True}]),
        MagicMock(data=[]),
    ]

    juris = MagicMock()
    juris.select.return_value = juris
    juris.eq.return_value = juris
    juris.limit.return_value = juris
    juris.execute.return_value = MagicMock(data=[{"name": "X"}])

    def table(name: str) -> MagicMock:
        return juris if name == "jurisdictions" else email_table

    mock_db.table.side_effect = table

    with patch("notifications.email_alerts.get_db", return_value=mock_db):
        with patch.object(svc, "send_welcome_email") as welcome:
            out = svc.subscribe(email="a@b.com", jurisdiction_id=1)
    assert out == {"status": "subscribed"}
    welcome.assert_not_called()
