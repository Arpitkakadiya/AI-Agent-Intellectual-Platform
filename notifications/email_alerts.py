from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

from config import settings
from core.regulations.update_checker import UpdateResult
from db.client import get_db


EMAILS_DIR = Path(__file__).resolve().parents[1] / "emails"


def save_email_to_folder(email_content: str) -> str:
    """
    Fallback storage when SMTP is not configured.
    Saves the raw email content into the `emails/` folder.
    """
    try:
        EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot create a folder, fail silently (best-effort backup).
        return ""

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = EMAILS_DIR / f"email_{timestamp}.txt"
    try:
        filename.write_text(email_content, encoding="utf-8")
    except Exception:
        return ""
    return str(filename)


def _smtp_send(to_email: str, subject: str, body: str) -> bool:
    if not settings.has_smtp:
        content = f"To: {to_email}\nSubject: {subject}\n\n{body}"
        save_email_to_folder(content)
        return True

    smtp_host = settings.SMTP_HOST or ""
    smtp_port = int(settings.SMTP_PORT or 0)
    smtp_user = settings.SMTP_EMAIL or ""
    smtp_pass = settings.SMTP_PASSWORD or ""

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        if smtp_pass:
            server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        return True
    except Exception:
        content = f"To: {to_email}\nSubject: {subject}\n\n{body}"
        save_email_to_folder(content)
        return True


def _format_detected_at(detected_at: Any) -> str:
    if detected_at is None:
        return datetime.utcnow().strftime("%Y-%m-%d")
    text = str(detected_at)
    # Common cases: "YYYY-MM-DD HH:MM:SS" or ISO.
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text[: len(fmt)], fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return text.split("T")[0].strip()


class EmailAlertsService:
    def _get_jurisdiction_name(self, db: Any, jurisdiction_id: int) -> str:
        res = (
            db.table("jurisdictions")
            .select("name")
            .eq("id", int(jurisdiction_id))
            .limit(1)
            .execute()
        )
        if res.data and res.data[0].get("name"):
            return str(res.data[0]["name"])
        return "your selected jurisdiction"

    def _send_welcome_email_body(
        self, jurisdiction_name: str
    ) -> str:
        return f"""
Welcome to Intelligence Platform!

You have successfully subscribed to receive email alerts for {jurisdiction_name} housing regulation updates.

What to expect:
- You'll receive email notifications when regulations are updated for {jurisdiction_name}
- Updates include summaries and links to official sources
- Alerts are sent automatically when changes are detected

{settings.LEGAL_DISCLAIMER}

---
Intelligence Platform
""".strip()

    def send_welcome_email(self, email: str, jurisdiction_name: str) -> None:
        subject = f"Welcome to Housing Regulation Alerts for {jurisdiction_name}"
        body = self._send_welcome_email_body(jurisdiction_name=jurisdiction_name)
        _smtp_send(to_email=email, subject=subject, body=body)

    def subscribe(self, email: str, jurisdiction_id: int) -> dict[str, Any]:
        db = get_db()
        jurisdiction_id = int(jurisdiction_id)
        jurisdiction_name = self._get_jurisdiction_name(db, jurisdiction_id)

        try:
            db.table("email_subscriptions").upsert(
                [
                    {
                        "email": email,
                        "jurisdiction_id": jurisdiction_id,
                        "is_active": True,
                    }
                ],
                on_conflict="email,jurisdiction_id",
                ignore_duplicates=True,
            ).execute()
        except Exception as exc:
            msg = str(exc)
            if "permission denied" in msg.lower() or "42501" in msg:
                raise PermissionError(
                    "Database permission denied for email_subscriptions. "
                    "Run the RLS policy SQL from LOCAL_DEVELOPMENT.md step 6, "
                    "or switch SUPABASE_KEY to the service_role key."
                ) from exc
            raise

        try:
            self.send_welcome_email(email=email, jurisdiction_name=jurisdiction_name)
        except Exception:
            pass

        return {"status": "subscribed"}

    def unsubscribe(self, email: str, jurisdiction_id: int) -> dict[str, Any]:
        db = get_db()
        try:
            db.table("email_subscriptions").update({"is_active": False}).eq(
                "email", email
            ).eq("jurisdiction_id", int(jurisdiction_id)).execute()
        except Exception as exc:
            msg = str(exc)
            if "permission denied" in msg.lower() or "42501" in msg:
                raise PermissionError(
                    "Database permission denied for email_subscriptions. "
                    "Run the RLS policy SQL from LOCAL_DEVELOPMENT.md step 6, "
                    "or switch SUPABASE_KEY to the service_role key."
                ) from exc
            raise
        return {"status": "unsubscribed"}

    def get_active_subscriptions(self, email: str) -> list[dict[str, Any]]:
        db = get_db()
        try:
            res = (
                db.table("email_subscriptions")
                .select("jurisdiction_id,is_active,subscribed_at")
                .eq("email", email)
                .eq("is_active", True)
                .execute()
            )
        except Exception as exc:
            msg = str(exc)
            if "permission denied" in msg.lower() or "42501" in msg:
                raise PermissionError(
                    "Database permission denied for email_subscriptions. "
                    "Run the RLS policy SQL from LOCAL_DEVELOPMENT.md step 6, "
                    "or switch SUPABASE_KEY to the service_role key."
                ) from exc
            raise
        subs = res.data or []
        jurisdiction_ids = [int(r["jurisdiction_id"]) for r in subs]
        if not jurisdiction_ids:
            return []

        juris_res = (
            db.table("jurisdictions")
            .select("id,name")
            .in_("id", jurisdiction_ids)
            .execute()
        )
        juris_by_id = {int(r["id"]): r for r in (juris_res.data or [])}

        out: list[dict[str, Any]] = []
        for jid in jurisdiction_ids:
            row = juris_by_id.get(jid)
            if row:
                out.append(
                    {"jurisdiction_id": jid, "jurisdiction_name": row.get("name")}
                )
        return out

    def notify_subscribers(self, update: UpdateResult) -> None:
        db = get_db()
        if not isinstance(update, UpdateResult):
            update = UpdateResult.model_validate(update)

        affected_ids = [int(x) for x in (update.affected_jurisdictions or [])]
        if not affected_ids:
            return

        # Fetch jurisdiction names in one round-trip.
        juris_res = (
            db.table("jurisdictions")
            .select("id,name")
            .in_("id", affected_ids)
            .execute()
        )
        juris_by_id = {int(r["id"]): str(r.get("name") or "") for r in (juris_res.data or [])}

        subs_res = (
            db.table("email_subscriptions")
            .select("email,jurisdiction_id")
            .in_("jurisdiction_id", affected_ids)
            .eq("is_active", True)
            .execute()
        )
        subs = subs_res.data or []

        subs_by_jid: dict[int, list[str]] = {}
        for s in subs:
            jid = int(s["jurisdiction_id"])
            subs_by_jid.setdefault(jid, []).append(str(s["email"]))

        subject = f"Housing Regulation Update: {update.source_name}"
        detected_date = _format_detected_at(update.detected_at)

        for jid, emails in subs_by_jid.items():
            jurisdiction_name = juris_by_id.get(jid) or "your selected jurisdiction"

            # Adapted from the old email formatting, with city replaced by jurisdiction_name.
            body = f"""
================================================================================

HOUSING REGULATION UPDATE ALERT

Real Estate Platform

================================================================================


CITY: {jurisdiction_name}

REGULATION: {update.source_name}

CATEGORY: {update.category or 'N/A'}

DATE: {detected_date}



================================================================================

WHAT CHANGED?

================================================================================

{update.update_summary or ''}



================================================================================

WHO IS IMPACTED?

================================================================================


This regulation update affects Leasing Managers and Property Managers in {jurisdiction_name}.

You need to be aware of these changes as they impact:

- Lease agreement compliance requirements
- Tenant relations and policies
- Property management procedures
- Legal obligations and disclosures


================================================================================

WHAT ACTION SHOULD YOU TAKE?

================================================================================

1. Review the full regulation text using the source link below
2. Update lease documents and property management procedures accordingly
3. Train your staff on the new requirements
4. Ensure all new leases comply with the updated regulation
5. Consult with legal counsel if you have questions


================================================================================

SOURCE INFORMATION

================================================================================

Official Source: {update.source_name}

Category: {update.category or 'N/A'}

Direct Link: {update.url or 'N/A'}

Please review the official source document for complete details and legal text.


================================================================================

⚠️ LEGAL DISCLAIMER: {settings.LEGAL_DISCLAIMER}

---
Intelligence Platform
""".strip()

            for email in emails:
                _smtp_send(to_email=email, subject=subject, body=body)

    def send_daily_digest(self, jurisdiction_id: int) -> None:
        db = get_db()
        jurisdiction_id = int(jurisdiction_id)

        jurisdiction_name = self._get_jurisdiction_name(db, jurisdiction_id)
        since = datetime.utcnow() - timedelta(hours=24)

        updates_res = (
            db.table("regulation_updates")
            .select("regulation_id,update_summary,detected_at")
            .gte("detected_at", since.isoformat())
            .order("detected_at", desc=True)
            .limit(1000)
            .execute()
        )
        updates = updates_res.data or []
        if not updates:
            # Still send an email stating there were no updates.
            subs_res = (
                db.table("email_subscriptions")
                .select("email")
                .eq("jurisdiction_id", jurisdiction_id)
                .eq("is_active", True)
                .execute()
            )
            emails = [str(r["email"]) for r in (subs_res.data or [])]
            if not emails:
                return
            subject = f"Daily Summary: {jurisdiction_name} Housing Regulations - No Updates"
            body = f"""
Daily Housing Regulation Summary for {jurisdiction_name}

Date: {datetime.utcnow().strftime('%Y-%m-%d')}

No new regulation updates were detected for {jurisdiction_name} in the last 24 hours.

All regulations are up to date.

{settings.LEGAL_DISCLAIMER}

---
Intelligence Platform
Daily Summary Report
""".strip()
            for email in emails:
                _smtp_send(to_email=email, subject=subject, body=body)
            return

        regulation_ids = sorted({int(u["regulation_id"]) for u in updates if u.get("regulation_id") is not None})

        regs_res = (
            db.table("regulations")
            .select("id,source_name,url,category,jurisdiction_id")
            .in_("id", regulation_ids)
            .execute()
        )
        regs_by_id = {int(r["id"]): r for r in (regs_res.data or [])}

        matched_updates: list[dict[str, Any]] = []
        for u in updates:
            rid = int(u["regulation_id"])
            reg = regs_by_id.get(rid)
            if not reg:
                continue
            if int(reg.get("jurisdiction_id") or 0) != jurisdiction_id:
                continue
            matched_updates.append(
                {
                    "source_name": reg.get("source_name") or "",
                    "category": reg.get("category") or "",
                    "url": reg.get("url") or "",
                    "detected_at": u.get("detected_at"),
                    "update_summary": u.get("update_summary") or "",
                }
            )

        subs_res = (
            db.table("email_subscriptions")
            .select("email")
            .eq("jurisdiction_id", jurisdiction_id)
            .eq("is_active", True)
            .execute()
        )
        emails = [str(r["email"]) for r in (subs_res.data or [])]
        if not emails:
            return

        subject = (
            f"Daily Summary: {jurisdiction_name} Housing Regulations - {len(matched_updates)} Update(s)"
        )
        body_parts: list[str] = []
        body_parts.append(
            f"Daily Housing Regulation Summary for {jurisdiction_name}\n"
            f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            f"📊 Summary: {len(matched_updates)} regulation update(s) detected in the last 24 hours.\n\n"
            f"{'=' * 60}"
        )

        for idx, u in enumerate(matched_updates, 1):
            body_parts.append(
                f"\n{idx}. {u['source_name']}\n"
                f"   Category: {u['category']}\n"
                f"   Detected: {_format_detected_at(u.get('detected_at'))}\n\n"
                f"   Summary:\n{u['update_summary']}\n\n"
                f"   URL: {u['url']}\n\n"
                f"   {'-' * 60}"
            )

        body_parts.append(
            f"\n{settings.LEGAL_DISCLAIMER}\n\n---\nIntelligence Platform\nDaily Summary Report"
        )
        body = "".join(body_parts).strip()

        for email in emails:
            _smtp_send(to_email=email, subject=subject, body=body)


email_alerts = EmailAlertsService()

