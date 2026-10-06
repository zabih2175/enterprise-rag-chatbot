"""Google OAuth (web flow) + Gmail/Calendar sync into the vector store."""
import base64
from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

SCOPES = [
    "[googleapis.com](https://www.googleapis.com/auth/gmail.readonly)",
    "[googleapis.com](https://www.googleapis.com/auth/calendar.readonly)",
]


# ---------- OAuth ----------
def make_flow(client_id: str, client_secret: str, redirect_uri: str) -> Flow:
    config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "[accounts.google.com](https://accounts.google.com/o/oauth2/auth)",
            "token_uri": "[oauth2.googleapis.com](https://oauth2.googleapis.com/token)",
            "redirect_uris": [redirect_uri],
        }
    }
    # PKCE is off because the page reloads after the redirect and a code
    # verifier could not be carried over.
    return Flow.from_client_config(
        config,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        autogenerate_code_verifier=False,
    )


def encode_state(client_name: str) -> str:
    return base64.urlsafe_b64encode(client_name.encode()).decode()


def decode_state(state: str) -> str:
    try:
        return base64.urlsafe_b64decode(state.encode()).decode()
    except Exception:
        return ""


def get_auth_url(flow: Flow, client_name: str) -> str:
    url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent select_account",  # always show the account picker
        include_granted_scopes="true",
        state=encode_state(client_name),
    )
    return url


def exchange_code(flow: Flow, code: str) -> dict:
    flow.fetch_token(code=code)
    return {
        "token": flow.credentials.token,
        "refresh_token": flow.credentials.refresh_token,
        "token_uri": flow.credentials.token_uri,
        "client_id": flow.credentials.client_id,
        "client_secret": flow.credentials.client_secret,
        "scopes": list(flow.credentials.scopes or SCOPES),
    }


def _service(creds_info: dict, api: str, version: str):
    creds = Credentials.from_authorized_user_info(creds_info, SCOPES)
    return build(api, version, credentials=creds, cache_discovery=False)


def get_account_email(creds_info: dict) -> str:
    profile = _service(creds_info, "gmail", "v1").users().getProfile(userId="me").execute()
    return profile.get("emailAddress", "")


# ---------- Sync ----------
def sync_gmail(vector_store, creds_info: dict, max_results: int = 5) -> int:
    service = _service(creds_info, "gmail", "v1")
    listing = service.users().messages().list(userId="me", maxResults=max_results).execute()

    texts, metas = [], []
    for item in listing.get("messages", []):
        msg = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=item["id"],
                format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        subject = headers.get("Subject", "No Subject")
        texts.append(
            f"Email Subject: {subject}\n"
            f"Sender: {headers.get('From', 'Unknown Sender')}\n"
            f"Date: {headers.get('Date', 'Unknown Date')}\n\n"
            f"Snippet:\n{msg.get('snippet', '')}"
        )
        metas.append({"source": "gmail", "subject": subject, "message_id": item["id"]})

    return vector_store.add_texts(texts, metas)


def sync_calendar(vector_store, creds_info: dict, max_results: int = 10) -> int:
    service = _service(creds_info, "calendar", "v3")
    now = datetime.now(timezone.utc).isoformat()
    events = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
        .get("items", [])
    )

    texts, metas = [], []
    for ev in events:
        start = ev["start"].get("dateTime", ev["start"].get("date"))
        summary = ev.get("summary", "Untitled Event")
        texts.append(
            f"Calendar Event: {summary}\n"
            f"Date/Time: {start}\n"
            f"Description: {ev.get('description', 'No description provided.')}"
        )
        metas.append({"source": "calendar", "event": summary, "event_id": ev.get("id")})

    return vector_store.add_texts(texts, metas)
