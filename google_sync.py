import os
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scopes required for reading Gmail and Calendar
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/calendar.readonly'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")

def get_google_service(api_name, version):
    """Handles OAuth2 authentication and saves token.json automatically using absolute paths."""
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(f"credentials.json not found at {CREDENTIALS_PATH}. Ensure Supabase/Google secrets are configured.")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
            
    return build(api_name, version, credentials=creds)

def sync_live_gmail_to_supabase(vector_store, openai_client, max_results=5):
    """Fetches recent emails and pushes them into Supabase vector store."""
    service = get_google_service('gmail', 'v1')
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    
    synced_count = 0
    for msg_info in messages:
        msg = service.users().messages().get(userId='me', id=msg_info['id']).execute()
        headers = msg['payload']['headers']
        
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
        snippet = msg.get('snippet', '')
        
        content = f"Email Subject: {subject}\nSender: {sender}\nDate: {date}\n\nSnippet/Body:\n{snippet}"
        
        # Generate embedding via OpenAI
        response = openai_client.embeddings.create(
            input=[content],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        
        # Insert into Supabase table matching your current tenant/workspace schema
        data = {
            "client_id": vector_store.client_id,
            "content": content,
            "metadata": {"source": "gmail_api", "subject": subject},
            "embedding": embedding
        }
        
        vector_store.client.table("client_knowledge_base").insert(data).execute()
        synced_count += 1
        
    return synced_count

def sync_live_calendar_to_supabase(vector_store, openai_client, max_results=10):
    """Fetches upcoming calendar events and pushes them into Supabase vector store."""
    service = get_google_service('calendar', 'v3')
    now = datetime.utcnow().isoformat() + 'Z'
    
    events_result = service.events().list(
        calendarId='primary', timeMin=now,
        maxResults=max_results, singleEvents=True,
        orderBy='startTime').execute()
    events = events_result.get('items', [])
    
    synced_count = 0
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        summary = event.get('summary', 'Untitled Event')
        description = event.get('description', 'No description provided.')
        
        content = f"Calendar Event: {summary}\nDate/Time: {start}\nDescription: {description}"
        
        response = openai_client.embeddings.create(
            input=[content],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        
        data = {
            "client_id": vector_store.client_id,
            "content": content,
            "metadata": {"source": "calendar_api", "event": summary},
            "embedding": embedding
        }
        
        vector_store.client.table("client_knowledge_base").insert(data).execute()
        synced_count += 1
        
    return synced_count