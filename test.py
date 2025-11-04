from fastapi import FastAPI, UploadFile, Form, BackgroundTasks, Query, HTTPException, Path
import os
import uuid
import time
import json
import pdfplumber
import boto3
import logging
import threading
import asyncpg
from datetime import date
logging.getLogger("pdfminer").setLevel(logging.ERROR)
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, EmailStr
import psycopg2
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import LLMChain
from langchain_community.llms import Bedrock
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chromadb
from email_validator import validate_email, EmailNotValidError
from collections import defaultdict, deque
import re
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import re
from datetime import datetime, timedelta
from collections import defaultdict
from langchain.memory import ConversationBufferMemory
from fastapi import Depends, Header
import bcrypt
import glob
from langchain_community.llms import Bedrock

bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")
app = FastAPI(title="Candidate Interview API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
persist_directory = "./data/chroma"
os.makedirs(persist_directory, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=persist_directory)
collection_name = "pdf_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)
AWS_REGION = "ap-south-1"
AGENT_ID = "TN1M6VJT6Z"
AGENT_ALIAS_ID = "LKST9LBZDS"
bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")
bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name="ap-south-1")

bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1",
    )

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'nibav_chatbot',
    'user': 'vasala',
    'password': 'newpassword123'
    }

db_pool = None
async def get_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(**DB_CONFIG)
    return db_pool

session_state = {}
SESSION_TIMEOUT_MINUTES = 300

async def is_session_active(email: str):
    session = session_state.get(email)
    if not session:
        return False
    last_activity = session.get("last_activity")
    if not last_activity:
        return False
    return datetime.now() - last_activity < timedelta(minutes=SESSION_TIMEOUT_MINUTES)

@app.post("/admin/login")
async def admin_login(email: str, password: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Check admin first
        result = await conn.fetchrow(
            "SELECT role, name FROM admin WHERE email = $1 AND password = $2",
            email, password
        )

        # Check sales if not found
        if not result:
            result = await conn.fetchrow(
                "SELECT role, name FROM admin WHERE email = $1 AND password = $2",
                email, password
            )

    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    role, name = result["role"], result["name"]

    # Create or update session for this user
    session_state[email] = {
        "authenticated": True,
        "role": role,
        "name": name,
        "email": email,
        "last_activity": datetime.now()
    }

    return {
        "success": True,
        "message": f"{role.capitalize()} authenticated!",
        "role": role.lower()
    }

@app.get("/sessions/active")
async def get_active_sessions():
    active_admins = []
    active_sales = []

    for email, session in session_state.items():
        if await is_session_active(email):  # ensure not expired
            role = session.get("role", "").lower()
            if role == "admin":
                active_admins.append(email)
            elif role == "sales":
                active_sales.append(email)

    return {
        "total_active": len(active_admins) + len(active_sales),
        "admins_count": len(active_admins),
        "sales_count": len(active_sales),
        "admins": active_admins,
        "sales": active_sales
    }

async def require_admin(session_email: str = Query(...)):
    """Check if the given user is logged in and has admin privileges."""
    session = session_state.get(session_email)

    if not session:
        raise HTTPException(status_code=401, detail="User not logged in")

    if not await is_session_active(session_email):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

    if session.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="Access denied: Admins only")

    return session

async def require_admin_or_sales(session_email: str = Query(...)):
    """Check if the given user is logged in and has admin or sales privileges."""
    session = session_state.get(session_email)

    if not session:
        raise HTTPException(status_code=401, detail="User not logged in")

    if not await is_session_active(session_email):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

    role = session.get("role", "").lower()
    if role not in ["admin", "sales"]:
        raise HTTPException(status_code=403, detail="Access denied: Only Admin or Sales allowed")

    return session


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"PDF extraction error for {file_path}: {e}")
        return ""

def clean_text(text: str):
    return " ".join(text.replace("\n", " ").replace("\t", " ").split())


def chunk_text(text: str, chunk_size=400, chunk_overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

async def embed_and_store(chunks, filename):
    # Create LangChain Document objects for each chunk
    docs = [
        Document(page_content=chunk, metadata={"filename": filename, "chunk": idx})
        for idx, chunk in enumerate(chunks)
    ]

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=bedrock_embeddings
    )
    vectorstore.add_documents(docs)
    print(f" Stored {len(chunks)} chunks for {filename}")

async def process_pdf(file_path, filename):
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        print(f"No text to process for {filename}")
        return
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)
    # chunks = chunk_text(text)
    print(f"Total chunks to embed: {len(chunks)}")
    await embed_and_store(chunks, filename)

@app.post("/upload")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks, session=Depends(require_admin) ):
    os.makedirs("./data/pdfs", exist_ok=True)
    # safe_filename = f"{int(time.time())}_{file.filename}"
    os.makedirs("./data/pdfs", exist_ok=True)
    allowed_extensions = [".pdf"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Sorry, this document type ({ext}) is not supported. Please upload a PDF file.")

    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join("./data/pdfs", safe_filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    background_tasks.add_task(process_pdf, file_path, safe_filename)

    return {
        "message": "PDF uploaded successfully. Embedding will be processed in the background.",
        "filename": safe_filename,
        "pdf_path": file_path
    }




@app.get("/verify_embeddings")
def verify_embeddings(session=Depends(require_admin_or_sales)):
    """
    Verify whether embeddings exist in the Chroma collection
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)

    results = collection.get(include=["documents", "metadatas", "embeddings"])

    total_docs = len(results["documents"])
    embeddings_exist = False

    for i in range(total_docs):
        emb = results["embeddings"][i]
        if emb is not None and len(emb) > 0:
            embeddings_exist = True
            break

    return {
        "total_documents": total_docs,
        "embeddings_stored": embeddings_exist
    }

@app.get("/inspect_chunks")
def inspect_chunks(session=Depends(require_admin_or_sales)):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    results = collection.get(include=["documents", "metadatas", "embeddings"])

    table = []
    for i in range(len(results["documents"])):
        doc_text = results["documents"][i]
        metadata = results["metadatas"][i]

        # embeddings might be numpy arrays or None
        emb = results["embeddings"][i]
        emb_preview = emb[:10].tolist() if emb is not None and len(emb) > 0 else []

        table.append({
            "filename": metadata.get("filename", "unknown"),
            "chunk_index": metadata.get("chunk", i),
            "text_snippet": " ".join(doc_text.split()[:150]),
            "embedding_preview": emb_preview
        })

    return {
        "total_chunks": len(results["documents"]),
        "chunks": table
    }

@app.get("/list_pdfs")
async def list_pdfs(user_session=Depends(require_admin_or_sales)):
    """
    List all uploaded PDF files in ./data/pdfs.
    Only users with role 'admin' or 'sales' can access this.
    """
    pdf_dir = "./data/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    pdf_list = [os.path.basename(pdf) for pdf in pdf_files]

    return {
        "total_pdfs": len(pdf_list),
        "pdf_files": pdf_list,
        "accessed_by": user_session["email"],
        "role": user_session["role"]
    }

@app.delete("/delete_pdf/{filename}")
def delete_pdf(filename: str,require_admin=Depends(require_admin)):
    """
    Delete all chunks and embeddings related to a specific PDF file permanently.
    """
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Fetch all metadata and IDs (IDs are always included, even without specifying in include)
    results = collection.get(include=["metadatas"])  # Removed "ids" from include
    all_metadatas = results.get("metadatas", [])
    all_ids = results.get("ids", [])  # IDs are still here

    # Filter IDs for this filename
    ids_to_delete = [
        str(all_ids[idx])  # Ensure it's a string
        for idx, meta in enumerate(all_metadatas)
        if meta.get("filename") == filename
    ]

    if not ids_to_delete:
        return {"message": f"No chunks found for filename '{filename}'."}

    # Delete embeddings from Chroma using string IDs
    collection.delete(ids=ids_to_delete)

    # Delete the actual PDF file
    pdf_path = os.path.join("./data/pdfs", filename)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    return {
        "message": f"Deleted PDF '{filename}' and {len(ids_to_delete)} associated chunks permanently."
    }

@app.post("/create_user")
async def create_user(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    admin_session=Depends(require_admin)
    ):
    """
    Create a new Admin or Sales user (form-based).
    Only admins can access this route.
    """
    pool = await get_pool()

    if role.lower() not in ["admin", "sales"]:
        raise HTTPException(status_code=400, detail="Role must be either 'admin' or 'sales'.")


    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT email FROM admin WHERE email = $1", email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already exists.")

        await conn.execute("""
            INSERT INTO admin (name, email, password, role, created_by, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, name, email, password, role.lower(),
           admin_session["email"], datetime.now())

    return {
        "success": True,
        "message": f"{role.capitalize()} user created successfully.",
        "created_by": admin_session["email"]
    }

@app.delete("/delete_sales_user")
async def delete_sales_user(
    email: str = Form(...),
    admin_session=Depends(require_admin)
    ):
    """
    Delete a Sales user by email.
    Only admins can access this route.
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM admin WHERE email = $1", email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        if user["role"].lower() != "sales":
            raise HTTPException(status_code=403, detail="Only Sales users can be deleted.")

        # Perform delete
        await conn.execute("DELETE FROM admin WHERE email = $1", email)

    return {
        "success": True,
        "message": f"Sales user '{email}' deleted successfully by {admin_session['email']}."
    }

@app.post("/logout")
async def logout_user(email: str = Form(...)):


    """
    Logout a user by email — remove from session_state and update DB session if needed.
    """
    # Check if user has an active session
    session = session_state.get(email)
    if not session:
        raise HTTPException(status_code=400, detail="No active session found for this email.")

    # Remove from session_state
    del session_state[email]



    return {
        "success": True,
        "message": f"User '{email}' has been logged out successfully.",
        "logout_time": datetime.now().isoformat()
    }

@app.get("/unique_emails")
async def get_unique_emails(session=Depends(require_admin_or_sales)):
    """
    Return unique emails from 'users' table with their count,
    latest logged date, status, and description.
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT
                DISTINCT ON (email)
                email,
                status,
                description,
                date AS last_logged_date,
                COUNT(*) OVER (PARTITION BY email) AS total_entries
            FROM users
            ORDER BY email, last_logged_date DESC;
        """)

    unique_emails = [
        {
            "email": record["email"],
            "status": record["status"],
            "description": record["description"],
            "logged_date": record["last_logged_date"].isoformat() if record["last_logged_date"] else None,
            "total_entries": record["total_entries"]
        }
        for record in results
    ]

    return {
        "total_unique_emails": len(unique_emails),
        "data": unique_emails
    }

async def user_exists(email: str) -> bool:
    pool = await get_pool()
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            "SELECT 1 FROM users WHERE email=$1",
            email
        )
        return result is not None

async def save_user_info(email: str, phone: str, country: str, status: str):
    pool = await get_pool()
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users (email, phone, country, status) VALUES ($1, $2, $3, $4)",
            email, phone, country, status
        )

@app.post("/user/register")
async def user_register(
    email: str = Form(...),
    phone: str = Form(...),
    country: str = Form('India'),
    status: str = Form('pending')
    ):
    # --- Phone Validation ---
    phone = phone.strip()
    if not re.fullmatch(r"\d{10}", phone):
        return {"success": False, "message": "Phone number must be exactly 10 digits."}
    if phone[0] not in ('6', '7', '8', '9'):
        return {"success": False, "message": "Phone number must start with 6, 7, 8, or 9."}

    # --- Email Validation ---
    email = email.strip()
    if '@' not in email or '.' not in email:
        return {"success": False, "message": "Email is not valid, please provide valid email address."}
    try:
        validate_email(email)
    except EmailNotValidError:
        return {"success": False, "message": "Invalid email address."}

    # --- Check if user exists ---
    if await user_exists(email):
        return {"success": False, "message": "User with this email or phone already exists."}

    # --- Save to DB ---
    await save_user_info(email, phone, country, status)

    return {"success": True, "message": "User registered successfully."}

@app.get("/get_chat_history/{target_email}")
async def get_chat_history(

    target_email: str ,
    session=Depends(require_admin_or_sales)
    # session_email: str = Query(..., description="Logged-in admin/sales email"),

    ):
    """
    Fetch all chat interactions for a specific candidate email.
    Only Admin or Sales (authenticated via session_email) can access this.
    Results are ordered by 'chat_date' DESC (latest first).
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT id, email, user_query, bot_response, chat_date, chat_time
            FROM chat_history
            WHERE email = $1
            ORDER BY chat_time DESC;
        """, target_email)

    if not results:
        raise HTTPException(status_code=404, detail=f"No chat history found for email: {target_email}")

    chats = [
        {
            "email": record["email"],
            "user_query": record["user_query"],
            "bot_response": record["bot_response"],
            "chat_date": record["chat_date"].isoformat() if record["chat_date"] else None,
        }
        for record in results
    ]

    return {
        "role": session.get("role"),
        "target_email": target_email,
        "total_interactions": len(chats),
        "chats": chats
    }

@app.delete("/delete_user/{email}")
async def delete_user_by_email(
    email: str,
    session=Depends(require_admin)
    ):
    """
    Permanently delete a user and all their related chat interactions.
    Accessible only by Admin users.
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1️⃣ Delete user record
            user_result = await conn.execute(
                "DELETE FROM users WHERE email = $1;", email
            )

            # 2️⃣ Delete chat history records
            chat_result = await conn.execute(
                "DELETE FROM chat_history WHERE email = $1;", email
            )

            # Get how many rows were deleted
            user_deleted = "DELETE 0" not in user_result
            chat_deleted_count = int(chat_result.split(" ")[-1])

    if not user_deleted:
        raise HTTPException(status_code=404, detail=f"No user found with email: {email}")

    return {
        "message": f"User '{email}' and related chats deleted successfully.",
        "deleted_user": user_deleted,
        "deleted_chat_records": chat_deleted_count
    }

@app.get("/get_today_interactions")
async def get_today_interactions(session=Depends(require_admin_or_sales)):
    """
    Get the total number of chat interactions that happened today.
    Accessible by both Admin and Sales users.

    Returns:
    - date: Today's date
    - total_interactions: Total count of chat messages today
    - role: Role of the user who accessed the API
    """

    today = date.today()
    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.fetchval("""
            SELECT COUNT(*) FROM chat_history
            WHERE chat_date = $1;
        """, today)

    return {
        "date": today.isoformat(),
        "role": session.get("role"),
        "total_interactions": result or 0
    }






@app.get("/get_daily_chat_counts")
async def get_daily_chat_counts(session=Depends(require_admin_or_sales)):
    """
    Get the total number of chat interactions for each day.
    Groups the chat_history table by chat_date and counts messages.
    Ordered by date ascending.
    Accessible by both Admin and Sales users.
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                chat_date AS date,
                COUNT(*) AS total_interactions
            FROM chat_history
            GROUP BY chat_date
            ORDER BY chat_date ASC;
        """)

    return {
        "role": session.get("role"),
        "daily_interactions": [
            {
                "date": row["date"].isoformat(),
                "total_interactions": row["total_interactions"]
            }
            for row in rows
        ]
    }
















@app.get("/get_user_summary")
async def get_user_summary(session=Depends(require_admin_or_sales)):
    """
    Fetch all users with key details:
    - email, phone_number, date, summary, status, description, updated_by
    Ordered by date (descending).
    Accessible by Admin and Sales.
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                email,
                phone,
                date,
                summary,
                status,
                description,
                updated_by
            FROM users
            ORDER BY date DESC;
        """)

    # Convert records to a list of dicts
    users_list = [dict(row) for row in rows]

    return {
        "date": date.today().isoformat(),
        "role": session.get("role"),
        "total_users": len(users_list),
        "users": users_list
    }

@app.get("/get_users_by_date")
async def get_users_by_date(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD"),
    session=Depends(require_admin_or_sales)
    ):
    """
    Fetch users between a custom date range.
    Returns user details:
    - email, phone, date, summary, status, description, updated_by
    Ordered by date (descending).
    Accessible by Admin and Sales.
    """

    # Convert query strings to date objects
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                email,
                phone,
                date,
                summary,
                status,
                description,
                updated_by
            FROM users
            WHERE date BETWEEN $1 AND $2
            ORDER BY date DESC;
        """, start, end)

    return {
        "role": session.get("role"),
        "from_date": start.isoformat(),
        "to_date": end.isoformat(),
        "total_users": len(rows),
        "users": [dict(row) for row in rows]
    }









@app.get("/get_admin_summary")
async def get_admin_summary(session=Depends(require_admin_or_sales)):
    """
    Fetch all admins with key details:
    - name, email, role, created_by
    Ordered by created_by (descending).
    Accessible by Admin and Sales.
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                name,
                email,
                role,
                created_by
            FROM admin
            ORDER BY created_by DESC;
        """)

    # Convert to list of dicts
    admin_list = [dict(row) for row in rows]

    return {
        "date": date.today().isoformat(),
        "role": session.get("role"),
        "total_admins": len(admin_list),
        "admins": admin_list
    }







@app.get("/get_total_chats")
async def get_total_chats(session=Depends(require_admin_or_sales)):
    """
    Get the overall total number of chat messages from the chat_history table.
    Accessible by Admin and Sales users.

    Returns:
    - date: Today's date
    - total_chats: Total number of chat messages across all time
    - role: Role of the user who accessed the API
    """

    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.fetchval("""
            SELECT COUNT(*) FROM chat_history;
        """)

    return {
        "date": date.today().isoformat(),
        "role": session.get("role"),
        "total_chats": result or 0
    }

























@app.get("/get_interactions_by_date")
async def get_interactions_by_date(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD"),
    session=Depends(require_admin_or_sales)
    ):
    """
    Get total chat interactions between a custom date range.
    Accessible by both Admin and Sales.
    """

    # Parse string dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM chat_history
            WHERE chat_date BETWEEN $1 AND $2;
        """, start, end)

    return {
        "role": session.get("role"),
        "from_date": start.isoformat(),
        "to_date": end.isoformat(),
        "total_interactions": count or 0
    }

@app.put("/update_user_status")
async def update_user_status(
    email: str = Form(..., description="Email of the user to update"),
    status: str = Form(..., description="New status: pending / in progress / complete"),
    session=Depends(require_admin_or_sales)
    ):
    """
    Update a user's status, description, and updated_by fields.
    Accessible by both Admin and Sales.
    """

    # ✅ Only these statuses are allowed
    allowed_statuses = ["pending", "in progress", "complete"]
    if status.lower() not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(allowed_statuses)}"
        )

    # ✅ Get the logged-in admin/sales email
    updater_email = session.get("email")

    if not updater_email:
        raise HTTPException(status_code=401, detail="Invalid session: no user email found")

    pool = await get_pool()

    async with pool.acquire() as conn:
        async with conn.transaction():
            result = await conn.execute("""
                UPDATE users
                SET status = $1,
                    updated_by = $2
                WHERE email = $3;
            """, status.lower(), updater_email, email)

    if "UPDATE 0" in result:
        raise HTTPException(status_code=404, detail=f"No user found with email: {email}")

    return {
        "message": f"User '{email}' status updated successfully.",
        "updated_by": updater_email,
        "new_status": status.lower(),
        "role": session.get("role")
    }








recent_queries = defaultdict(lambda: deque(maxlen=5))  # Store last few queries per user

async def get_user_intent(query_list):
    all_text = " ".join(query_list).lower()
    if any(word in all_text for word in ["price", "cost", "buy", "purchase", "quotation", "install", "book", "order","installation", "quote"]):
        return "Interested in buying Nibav lift"
    elif any(word in all_text for word in ["service", "maintenance", "repair", "support"]):
        return "Existing customer - service enquiry"
    elif any(word in all_text for word in ["not interested", "no need", "later", "cancel","stop", "don't want"]):
        return "Not interested currently"
    elif any(word in all_text for word in ["features", "details", "info", "about", "benefits"]):
        return "Exploring product information"
    else:
        return "General enquiry"

async def get_user_summary(query_list):
    last_query = query_list[-1] if query_list else ""
    return f"User recently asked about: {last_query[:120]}"





async def update_user_info(email, description=None, summary=None):
    """Update user's description and summary in PostgreSQL"""
    pool = await get_pool()  # use your asyncpg pool
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE users
            SET description = $1, summary = $2, updated_at = NOW()
            WHERE email = $3
            """,
            description, summary, email
        )
    print(f"✅ Updated users table for {email}: desc='{description}', summary='{summary}'")


async def process_user_query(email, user_query):
    """Process each new query and update DB every 4 queries"""
    recent_queries[email].append(user_query)
    qlen = len(recent_queries[email])

    # Once user sends 4 messages, detect intent and reset their history
    if qlen == 4:
        desc = get_user_intent(recent_queries[email])
        summ = get_user_summary(recent_queries[email])

        await update_user_info(email, description=desc, summary=summ)

        # Clear deque for that user after update
        recent_queries[email].clear()
        print(f"[INFO] Intent captured and cleared for {email}")



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)




async def save_chat(email: str, user_message: str, bot_response: str):
    now = datetime.now()
    chat_date = now.date()
    chat_time = now.time()

    pool = await get_pool()  # ✅ use your global async pool
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chat_history (email, user_query, bot_response, timestamp, chat_date, chat_time)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            email, user_message, bot_response, now, chat_date, chat_time
        )



def invoke_nova_agent(email: str, user_query: str, context: str):
    final_prompt = f"""
    You are an official document-based assistant.
    Answer ONLY using the information provided in the PDF content.
    Do NOT say phrases like "according to the PDF", "based on the provided document", or similar.
    Respond directly and concisely with the answer.
    If the PDFs do not contain relevant details, respond exactly with:
    "I couldn’t find that information, please ask another question."

    User Question: {user_query}
    Context from PDFs:
    {context}
    """

    #  unique Bedrock session for each user (maintains memory)
    session_id = email.replace("@", "_").replace(".", "_")

    response = bedrock_agent_client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=final_prompt
    )

    # Extract streamed text
    answer_text = ""
    for event in response["completion"]:
        if "chunk" in event:
            answer_text += event["chunk"]["bytes"].decode("utf-8")

    return answer_text.strip()

    unwanted_phrases = [
        "according to the pdf",
        "based on the pdf",
        "based on the provided document",
        "according to provided document",
        "from the pdf content",
        "based on the uploaded file",
    ]

    for phrase in unwanted_phrases:
        if clean_answer.lower().startswith(phrase):
            clean_answer = clean_answer[len(phrase):].lstrip(" ,:.-")

    return clean_answer.strip()


user_memories = defaultdict(lambda: ConversationBufferMemory(memory_key="chat_history", return_messages=True))

user_last_active = {}


def cleanup_inactive_users():
    while True:
        now = datetime.now()
        inactive_users = []
        for email, last_seen in list(user_last_active.items()):
            # If user inactive for more than 2 hours → cleanup
            if now - last_seen > timedelta(minutes=30):
                user_memories.pop(email, None)
                user_last_active.pop(email, None)
                inactive_users.append(email)
        if inactive_users:
            print(f"🧹 Cleaned inactive users: {inactive_users}")
        time.sleep(600)

threading.Thread(target=cleanup_inactive_users, daemon=True).start()

@app.post("/chat")
async def chat_with_nibav(email: str = Form(...), user_query: str = Form(...)):
    # Basic greetings
    if user_query.lower().strip() in ["hi", "hello", "hey", "good morning", "good evening"]:
        return {"query": user_query, "answer": "Hi! How can I help you today?", "sources": []}

    memory = user_memories[email]
    user_last_active[email] = datetime.now()

    # --- Step 1: Retrieve relevant context from PDFs ---
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=bedrock_embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # retrieved_docs = retriever.get_relevant_documents(user_query)
    # query_with_context = f"{user_query} about Nibav lift"
    retrieved_docs = retriever.get_relevant_documents(user_query)
    if not retrieved_docs:
        bot_response = "I couldn’t find that information, please ask another question."
        await save_chat(email, user_query, bot_response)
        return {"query": user_query, "answer": bot_response, "sources": []}

    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    sources = list(set([doc.metadata.get("filename", "Unknown") for doc in retrieved_docs]))

    # --- Step 2: Ask Nova Agent using retrieved context ---
    bot_response = invoke_nova_agent(email, user_query, context_text)

    # --- Step 3: Save chat memory & DB history ---
    memory.save_context({"query": user_query}, {"result": bot_response})
    await save_chat(email, user_query, bot_response)

    recent_queries[email].append(user_query)
    qlen = len(recent_queries[email])



    if qlen % 4 == 0:
        desc = await get_user_intent(recent_queries[email])
        summ = await get_user_summary(recent_queries[email])

        await update_user_info(email, description=desc, summary=summ)
        recent_queries[email].clear()
        print(f"[INFO] Cleared query history for {email} after 4 interactions")



    return {"query": user_query, "answer": bot_response, "sources": sources}














@app.get("/active_users")
def get_active_users():
    active_users = list(user_memories.keys())
    return {
        "active_user_count": len(active_users),
        "emails": active_users
    }

