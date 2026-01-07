from fastapi import FastAPI, UploadFile, Form, BackgroundTasks, Query, HTTPException
import os
import uuid
import time
import json
import pdfplumber
import boto3
import logging
import asyncpg
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
from collections import defaultdict
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
    'database': 'postgres',
    'user': 'postgres',
    'password': 'Vasala@#66118'
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

async def require_admin(email: str = Query(...)):
    """Check if the given user is logged in and has admin privileges."""
    session = session_state.get(email)

    if not session:
        raise HTTPException(status_code=401, detail="User not logged in")

    if not await is_session_active(email):
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

    if session.get("role", "").lower() != "admin":
        raise HTTPException(status_code=403, detail="Access denied: Admins only")

    return session


async def require_admin_or_sales(email: str = Query(...)):
    """Check if the given user is logged in and has admin or sales privileges."""
    session = session_state.get(email)

    if not session:
        raise HTTPException(status_code=401, detail="User not logged in")

    if not await is_session_active(email):
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

def chunk_text(text: str, chunk_size=300, chunk_overlap=100):
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
    
    # Use LangChain's Chroma vectorstore to add docs (this handles embeddings internally)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=bedrock_embeddings
    )
    vectorstore.add_documents(docs)
    print(f"✅ Stored {len(chunks)} chunks for {filename}")

async def process_pdf(file_path, filename):
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        print(f"No text to process for {filename}")
        return
    chunks = chunk_text(text)
    print(f"Total chunks to embed: {len(chunks)}")
    await embed_and_store(chunks, filename)

@app.post("/upload")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks, session=Depends(require_admin) ):
    os.makedirs("./data/pdfs", exist_ok=True)
    safe_filename = f"{int(time.time())}_{file.filename}"
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













async def user_exists(email: str, phone: str) -> bool:
    pool = await get_pool()
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            "SELECT 1 FROM users WHERE email=$1 OR phone=$2",
            email, phone
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
    if await user_exists(email, phone):
        return {"success": False, "message": "User with this email or phone already exists."}

    # --- Save to DB ---
    await save_user_info(email, phone, country, status)

    return {"success": True, "message": "User registered successfully."}


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
def save_chat(email: str, user_message: str, bot_response: str):
    now = datetime.now()
    chat_date = now.date()
    chat_time = now.time()
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (email, user_query, bot_response, timestamp, chat_date, chat_time) VALUES (%s, %s, %s, %s, %s, %s)",
        (email, user_message, bot_response, now, chat_date, chat_time)
    )
    conn.commit()
    cur.close()
    conn.close()

user_memories = defaultdict(lambda: ConversationBufferMemory(memory_key="chat_history", return_messages=True))


def invoke_nova_agent(user_query: str, context: str):
    # Combine your prompt with context
    final_prompt = f"""
    You are Nibav Lifts' official PDF assistant.
    Answer ONLY using the provided PDF content.
    If the PDFs do not contain relevant details, respond exactly with:
    "I couldn’t find that information, please ask another question."

    User Question: {user_query}
    Context from PDFs:
    {context}
    """

    response = bedrock_agent_client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId="nibav-session",
        inputText=final_prompt
    )

    # Extract text from the streaming response
    answer_text = ""
    for event in response["completion"]:
        if "chunk" in event:
            answer_text += event["chunk"]["bytes"].decode("utf-8")

    return answer_text.strip()


@app.post("/chat")
def chat_with_nibav(email: str = Form(...), user_query: str = Form(...)):
    # greetings check
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if user_query.lower().strip() in greetings:
        return {"query": user_query, "answer": "Hi! How can I help you today?", "sources": [], "top_chunks": []}

    memory = user_memories[email]

    # --- Retrieval ---
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=bedrock_embeddings
    )
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7})

    retrieved_docs = retriever.get_relevant_documents(user_query)

    if not retrieved_docs:
        bot_response = "I couldn’t find that information, please ask another question."
        memory.save_context({"query": user_query}, {"result": bot_response})
        save_chat(email, user_query, bot_response)
        return {"query": user_query, "answer": bot_response, "sources": [], "top_chunks": []}

    context_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # --- Use Nova Pro Agent ---
    bot_response = invoke_nova_agent(user_query, context_text)

    memory.save_context({"query": user_query}, {"result": bot_response})
    save_chat(email, user_query, bot_response)
    sources = list(set([doc.metadata.get("filename", "Unknown") for doc in retrieved_docs]))

    return {"query": user_query, "answer": bot_response, "sources": sources}






























