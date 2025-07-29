from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
import re
import json
from fuzzywuzzy import fuzz
import redis.asyncio as redis
import aiosmtplib
from email.message import EmailMessage
import os
import io
import datetime

# Import llama-cpp-python for local model inference
from llama_cpp import Llama

app = FastAPI()

from api_decision_tree import router as decision_tree_router

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(decision_tree_router, prefix="/api")

# Load and parse Q/A text file
def load_qa_blocks(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    lines = raw_text.strip().splitlines()
    documents = []
    current_qa = ""
    for line in lines:
        if line.startswith("Q:"):
            if current_qa:
                documents.append(current_qa.strip())
            current_qa = line + "\n"  # Start new block with current Q: line
        elif line.startswith("A:"):
            current_qa += line + "\n"
        else:
            current_qa += line + "\n"
    if current_qa:
        documents.append(current_qa.strip())
    return documents

# Initialize sentence-transformers model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = load_qa_blocks("base_qa_navitrends.txt")
vectors = model.encode(documents)
np_vectors = np.array(vectors).astype('float32')
dimension = np_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np_vectors)

# Initialize local Llama model for inference
llm = Llama(model_path="C:/Users/ybouk/Downloads/builder-mystic-den-main/yasmine/backend/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# Load decision tree JSON at startup
decision_tree_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "descion_tree.json"))
with open(decision_tree_path, "r", encoding="utf-8") as f:
    decision_tree = json.load(f)

# Flatten decision tree nodes for fuzzy matching
def flatten_decision_tree(tree):
    flat_list = []
    visited = set()
    def recurse(node_key):
        if node_key in visited:
            return
        visited.add(node_key)
        node = tree.get(node_key)
        if node:
            flat_list.append((node_key, node.get("message", "")))
            choices = node.get("choices", {})
            for next_key in choices.values():
                if isinstance(next_key, str):
                    recurse(next_key)
    recurse("start")
    return flat_list

flat_decision_nodes = flatten_decision_tree(decision_tree)

class QuestionRequest(BaseModel):
    question: str

class ContactForm(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    message: str
    conversation: str

redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)

async def send_email(contact: ContactForm):
    msg = EmailMessage()
    msg["From"] = "no-reply@navitrends.com"
    msg["To"] = "commercial@navitrends.com"
    msg["Subject"] = f"Nouveau contact de {contact.first_name} {contact.last_name}"

    body = f"""
    Vous avez reçu un nouveau message de contact :

    Nom: {contact.first_name} {contact.last_name}
    Email: {contact.email}
    Téléphone: {contact.phone}
    Message: {contact.message}
    """

    msg.set_content(body)

    # Attach conversation as a text file
    conversation_bytes = contact.conversation.encode("utf-8")
    msg.add_attachment(conversation_bytes, maintype="text", subtype="plain", filename="conversation.txt")

    # Updated SMTP configuration to use a public SMTP server or disable email sending if not configured
    SMTP_HOST = os.getenv("SMTP_HOST", None)
    SMTP_PORT = int(os.getenv("SMTP_PORT", 25))
    SMTP_USER = os.getenv("SMTP_USER", None)
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", None)

    if SMTP_HOST is None:
        # SMTP not configured, skip sending email
        print("SMTP not configured, skipping email sending")
        return

    await aiosmtplib.send(
        msg,
        hostname=SMTP_HOST,
        port=SMTP_PORT,
        username=SMTP_USER,
        password=SMTP_PASSWORD,
        start_tls=True if SMTP_PORT == 587 else False,
    )

@app.post("/contact/submit")
async def submit_contact(contact: ContactForm):
    try:
        # Save to Redis with a unique key
        timestamp = datetime.datetime.utcnow().isoformat()
        key = f"contact:{contact.email}:{timestamp}"
        await redis_client.hset(key, mapping=contact.dict())

        # Send email notification
        await send_email(contact)

        return JSONResponse(content={"message": "Contact enregistré et email envoyé avec succès."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

from fastapi import Body

@app.post("/chatbot/")
async def chatbot_response(
    request: Request,
    direct_answer: bool = Query(False),
):
    data = await request.json()
    question = data.get("question")
    session_id = data.get("session_id")
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    # Save user message to Redis conversation list
    if session_id:
        user_entry = f"user: {question}"
        await redis_client.rpush(f"conversation:{session_id}", user_entry)

    # Check for simple greetings to respond quickly and intelligently
    greetings = {
        "bonjour": "Bonjour! Comment puis-je vous aider ?",
        "bonsoir": "Bonsoir! Comment puis-je vous aider ?",
        "salut": "Salut! Comment puis-je vous aider ?",
        "hello": "Hello! How can I help you?",
        "hi": "Hi! How can I help you?"
    }
    question_lower = question.lower().strip()
    for greet, response_greet in greetings.items():
        if question_lower.startswith(greet):
            # Save bot response to Redis
            if session_id:
                bot_entry = f"bot: {response_greet}"
                await redis_client.rpush(f"conversation:{session_id}", bot_entry)
            return {"answer": response_greet}

    # Step 0: Fuzzy match user question against decision tree messages
    best_match = None
    best_score = 0
    for node_key, message in flat_decision_nodes:
        score = fuzz.token_set_ratio(question.lower(), message.lower())
        if score > best_score:
            best_score = score
            best_match = node_key

    MATCH_THRESHOLD = 80
    if best_score >= MATCH_THRESHOLD:
        node = decision_tree.get(best_match)
        if node:
            response_message = node.get("message", "")
            choices = node.get("choices", {})
            # Save bot response to Redis
            if session_id:
                bot_entry = f"bot: {response_message}"
                await redis_client.rpush(f"conversation:{session_id}", bot_entry)
            # Return structured response with message and choices
            if choices:
                choices_list = []
                for label, node_key in choices.items():
                    choices_list.append({"label": label, "nodeKey": node_key})
                return {"message": response_message, "choices": choices_list}
            else:
                return {"message": response_message}

    # Step 1: Use FAISS to find best matching QA block
    question_vec = model.encode([question]).astype('float32')
    D, I = index.search(question_vec, k=1)
    if I[0][0] == -1:
        raise HTTPException(status_code=404, detail="No relevant answer found")
    qa_block = documents[I[0][0]]

    # Step 2: Extract answer text from QA block (cleaned)
    lines = qa_block.splitlines()
    answer_lines = []
    in_answer = False
    for line in lines:
        stripped_line = line.lstrip()
        if stripped_line.startswith("A:"):
            in_answer = True
            answer_lines.append(stripped_line[2:].strip())
        elif in_answer:
            if stripped_line.startswith("Q:"):
                break
            cleaned_line = stripped_line
            if cleaned_line.startswith("Q:"):
                cleaned_line = cleaned_line[2:].strip()
            if cleaned_line.startswith("A:"):
                cleaned_line = cleaned_line[2:].strip()
            answer_lines.append(cleaned_line)
    context = "\n".join(answer_lines).strip()

    # If direct_answer is True, return extracted answer directly without Llama reformulation
    if direct_answer:
        # Save bot response to Redis
        if session_id:
            bot_entry = f"bot: {context}"
            await redis_client.rpush(f"conversation:{session_id}", bot_entry)
        return {"answer": context}

    # Step 3: Use local Llama model to generate response based on question and context
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    response = llm.create_completion(prompt=prompt, max_tokens=256, temperature=0.7)
    # Debug prints for diagnosis
    print("Matched QA block:", qa_block)
    print("Context extracted:", context)
    print("Prompt sent to model:", prompt)
    print("Raw response from model:", response)
    # Adjusted to access response text correctly as a dict with 'choices' list
    answer_text = response["choices"][0]["text"].strip()
    print("Generated answer:", answer_text)

    if not answer_text:
        answer_text = "Désolé, je n'ai pas pu générer de réponse pour le moment."

    # Save bot response to Redis
    if session_id:
        bot_entry = f"bot: {answer_text}"
        await redis_client.rpush(f"conversation:{session_id}", bot_entry)

    return {"answer": answer_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
