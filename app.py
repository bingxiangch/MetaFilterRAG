import os, json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from jose import JWTError, jwt
from passlib.context import CryptContext
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer

# === Load env ===
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
collectionName = os.getenv("collection_name")
openai_client = OpenAI(api_key=os.getenv("api_key"))
qdrant_client = QdrantClient(url=os.getenv("qdrant_url"), api_key=os.getenv("qdrant_api_key"))
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === DB Setup ===
SQLALCHEMY_DATABASE_URL = "sqlite:///./sqlite.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)

Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def get_password_hash(password): return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# === FastAPI App ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Auth Endpoints ===
class UserCreate(BaseModel):
    username: str
    password: str
    role: str
@app.delete("/clean_all_users")
def clean_all_users(db: Session = Depends(get_db)):
    num_deleted = db.query(User).delete()
    db.commit()
    return {"message": f"Deleted {num_deleted} users from the database."}


@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(
        username=user.username,
        hashed_password=get_password_hash(user.password),
        role=user.role
    )
    db.add(db_user)
    db.commit()
    return {"message": "User registered"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}

# === Query Metadata Extraction ===
system_prompt = """
You are part of an information system that processes user queries related to medical and pharmaceutical topics.

Your task is to read the user's query and identify which specific information type(s) it is asking about. Choose only from the following predefined list of medical information types:

- information
- symptoms
- treatment
- inheritance
- frequency
- genetic changes
- causes
- exams and tests
- research
- outlook
- susceptibility
- considerations
- prevention
- stages
- complications
- support groups

Return your answer strictly in the following JSON format:

{ "qtype": ["<type1>", "<type2>", ...] }

Only include types that are clearly relevant based on the query. If the query touches on multiple topics, include all that apply. If none match clearly, return an empty list:

{ "qtype": [] }
"""
role_to_qtypes = {
    "patient": [
        "symptoms", "information", "prevention", "support groups", "considerations"
    ],
    "doctor": [
        "symptoms", "treatment", "inheritance", "frequency", "genetic changes",
        "causes", "exams and tests", "susceptibility", "outlook", "complications", "stages"
    ],
    "researcher": [
        "research", "genetic changes", "outlook", "causes"
    ],
    "admin": [
        # Admin can access all qtypes explicitly listed here (union of all above plus any extra)
        "symptoms", "diagnosis", "treatment", "stages", "exams and tests", "causes",
        "susceptibility", "outlook", "frequency", "inheritance", "genetic changes",
        "prevention", "complications", "medication", "dosage", "usage",
        "storage and disposal", "contraindication", "side effects", "research",
        "patient education", "general information", "support groups", "considerations"
    ]
}


def is_access_allowed(user_role: str, requested_qtypes: list[str], role_to_qtypes: dict[str, list[str]]):
    """
    Check if the user role has permission to access all requested qtypes.

    Args:
        user_role (str): Role of the user, e.g., "doctor"
        requested_qtypes (list[str]): List of qtypes extracted from query, e.g., ["symptoms", "treatment"]
        role_to_qtypes (dict[str, list[str]]): Mapping from roles to allowed qtypes

    Returns:
            allowed: True if all qtypes allowed, False otherwise
    """
    allowed_qtypes = role_to_qtypes.get(user_role, [])
    for qt in requested_qtypes:
        if qt not in allowed_qtypes:
            return False

    return True


class QueryRequest(BaseModel):
    prompt: str

@app.post("/query")
def query(req: QueryRequest, user: User = Depends(get_current_user)):
    query = req.prompt
    qtypes = extract_qtypes(query)
    allowed_qtypes = role_to_qtypes.get(user.role, [])

    # Step 1: Embed query
    vector = embedding_model.encode(query).tolist()

    # Step 2: Deny if top-1 hit is restricted
    top1_blocked = is_top1_hit_forbidden(vector, allowed_qtypes, user.role)
    if top1_blocked:
        return top1_blocked

    # Step 3: Build Qdrant filter for role
    query_filter, access_suggestion = build_qtype_filter(user.role, qtypes)

    # Step 4: Get relevant context from Qdrant
    filtered_result = run_filtered_qdrant_query(vector, query_filter)
    context, sources_used = build_context(filtered_result)

    # Step 5: Call OpenAI
    prompt = build_system_prompt()
    raw_answer = get_answer_from_openai(query, context, prompt)
    final_answer = format_answer_with_source(raw_answer, sources_used)

    return {
        "answer": final_answer,
        "user_role": user.role,
        "used_qtypes": qtypes,
        "access_suggestion": access_suggestion
    }
def extract_qtypes(query: str):
    metadata = extract_metadata(query)
    return metadata.get("qtype", [])

def is_top1_hit_forbidden(vector, allowed_qtypes, user_role):
    try:
        result = qdrant_client.query_points(
            collection_name=collectionName,
            query=vector,
            limit=1
        )
        if result.points:
            top_hit = result.points[0]
            qtypes = top_hit.payload.get("qtype", [])
            qtypes = [qtypes] if isinstance(qtypes, str) else qtypes
            denied = [qt for qt in qtypes if qt not in allowed_qtypes]

            if denied:
                return {
                    "answer": (
                        "The answer is not available because the top relevant content could not be retrieved.\n"
                        f"Context is not provided because your role ('{user_role}') is not allowed to access content about: {', '.join(denied)}."
                    ),
                    "user_role": user_role,
                    "used_qtypes": qtypes,
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unfiltered Qdrant query failed: {e}")
    return None

def build_qtype_filter(user_role, qtypes):
    allowed = role_to_qtypes.get(user_role, [])
    if not is_access_allowed(user_role, qtypes, role_to_qtypes):
        denied = [qt for qt in qtypes if qt not in allowed]
        suggestion = (
            "Note: Your query includes restricted content and some parts were filtered.\n"
            f"Denied qtypes: {', '.join(denied)}"
        )
        return Filter(should=[FieldCondition(key="qtype", match=MatchText(text=qt)) for qt in allowed]), suggestion
    else:
        return Filter(should=[FieldCondition(key="qtype", match=MatchText(text=qt)) for qt in qtypes]), ""

def run_filtered_qdrant_query(vector, query_filter):
    try:
        result = qdrant_client.query_points(
            collection_name=collectionName,
            query=vector,
            query_filter=query_filter,
            limit=5
        )
        return [pt for pt in result.points if pt.score >= 0.85]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filtered Qdrant query failed: {e}")

def build_context(filtered_result):
    context_blocks = []
    sources_used = set()
    for hit in filtered_result:
        question = hit.payload.get("question", "").strip()
        answer = hit.payload.get("answer", "").strip()
        source = hit.payload.get("source", "Unknown")
        if answer:
            context_blocks.append(f"Q: {question}\nA: {answer}")
            sources_used.add(source)
    return "\n\n".join(context_blocks), sources_used

def build_system_prompt():
    return (
        "You are a medical assistant. Use ONLY the information explicitly provided in the context to answer the question. "
        "If the answer is not found in the context, respond exactly with: \"The answer is not available in the provided context.\" "
        "Do not use prior knowledge or make assumptions beyond the text."
    )

def format_answer_with_source(raw_answer, sources_used):
    if not is_answer_missing(raw_answer) and sources_used:
        return f"{raw_answer}\n\n(Answer retrieved from: {', '.join(sorted(sources_used))})"
    return raw_answer



# === Helper Functions ===
def extract_metadata(query: str) -> dict:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {e}")

def get_answer_from_openai(query, context, prompt):
    try:
        full_input = f"Context:\n{context}\n\nQuery:\n{query}"

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": full_input}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

def format_extracted_filter(filter_dict):
    parts = []
    for key, value in filter_dict.items():
        if isinstance(value, list):
            parts.append(f"{key}: $in [{', '.join(value)}]")
        else:
            parts.append(f"{key}: {value}")
    return ', '.join(parts)

def is_answer_missing(answer: str) -> bool:
    return "the answer is not available in the provided context." in answer.lower()
