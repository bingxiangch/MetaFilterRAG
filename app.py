from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import OpenAI
import os
import json
from dataclasses import is_dataclass, asdict

# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
collectionName = os.getenv("collection_name")
qdrant_url = os.getenv("qdrant_url")
# OpenAI + Qdrant setup
openai_client = OpenAI(api_key=os.getenv("api_key"))
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=os.getenv("qdrant_api_key")
)


# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./sqlite.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine, autoflush=False)

# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)
    department = Column(String)

Base.metadata.create_all(bind=engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

# Token generation
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependencies
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role = payload.get("role")
        department = payload.get("department")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    user.role = role
    user.department = department
    return user

# FastAPI app
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


system_prompt = """
Extract the following metadata from the user's query and return it in valid JSON format. 

- `patient_id`: A list of full patient name(s), e.g., ["Barbara Cantu"].
- `department`: a list of department,  e.g., [Cardiology"].
- `section`: A list of section,  e.g., ["Basic Information"]:
  - "Basic Information"
  - "Current Condition"
  - "Treatment"
  - "Medical History"
  - "Lab Results"
  - "Billing Information"

Only include fields that are explicitly mentioned or clearly implied in the query. Omit any field that is missing or ambiguous.

Query: "{query}"
"""


# Request schemas
class QueryRequest(BaseModel):
    prompt: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str
    department: str

class UserUpdate(BaseModel):
    password: str
    role: str
    department: str

# User Registration
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    hashed_pw = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        hashed_password=hashed_pw,
        role=user.role,
        department=user.department
    )
    db.add(db_user)
    db.commit()
    return {"message": "User registered successfully"}

@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [
        {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "department": user.department
        }
        for user in users
    ]

@app.put("/update_user/{username}")
def update_user(username: str, user_update: UserUpdate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == username).first()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if user_update.password:
        db_user.hashed_password = get_password_hash(user_update.password)

    if user_update.role:
        db_user.role = user_update.role

    if user_update.department:
        db_user.department = user_update.department

    db.commit()
    return {"message": "User updated successfully"}

# Login
@app.post("/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
            "department": user.department
        },
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# Query Endpoint with Permission Check
@app.post("/query")
async def handle_query(req: QueryRequest, user: User = Depends(get_current_user)):
    query = req.prompt
    user_department = user.department

    metadata = extract_metadata(query)
    metadata['department'] = user_department
    # Move "department" to the beginning
    ordered_metadata = {'department': metadata.pop('department'), **metadata}
    extracted_filter = format_extracted_filter(ordered_metadata)
    print("extracted_filter: ", extracted_filter)
    extracted_filter = format_extracted_filter(ordered_metadata)

    print('metadata: ', metadata)
    is_allowed = check_section_permission(metadata.get("section", []), user.role)
    if not is_allowed:
        answer = f"Sorry, you do not have permission to access the section: {', '.join(metadata['section'])}."
        return {"answer": answer, "query_filter": extracted_filter}
    query_filter = build_qdrant_filter(metadata, user_department)
    print('query_filter: ', query_filter)
    # query_filter_json = generate_in_json(user_department, metadata['patient_id'], metadata['section'])
    # print("query_filter_json: ", query_filter_json)


    search_result = search_with_fallback(query, query_filter)
    print('search_result: ', search_result)
    answer_system_prompt = build_answer_prompt(search_result, user)
    context = "\n\n".join([hit.document for hit in search_result])

    final_response = get_answer_from_openai(query, context, answer_system_prompt)
    # Prepare the final response
    
    return {"answer": final_response, "query_filter": extracted_filter}


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


def check_section_permission(sections: list, role: str):
    allowed_sections = {
        "doctor": ["Basic Information", "Current Condition", "Treatment", "Medical History", "Lab Results"],
        "financial_staff": ["Basic Information", "Billing Information"],
        "receptionist": ["Basic Information"],
        "pharmacist": ["Basic Information", "Treatment"]
    }
    allowed = allowed_sections.get(role, [])
    if sections and not any(s in allowed for s in sections):
        return False
        # system_prompt = f"Sorry, you do not have permission to access the section(s): {', '.join(sections)}."
    else:
        return True
def build_qdrant_filter(metadata: dict, user_department: str):
    must_conditions = []

    if user_department:
        must_conditions.append(models.FieldCondition(
            key="department",
            match=models.MatchValue(value=user_department)
        ))

    if "patient_id" in metadata:
        patient_conditions = [
            models.FieldCondition(
                key="patient_id",
                match=models.MatchText(text=pid)
            ) for pid in metadata["patient_id"]
        ]
        if patient_conditions:
            must_conditions.append(models.Filter(should=patient_conditions))

    if "section" in metadata:
        must_conditions.extend([
            models.FieldCondition(
                key="section",
                match=models.MatchText(text=section)
            ) for section in metadata["section"]
        ])

    return models.Filter(must=must_conditions) if must_conditions else None


def search_with_fallback(query: str, query_filter):
    try:
        result = qdrant_client.query(
            collection_name=collectionName,
            query_text=query,
            limit=10,
            query_filter=query_filter
        )
        if result:
            return result

        # Retry without filter
        fallback_result = qdrant_client.query(
            collection_name=collectionName,
            query_text=query,
            limit=10,
            query_filter=None
        )
        return fallback_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant query failed: {e}")


def build_answer_prompt(search_result, user: User) -> str:
    if not search_result:
        return "You are a helpful medical assistant. No information was found in the context."

    top_result = search_result[0]
    result_dept = top_result.metadata.get("department")
    result_patient = top_result.metadata.get("patient_id")

    if result_dept != user.department:
        return (
            f"You are a helpful medical assistant. The user tried to access '{result_patient}' in '{result_dept}' "
            f"but their department is '{user.department}'. Respond with: "
            f"\"You don't have permission to access {result_patient}'s records in {result_dept} department.\""
        )

    return (
        "You are a helpful medical assistant. Answer only using the context. "
        "If no answer is found, say 'No information found in the context.'"
    )


def get_answer_from_openai(query: str, context: str, system_prompt: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}"},
                {"role": "user", "content": f"Query:\n{query}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI response error: {e}")


# Protected query endpoint
@app.post("/test_query")
async def handle_query(req: QueryRequest):
    query = req.prompt
    # Extract metadata from query
    try:
        meta_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        metadata = json.loads(meta_response.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {e}")
    try:
        search_result = qdrant_client.query(
            collection_name=collectionName,
            query_text=query,
            limit=10,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant query error: {e}")

    # Combine context
    context = "\n\n".join([hit.document for hit in search_result])
    print('context: ',context)

    # Ask OpenAI with context
    try:
        final_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Answer only using the context. If no answer, say 'No information found in the context.'"},
                {"role": "user", "content": f"Context:\n{context}"},
                {"role": "user", "content": query}
            ]
        )
        answer = final_response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI answer error: {e}")

    return {"query": query, "metadata": metadata, "answer": answer}


def generate_in_json(department_values, patient_id_values, section_values):
    """
    Generates a query filter JSON with $in for patient_id and section fields,
    but not for department if there's only one value.

    Parameters:
    - department_values: List of department values (e.g., ["Cardiology"])
    - patient_id_values: List of patient_id values (e.g., ["Abigail Collins", "Layla Harris"])
    - section_values: List of section values (e.g., ["Current Condition", "Medical History"])

    Returns:
    - JSON object with the query filter.
    """
    must_conditions = []
    
    # Use department value directly (no $in)
    must_conditions.append({"key": "department", "match": department_values})
    
    # Use $in for patient_id and section if there are multiple values
    if len(patient_id_values) > 1:
        must_conditions.append({"key": "patient_id", "match": {"$in": patient_id_values}})
    else:
        must_conditions.append({"key": "patient_id", "match": patient_id_values[0]})
    
    if len(section_values) > 1:
        must_conditions.append({"key": "section", "match": {"$in": section_values}})
    else:
        must_conditions.append({"key": "section", "match": section_values[0]})
    
    # Final query filter
    query_filter = {"must": must_conditions}
    
    return json.dumps(query_filter, indent=2)

def format_extracted_filter(filter_dict):
    parts = []
    for key, value in filter_dict.items():
        if isinstance(value, list):
            parts.append(f"{key}: $in [{', '.join(value)}]")
        else:
            parts.append(f"{key}: {value}")
    return ', '.join(parts)