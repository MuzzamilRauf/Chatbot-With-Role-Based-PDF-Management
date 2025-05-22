import os
import uuid
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from typing import List
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models, schemas, utils, auth
from chatbot_utils1 import RAGPipeline
from auth import get_current_user
from schemas import PDFUploadOut
from models import PDFUpload

# Initialize the models and create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Simplified request model
class ChatQuery(BaseModel):
    query: str

# RAGPipeline setup
rag_pipeline = None
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    try:
        rag_pipeline = RAGPipeline()
        print("RAGPipeline initialized successfully on startup!")
    except Exception as e:
        print(f"Failed to initialize RAGPipeline: {str(e)}")
        raise

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "API is running"}


@app.post("/register")
def register(user: schemas.UserRegister, db: Session = Depends(get_db)):
    # Check if username or email already exists
    existing_user = db.query(models.User).filter(
        (models.User.username == user.username) |
        (models.User.email == user.email)
    ).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")

    # Validate role (optional step, you can modify as per your needs)
    if user.role not in ["user", "admin"]:
        raise HTTPException(status_code = 400, detail="Invalid role. Valid roles are 'user' and 'admin'")

    # Hash the password
    hashed_password = utils.get_password_hash(user.password)

    # Create a new user with role and hashed password
    new_user = models.User(
        username=user.username,
        email=user.email,
        password_hash=hashed_password,
        role=user.role  # Assign role here
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully", "status": True}


@app.post("/login")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not utils.verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create access token with username and role
    access_token = auth.create_access_token(data={"sub": db_user.username, "role": db_user.role})

    return {"access_token": access_token, "token_type": "bearer", "user": db_user, "status": True}


@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin users can upload PDFs")

    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        pdf_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Generate a unique file ID
        file_id = str(uuid.uuid4())

        # Process with RAG
        rag_pipeline.process_pdf_and_store_embeddings(pdf_path, file_id=file_id)

        # Save PDF upload metadata in database
        pdf_record = models.PDFUpload(
            file_id=file_id,
            file_name=file.filename,
            pdf_path=pdf_path,
            admin_id=current_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(pdf_record)
        db.commit()
        db.refresh(pdf_record)

        return {
            "message": f"PDF {file.filename} uploaded and processed successfully",
            "pdf_id": pdf_record.id,
            "file_id": file_id,
            "admin_id": pdf_record.admin_id,
            "uploaded_by": current_user.username,
            "timestamp": pdf_record.created_at,
            "status": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# @app.delete("/delete_pdf/{pdf_id}")
# async def delete_pdf(
#     pdf_id: str,
#     current_user: models.User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     if current_user.role != "admin":
#         raise HTTPException(status_code=403, detail="Only admin users can delete PDFs")
#
#     # Get PDF record by ID
#     pdf_record = db.query(models.PDFUpload).filter(models.PDFUpload.file_id == pdf_id).first()
#
#     if not pdf_record:
#         raise HTTPException(status_code=404, detail="PDF not found")
#
#     # Extract file_id for vector deletion (we’ll use file name here as unique ID)
#     file_id = pdf_record.file_id
#
#     # Delete vectors from Pinecone
#     deletion_result = rag_pipeline.delete_vectors_by_file_id(file_id)
#
#     # Delete PDF file from disk
#     if os.path.exists(pdf_record.pdf_path):
#         os.remove(pdf_record.pdf_path)
#
#     # Remove the record from the database
#     db.delete(pdf_record)
#     db.commit()
#
#     return {
#         "message": f"PDF '{pdf_record.file_name}' and {deletion_result['deleted']} vectors deleted successfully.",
#         "status": True
#     }

from uuid import UUID
import logging
import os

@app.delete("/delete_pdf/{pdf_id}")
async def delete_pdf(
    pdf_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admin users can delete PDFs")

    logging.info(f"Attempting to delete PDF with ID: {pdf_id}")

    # Get PDF record by ID, ensure pdf_id is converted to UUID if needed
    try:
        # pdf_record = db.query(models.PDFUpload).filter(models.PDFUpload.file_id == UUID(pdf_id)).first()
        pdf_record = db.query(models.PDFUpload).filter(models.PDFUpload.file_id == pdf_id).first()

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF ID format")

    if not pdf_record:
        raise HTTPException(status_code=404, detail="PDF not found")

    # Extract file_id for vector deletion (we’ll use file name here as unique ID)
    file_id = pdf_record.file_id

    # Delete vectors from Pinecone
    deletion_result = rag_pipeline.delete_vectors_by_file_id(file_id)

    # Delete PDF file from disk
    if os.path.exists(pdf_record.pdf_path):
        os.remove(pdf_record.pdf_path)

    # Remove the record from the database
    db.delete(pdf_record)
    db.commit()

    return {
        "message": f"PDF '{pdf_record.file_name}' and {deletion_result['deleted']} vectors deleted successfully.",
        "status": True
    }



@app.get("/get_files", response_model=List[PDFUploadOut])
def get_files(admin_id: int = Query(...), db: Session = Depends(get_db)):
    pdfs = db.query(PDFUpload).filter(PDFUpload.admin_id == admin_id).all()
    if not pdfs:
        raise HTTPException(status_code=404, detail="No PDFs found for this admin ID")

    return pdfs


@app.post("/answer")
async def chat(request: ChatQuery):
    try:
        response = rag_pipeline.retrieve_and_generate_response(request.query)
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)