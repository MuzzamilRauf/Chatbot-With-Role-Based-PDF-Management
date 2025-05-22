from sqlalchemy import Column, Integer, String, Enum, TIMESTAMP, ForeignKey, text, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False, default="user")

    # Relationship to UploadedPDF
    uploaded_pdfs = relationship("PDFUpload", back_populates="admin")


class PDFUpload(Base):
    __tablename__ = "uploaded_pdfs"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String(255), unique=True, nullable=False)
    file_name = Column(String(255), nullable=False)
    pdf_path = Column(String(255), nullable=False)
    admin_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to User
    admin = relationship("User", back_populates="uploaded_pdfs")
