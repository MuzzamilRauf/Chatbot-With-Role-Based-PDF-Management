from pydantic import BaseModel, EmailStr
from typing import Literal
from datetime import datetime


class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: Literal["user", "admin"] = "user"

class UserLogin(BaseModel):
    username: str
    password: str

class PDFMeta(BaseModel):
    id: int
    file_name: str
    admin_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class PDFUploadOut(BaseModel):
    id: int
    file_name: str
    file_id: str
    pdf_path: str
    admin_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True