from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from filters import MetadataFilter

class ChunkMetadata(BaseModel):
    document_id: str
    file_name: str
    source: str
    page: int
    chunk_id: str
    section: Optional[str] = None

class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: ChunkMetadata

class Citation(BaseModel):
    source_index: int
    source_marker: str
    filename: str
    page: int
    section: Optional[str] = None
    chunk_id: str

class RagAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int


class DocumentInfo(BaseModel):
    filename: str
    document_id: str
    size_bytes: int
    modified_at: datetime

    
