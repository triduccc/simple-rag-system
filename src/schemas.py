from pydantic import BaseModel, Field
from typing import Optional

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

    
