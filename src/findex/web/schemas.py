from pydantic import BaseModel, Field


class SearchItem(BaseModel):
    doc_id: int = Field(description="Document integer ID")
    title: str = Field(description="Document title")
    score: float = Field(description="Relevance ranking score")
    snippet: str = Field(description="Highlighted text window")


class SearchResponse(BaseModel):
    query: str
    total: int
    page: int
    took_ms: float
    results: list[SearchItem]


class StatsResponse(BaseModel):
    index_size_bytes: int
    doc_count: int
    vocabulary_size: int
    uptime_seconds: float
