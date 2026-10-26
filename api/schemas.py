from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    question: Question = Field(description="A natural-language question about the indexed documents.")
    k: int = Field(default=4, ge=1, le=10, description="How many chunks to retrieve as context.")

    model_config = {"json_schema_extra": {"examples": [
        {"question": "How many days of paid annual leave do full-time employees get?", "k": 4}
    ]}}


class Source(BaseModel):
    source: str = Field(description="PDF file name the chunk came from.")
    page: int
    score: float = Field(description="Cosine similarity between the question and the chunk.")
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    generated: bool = Field(
        description="True if `answer` was written by the LLM. False when no GEMINI_API_KEY is configured, "
                    "in which case `answer` is a placeholder and only `sources` are meaningful."
    )
    sources: list[Source]


class DocumentInfo(BaseModel):
    filename: str
    origin: Literal["sample", "uploaded"]
    pages: int


class UploadResponse(BaseModel):
    filename: str = Field(description="The (sanitised) name the document was stored under.")
    pages: int
    replaced: bool = Field(description="True if an earlier upload with the same name was replaced.")
    total_documents: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    backend: str
    documents: int
    chunks: int
    llm_configured: bool
    model: str
