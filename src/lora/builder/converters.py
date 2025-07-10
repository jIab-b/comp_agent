import csv
import json
import pdfplumber
from pathlib import Path
from typing import Dict, List, Protocol

from src.core.schema import Example
from .validators import MAX_CONTEXT_LENGTH

class RawFileConverter(Protocol):
    def __call__(self, file_path: Path) -> List[Example]:
        ...

def chunk_text(text: str, source_name: str) -> List[Example]:
    """Splits a long text into multiple smaller Example chunks."""
    # Simple split by paragraphs. A more sophisticated method could use a proper tokenizer.
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 > MAX_CONTEXT_LENGTH:
            if current_chunk:
                messages: List[dict] = [{"role": "user", "content": current_chunk.strip()}]
                chunks.append(Example(messages=messages, meta={"source": source_name}))
            current_chunk = p
        else:
            current_chunk += "\n\n" + p

    if current_chunk.strip():
        messages: List[dict] = [{"role": "user", "content": current_chunk.strip()}]
        chunks.append(Example(messages=messages, meta={"source": source_name}))

    return chunks

def _from_json(file_path: Path) -> List[Example]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects.")

    return [Example(**item) for item in data]

def _from_csv(file_path: Path) -> List[Example]:
    examples = []
    with file_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages = [
                {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": row["assistant"]},
            ]
            examples.append(Example(messages=messages, meta={"source": file_path.name}))
    return examples

def _from_markdown(file_path: Path) -> List[Example]:
    content = file_path.read_text(encoding="utf-8")
    return chunk_text(content, file_path.name)

def _from_pdf(file_path: Path) -> List[Example]:
    """Extracts text from a PDF file and chunks it."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return chunk_text(text, file_path.name)

def _from_code(file_path: Path) -> List[Example]:
    """Loads content from a source code file and chunks it."""
    content = file_path.read_text(encoding="utf-8")
    return chunk_text(content, file_path.name)


CONVERTERS: Dict[str, RawFileConverter] = {
    ".json": _from_json,
    ".csv": _from_csv,
    ".md": _from_markdown,
    ".pdf": _from_pdf,
    ".py": _from_code,
    ".cpp": _from_code,
}

def convert_file(file_path: Path) -> List[Example]:
    ext = file_path.suffix.lower()
    if ext not in CONVERTERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return CONVERTERS[ext](file_path)