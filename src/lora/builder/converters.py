import csv
import json
import pdfplumber
from pathlib import Path
from typing import Dict, List, Protocol

from src.core.schema import Example

class RawFileConverter(Protocol):
    def __call__(self, file_path: Path) -> List[Example]:
        ...

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
    messages = [{"role": "user", "content": content}]
    return [Example(messages=messages, meta={"source": file_path.name})]

def _from_pdf(file_path: Path) -> List[Example]:
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    messages = [{"role": "user", "content": text}]
    return [Example(messages=messages, meta={"source": file_path.name})]

def _from_code(file_path: Path) -> List[Example]:
    """Loads content from a source code file."""
    content = file_path.read_text(encoding="utf-8")
    messages = [{"role": "user", "content": content}]
    return [Example(messages=messages, meta={"source": file_path.name})]


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