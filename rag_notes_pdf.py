"""Small local RAG app for notes and PDF files.

Put .txt, .md, and (optionally) .pdf files in the ``rag_documents`` folder
next to this script, then run: python rag_notes_pdf.py
"""

from __future__ import annotations

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


APP_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = APP_DIR / "rag_documents"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("Install PDF support with: python -m pip install pypdf") from error
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def make_chunks(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    words = text.split()
    chunks = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def load_documents(folder: Path) -> list[tuple[str, str]]:
    folder.mkdir(exist_ok=True)
    results = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            for chunk in make_chunks(read_file(path)):
                results.append((path.name, chunk))
        except (OSError, RuntimeError) as error:
            print(f"Skipped {path.name}: {error}")
    return results


def retrieve(question: str, chunks: list[tuple[str, str]], limit: int = 4) -> list[tuple[str, str]]:
    query_words = set(WORD_PATTERN.findall(question.lower()))
    scored = []
    for source, chunk in chunks:
        words = WORD_PATTERN.findall(chunk.lower())
        word_set = set(words)
        score = sum(words.count(word) for word in query_words)
        score += 2 * len(query_words & word_set)
        scored.append((score, source, chunk))
    scored.sort(reverse=True)
    return [(source, chunk) for score, source, chunk in scored[:limit] if score > 0]


def ask_ollama(question: str, context: list[tuple[str, str]], model: str) -> str:
    context_text = "\n\n".join(f"[{source}]\n{chunk}" for source, chunk in context)
    prompt = (
        "Answer only from the supplied context. If the answer is not present, say "
        "that the documents do not contain enough information.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}"
    )
    payload = json.dumps({"model": model or DEFAULT_MODEL, "prompt": prompt, "stream": False}).encode()
    request = Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        answer = json.loads(response.read().decode()).get("response", "").strip()
    return answer or "Ollama returned an empty answer."


class RagApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Notes + PDF RAG")
        self.root.geometry("900x650")
        self.chunks: list[tuple[str, str]] = []

        toolbar = tk.Frame(root, padx=10, pady=10)
        toolbar.pack(fill=tk.X)
        tk.Button(toolbar, text="Choose folder", command=self.choose_folder).pack(side=tk.LEFT)
        self.folder_label = tk.Label(toolbar, text=f"Folder: {DOCUMENT_DIR}", anchor="w")
        self.folder_label.pack(side=tk.LEFT, padx=10)
        tk.Label(toolbar, text="Ollama model:").pack(side=tk.LEFT)
        self.model = tk.Entry(toolbar, width=14)
        self.model.insert(0, DEFAULT_MODEL)
        self.model.pack(side=tk.LEFT, padx=5)

        self.chat = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, padx=10, pady=10)
        self.chat.pack(fill=tk.BOTH, expand=True, padx=10)
        bottom = tk.Frame(root, padx=10, pady=10)
        bottom.pack(fill=tk.X)
        self.question = tk.Entry(bottom)
        self.question.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.question.bind("<Return>", lambda _event: self.answer())
        tk.Button(bottom, text="Ask", command=self.answer).pack(side=tk.LEFT, padx=(8, 0))
        self.load()

    def write(self, text: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, text + "\n\n")
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def load(self) -> None:
        self.chunks = load_documents(DOCUMENT_DIR)
        self.write(f"Loaded {len(self.chunks)} chunks from {DOCUMENT_DIR}.")

    def choose_folder(self) -> None:
        global DOCUMENT_DIR
        selected = filedialog.askdirectory(initialdir=str(DOCUMENT_DIR))
        if selected:
            DOCUMENT_DIR = Path(selected)
            self.folder_label.configure(text=f"Folder: {DOCUMENT_DIR}")
            self.load()

    def answer(self) -> None:
        question = self.question.get().strip()
        if not question:
            return
        self.question.delete(0, tk.END)
        matches = retrieve(question, self.chunks)
        self.write(f"You: {question}")
        if not matches:
            self.write("No relevant passage was found. Add notes or PDFs to the selected folder.")
            return
        try:
            answer = ask_ollama(question, matches, self.model.get().strip())
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            answer = "Ollama is unavailable. Retrieved passages:\n\n" + "\n\n".join(
                f"[{source}] {chunk}" for source, chunk in matches
            ) + f"\n\n(Start Ollama for generated answers: {error})"
        self.write(f"Assistant: {answer}")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        RagApp(root)
        root.mainloop()
    except tk.TclError as error:
        messagebox.showerror("RAG app", str(error))