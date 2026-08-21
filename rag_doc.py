"""Local RAG chat app for notes and PDF documents.

Usage:
	python d:\rag_doc.py

Place .txt, .md, and PDF files in the ``rag_documents`` folder next to this
file. PDF support requires: python -m pip install pypdf
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
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def read_document(path: Path) -> str:
	"""Read a text, Markdown, or PDF document."""
	if path.suffix.lower() == ".pdf":
		try:
			from pypdf import PdfReader
		except ImportError as error:
			raise RuntimeError(
				"PDF support is missing. Run: python -m pip install pypdf"
			) from error
		return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
	return path.read_text(encoding="utf-8", errors="ignore")


def split_into_chunks(text: str, size: int = 180, overlap: int = 30) -> list[str]:
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
	chunks = []
	for path in sorted(folder.iterdir()):
		if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
			continue
		try:
			chunks.extend((path.name, chunk) for chunk in split_into_chunks(read_document(path)))
		except (OSError, RuntimeError) as error:
			print(f"Skipped {path.name}: {error}")
	return chunks


def search_documents(question: str, chunks: list[tuple[str, str]], limit: int = 4):
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


def generate_answer(question: str, matches, model: str) -> str:
	context = "\n\n".join(f"[{source}]\n{chunk}" for source, chunk in matches)
	prompt = (
		"Answer using only the context below. If the answer is not in the context, "
		"say that the documents do not contain enough information.\n\n"
		f"Context:\n{context}\n\nQuestion: {question}"
	)
	payload = json.dumps(
		{"model": model or "mistral", "prompt": prompt, "stream": False}
	).encode("utf-8")
	request = Request(
		OLLAMA_URL,
		data=payload,
		headers={"Content-Type": "application/json"},
	)
	with urlopen(request, timeout=90) as response:
		answer = json.loads(response.read().decode("utf-8")).get("response", "")
	return answer.strip() or "Ollama returned an empty answer."


class RagWindow:
	def __init__(self, root: tk.Tk):
		self.root = root
		self.root.title("Notes and PDF RAG")
		self.root.geometry("900x650")
		self.chunks: list[tuple[str, str]] = []

		toolbar = tk.Frame(root, padx=10, pady=10)
		toolbar.pack(fill=tk.X)
		tk.Button(toolbar, text="Choose folder", command=self.choose_folder).pack(side=tk.LEFT)
		self.folder_label = tk.Label(toolbar, text=str(DOCUMENT_DIR), anchor="w")
		self.folder_label.pack(side=tk.LEFT, padx=10)
		tk.Label(toolbar, text="Ollama model:").pack(side=tk.LEFT)
		self.model = tk.Entry(toolbar, width=14)
		self.model.insert(0, "mistral")
		self.model.pack(side=tk.LEFT, padx=5)

		self.output = scrolledtext.ScrolledText(
			root, wrap=tk.WORD, state=tk.DISABLED, padx=10, pady=10
		)
		self.output.pack(fill=tk.BOTH, expand=True, padx=10)
		bottom = tk.Frame(root, padx=10, pady=10)
		bottom.pack(fill=tk.X)
		self.question = tk.Entry(bottom)
		self.question.pack(side=tk.LEFT, fill=tk.X, expand=True)
		self.question.bind("<Return>", lambda _event: self.ask())
		tk.Button(bottom, text="Ask", command=self.ask).pack(side=tk.LEFT, padx=(8, 0))
		self.load()

	def write(self, text: str) -> None:
		self.output.configure(state=tk.NORMAL)
		self.output.insert(tk.END, text + "\n\n")
		self.output.configure(state=tk.DISABLED)
		self.output.see(tk.END)

	def load(self) -> None:
		self.chunks = load_documents(DOCUMENT_DIR)
		self.write(f"Loaded {len(self.chunks)} document chunks from {DOCUMENT_DIR}.")

	def choose_folder(self) -> None:
		global DOCUMENT_DIR
		selected = filedialog.askdirectory(initialdir=str(DOCUMENT_DIR))
		if selected:
			DOCUMENT_DIR = Path(selected)
			self.folder_label.configure(text=str(DOCUMENT_DIR))
			self.load()

	def ask(self) -> None:
		question = self.question.get().strip()
		if not question:
			return
		self.question.delete(0, tk.END)
		self.write(f"You: {question}")
		matches = search_documents(question, self.chunks)
		if not matches:
			self.write("No relevant passage was found in the loaded documents.")
			return
		try:
			answer = generate_answer(question, matches, self.model.get().strip())
		except (HTTPError, URLError, TimeoutError, OSError) as error:
			passages = "\n\n".join(f"[{source}] {chunk}" for source, chunk in matches)
			answer = f"Ollama is unavailable. Retrieved passages:\n\n{passages}\n\n{error}"
		self.write(f"Assistant: {answer}")


if __name__ == "__main__":
	try:
		app_root = tk.Tk()
		RagWindow(app_root)
		app_root.mainloop()
	except tk.TclError as error:
		messagebox.showerror("RAG app", str(error))
