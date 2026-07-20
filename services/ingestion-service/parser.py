import os
import re
from pathlib import Path

import pandas as pd
from docx import Document as DocxDocument
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredImageLoader, UnstructuredFileLoader
import pytesseract

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
OCR_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)


def clean_ocr_text(raw_text: str) -> str:
    """Normalizes OCR output into readable text for query routing and indexing."""
    if not raw_text:
        return ""

    cleaned = re.sub(r"\s+", " ", raw_text)
    return cleaned.strip()


def extract_text_from_ocr_file(file_path: str) -> list[str]:
    """Extracts readable text from images and PDFs using OCR loaders."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        page_content = []

        try:
            loader = UnstructuredImageLoader(str(path))
            raw_docs = loader.load()
            for doc in raw_docs:
                page_text = getattr(doc, "page_content", "") or ""
                if page_text.strip():
                    page_content.append(page_text.strip())
        except Exception as exc:
            print(f"Unstructured image OCR fallback failed: {exc}")

        if not page_content:
            try:
                with Image.open(path) as img:
                    page_text = pytesseract.image_to_string(img)
                if page_text.strip():
                    page_content.append(page_text.strip())
            except Exception as exc:
                print(f"Pytesseract image OCR failed: {exc}")

        if page_content:
            return OCR_TEXT_SPLITTER.split_text("\n\n".join(page_content))
        return []

    if ext == ".pdf":
        page_content = []

        try:
            loader = UnstructuredFileLoader(str(path), mode="elements")
            raw_docs = loader.load()
            for doc in raw_docs:
                page_text = getattr(doc, "page_content", "") or ""
                if page_text.strip():
                    page_content.append(page_text.strip())
        except Exception as exc:
            print(f"Unstructured PDF loader failed: {exc}")

        if not page_content:
            try:
                pages = convert_from_path(path, dpi=300)
                for page_number, page in enumerate(pages, start=1):
                    page_text = pytesseract.image_to_string(page)
                    if page_text.strip():
                        page_content.append(f"Page {page_number}:\n{page_text.strip()}")
            except Exception as exc:
                print(f"PDF OCR fallback failed: {exc}")

        if page_content:
            return OCR_TEXT_SPLITTER.split_text("\n\n".join(page_content))
        return []

    return []


def extract_text_from_file(file_path: str) -> list[str]:
    """Reads a file based on its extension and returns a list of text chunks."""
    path = Path(file_path)
    ext = path.suffix.lower()
    docs = []

    if ext == ".csv":
        df = pd.read_csv(path, encoding="latin1")
        for _, row in df.iterrows():
            text = f"""
Laptop:
Company: {row.get('Company', 'Unknown')}
Inches: {row.get('Inches', 'Unknown')}
TypeName: {row.get('TypeName', 'Unknown')}
Operating System: {row.get('OpSys', 'Unknown')}
Screen Size: {row.get('Inches', 'Unknown')} inches
Screen Resolution: {row.get('ScreenResolution', 'Unknown')}
CPU: {row.get('Cpu', 'Unknown')}
GPU: {row.get('Gpu', 'Unknown')}
RAM: {row.get('Ram_GB', 'Unknown')}
ROM: {row.get('Total_Storage_GB', 'Unknown')}
"""
            docs.append(text.strip())

    elif ext in (".txt", ".docx"):
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
        else:
            doc = DocxDocument(path)
            full_text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""],
        )
        docs = text_splitter.split_text(full_text)

    elif ext in IMAGE_EXTENSIONS or ext == ".pdf":
        docs = extract_text_from_ocr_file(file_path)

    else:
        raise ValueError(f"Unsupported file format: {ext}. Please upload .csv, .txt, .docx, .pdf, or an image file")

    return docs


def extract_text_as_query(file_path: str) -> str:
    """Convert OCR-loaded image/PDF content into a single query string for the chat pipeline."""
    extracted_chunks = extract_text_from_file(file_path)
    if not extracted_chunks:
        return ""

    normalized_chunks = [clean_ocr_text(chunk) for chunk in extracted_chunks if clean_ocr_text(chunk)]
    return "\n".join(normalized_chunks)
