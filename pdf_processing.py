import os
import json
import base64
import fitz  # PyMuPDF for text extraction
import pdfplumber  # For table extraction
import openai
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define Paths
PDF_FOLDER = "irish"
OUTPUT_JSON = "processed_data.json"
FAISS_INDEX_FILE = "vectorstore/db_faiss"

# Initialize FAISS & OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Storage for extracted data
data_storage = {}

def chunk_text(text, chunk_size=1200, overlap=300):
    """Splits text into smaller overlapping chunks for better FAISS retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def extract_text_with_openai(text):
    """Uses OpenAI API to process and improve extracted text from PDFs."""
    if not text.strip():
        return ""

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Extract and format the text from this document, ensuring clarity and completeness."},
            {"role": "user", "content": text}
        ]
    )
    
    return response.choices[0].message.content

def extract_text_from_pdf(pdf_path):
    """Extracts text from PDFs using PyMuPDF first, then refines with OpenAI."""
    try:
        doc = fitz.open(pdf_path)
        extracted_text = "\n".join(page.get_text("text") for page in doc)

        if not extracted_text.strip():
            raise ValueError("No text extracted, switching to OCR")

        # Use OpenAI to refine and improve text extraction
        processed_text = extract_text_with_openai(extracted_text)
        return processed_text if processed_text.strip() else extracted_text

    except Exception as e:
        print(f"⚠️ MuPDF failed for {pdf_path}, using OpenAI OCR. Error: {e}")
        return extract_text_from_images(pdf_path)

def extract_tables_from_pdf(pdf_path):
    """Extracts tables from PDFs using pdfplumber and refines with OpenAI."""
    tables_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                table_str = "\n".join([" | ".join(str(cell) if cell else "" for cell in row) for row in table])
                tables_text.append(table_str)

    if tables_text:
        combined_tables = "\n\n".join(tables_text)
        return extract_text_with_openai(combined_tables)  # Process with OpenAI for better structure
    else:
        return "No tables found."

def extract_text_from_images(pdf_path):
    """Extracts text from images inside a PDF using OpenAI OCR."""
    ocr_text = []
    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        images = doc[page_index].get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Convert image to Base64 for OpenAI API
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            # Send to OpenAI Vision API
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Extract all readable text from the image."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract the text from this image:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_ext};base64,{encoded_image}"}}
                    ]}
                ]
            )

            # Extract text from response
            extracted_text = response.choices[0].message.content
            ocr_text.append(extracted_text)

    return "\n".join(ocr_text)

def process_pdfs():
    """Processes all PDFs, extracts text, tables, images, stores in JSON, and updates FAISS."""
    global data_storage

    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Folder '{PDF_FOLDER}' does not exist.")
        return

    all_text_chunks = []  # Store all chunks before indexing in FAISS

    for pdf_file in os.listdir(PDF_FOLDER):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            print(f"📄 Processing: {pdf_file}")

            # Extract text (Normal or OCR fallback)
            text = extract_text_from_pdf(pdf_path)

            # Extract tables as text
            tables = extract_tables_from_pdf(pdf_path)

            # Extract text from images (OCR)
            ocr_text = extract_text_from_images(pdf_path)

            # Combine all extracted text
            combined_text = f"{text}\n\n{tables}\n\n{ocr_text}"

            # Split into chunks before storing in FAISS
            text_chunks = chunk_text(combined_text)

            if not text_chunks:
                print(f"⚠️ No valid text extracted from {pdf_file}. Skipping indexing.")
                continue  # Skip PDFs with no useful text

            # Store extracted data
            data_storage[pdf_file] = {
                "text": text,
                "tables": tables,
                "ocr_text": ocr_text,
                "chunks": text_chunks,
            }

            # Store chunks for FAISS
            all_text_chunks.extend(text_chunks)

    if not all_text_chunks:
        print("❌ No text chunks were extracted. FAISS index not updated.")
        return

    # Create FAISS index
    print("🔄 Updating FAISS index...")
    index = FAISS.from_texts(all_text_chunks, embedding_model)
    index.save_local(FAISS_INDEX_FILE)
    print(f"✅ FAISS index saved to {FAISS_INDEX_FILE}")

    # Save extracted data to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as json_file:
        json.dump(data_storage, json_file, indent=4)
    print(f"✅ Extracted data saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    process_pdfs()
