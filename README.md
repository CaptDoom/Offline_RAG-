# Offline RAG

## Project Description
**Offline RAG (Local-Archive AI)** is a powerful, single-page, entirely offline Retrieval-Augmented Generation (RAG) dashboard designed for complete privacy and local execution. It enables users to index, search, and converse with their own local documents and images securely on their machine, without ever transmitting data to external APIs.

The system features robust ingestion capabilities, supporting everything from PDFs and code files to complex image semantic search via local vector stores and multi-engine OCR fallback chains. Answers and insights are generated utilizing self-hosted open-source language models managed by Ollama, backing up every claim with citation-linked evidence directly from your ingested local files.

### Key Features
- **Total Privacy & Local Execution**: Runs 100% locally with zero external API calls for embedding or inference.
- **Unified Web Dashboard**: A modern, responsive FastAPI-served UI that operates primarily as a Single Page Application (SPA).
- **Extensive Document Ingestion**: Process PDFs, text files, markdown, code files, zipped archives, and direct folders.
- **Image Semantic Search**: Ingest JPG, PNG, TIFF, and WEBP formats featuring OCR bounding block tracking and retrieval.
- **Fail-safe Indexing Engine**: Crash-proof processing with multi-tiered OCR failovers (`pytesseract`, `easyocr`, `pdfplumber`).
- **Deep Visibility & Debugging**: Visually verify vector embeddings, retrieval sim metrics (similarity scoring), latency details, and complete LLM prompts payload inspection real-time.
- **Batch Processing**: Run multiple programmatic queries and export insights in CSV files.

---

## Installation Instructions

### Prerequisites
Before you start, ensure you have the following installed on your machine:
- **Python**: Version `3.12.x` is validated and officially supported.
- **Ollama**: Follow the instructions at [ollama.com](https://ollama.com/) to install Ollama for entirely local inference.
- **Tesseract OCR**: Required for comprehensive document extraction if you intend to index images or scanned PDFs.

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/CaptDoom/Offline_RAG-.git
   cd Offline_RAG-
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Local Language Model**
   Start the Ollama service, then pull the required base model for generation workflows. We currently default to `llama3.2:1b` for swift, local responses:
   ```bash
   ollama pull llama3.2:1b
   ```

5. **Start the Application**
   Run the ASGI server application via Uvicorn. The `--reload` flag applies updates rapidly:
   ```bash
   uvicorn api:app --reload
   ```

6. **Access the Dashboard**
   Open your browser and navigate to `http://localhost:8000` to start exploring the Offline RAG pipeline!

---

## Technical Details

### Architecture & Workflows
- **Indexing Phase**: Reads diverse document structures natively and token-aware chunks them. Fallbacks use Tesseract OCR if text extraction meets silent failure criteria. 
- **Embeddings**: Employs Python `sentence-transformers` libraries against text and images (leveraging MiniLM or CLIP variants internally).
- **Vector Search Engine**: Embedded dimensions rely on high-performance Local `FAISS` indexes for near-instant geometric proximity searches.
- **Answer Generation**: Synthesizes the nearest document matches through an Ollama connection.

### Advanced Usage & CLI Resilience
The library features `index_documents_resilient` within `local_archive_ai.services` to trace every failed OCR pass and provide detailed indexing logs. It implements progress-bar handling and tracking success benchmarks seamlessly under the hood.

> **Note**: Python versions `3.13` or `3.14` may compile and run, but are treated as experimental and will log a warning during startup execution.
