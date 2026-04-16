# Local-Archive AI (Offline Streamlit App)

Single-page offline RAG dashboard for local document indexing, retrieval, and pipeline observability.

## Quick start

1. Use Python `3.12.x` for the supported runtime.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Ensure Ollama is running locally and the configured model is available:
   - `ollama pull llama3.2:1b`
5. Run the canonical Streamlit app:
   - `streamlit run app.py`
6. Optional secondary client:
   - `uvicorn api:app --reload`

## Runtime notes

- The project is validated on Python `3.12`.
- Python `3.14` may run in development, but it is treated as unsupported and will show a warning in the UI.
- Secrets are env-only; `config.yaml` is intentionally written without API keys.

## Secondary Web Client

- The FastAPI-served static client remains available at `/`.
- It is kept as a lightweight compatibility surface for `chat`, `debug`, and `batch`.
- The Streamlit app is the source of truth for the full product flow.

## Offline Image Semantic Search

A new lightweight Streamlit entrypoint is available for image-only semantic search:

- `streamlit run image_search.py`
- Supports JPG, PNG, TIFF, BMP, and WEBP image ingestion
- Extracts text with spatial OCR blocks and chunks them by proximity + token budget
- Embeds chunks locally with `sentence-transformers`
- Stores vectors and metadata in FAISS for natural language search
- No external OpenAI/Astra APIs required for retrieval

## Frontend navigation map

Top-level sections:

- `LANDING`: product overview, quick actions, and system snapshot
- `PIPELINE`: interactive end-to-end RAG flow with stage details and diagnostics
- `CHAT_ENGINE`: query execution with citation-backed responses
- `DEBUG_LOGS`: embedding/retrieval inspection and prompt visibility
- `BATCH_QUEUE`: multi-query execution and CSV export

Route flow:

- `LANDING` -> `PIPELINE`
- `LANDING` -> `CHAT_ENGINE`
- `LANDING` -> `DEBUG_LOGS`
- `LANDING` -> `BATCH_QUEUE`
- `PIPELINE` -> `CHAT_ENGINE`

## Configuration

- UI settings persist to `config.yaml`.
- Environment variables:
  - `OLLAMA_API_KEY`: Ollama API key (optional, overrides config)
  - `OLLAMA_ENDPOINT`: Ollama endpoint URL (optional, overrides config)
- Config validation: Run `python app.py --check-config` to validate config before startup.
- Secrets are never saved to `config.yaml`; use environment variables for sensitive data.
- Unknown keys in `config.yaml` are rejected during validation.

## Ingestion Resilience Agent

The Ingestion Resilience Agent provides crash-proof document processing with multi-engine OCR fallback chains and comprehensive error reporting.

### Features

- **Multi-Engine OCR Chain**: pytesseract → easyocr → pdfplumber → fail with detailed error report
- **File Hash Tracking**: Automatically skips unchanged files on re-indexing for efficiency
- **Per-File Status Tracking**: SUCCESS/WARNING/ERROR/SKIPPED status for each document
- **Progress Bars**: Visual progress tracking for batch folder processing
- **Zero Silent Failures**: All processing attempts are logged with detailed error messages

### OCR Processing Chain

1. **PDF Files**:
   - Primary: pypdf (text extraction)
   - Fallback: pdfplumber (text extraction)
   - Final Fallback: Convert to images → OCR chain

2. **Image Files** (jpg, png, bmp, tiff, webp):
   - Primary: pytesseract
   - Fallback: easyocr
   - Fail: Detailed error report

3. **Text Files** (txt, md, json, csv, code files):
   - Direct text reading with encoding detection

### Usage

The resilient ingestion is automatically used when indexing documents. For programmatic access:

```python
from local_archive_ai.services import index_documents_resilient, get_ingestion_report

# Index with resilience and progress tracking
report = index_documents_resilient(
    folder_path="/path/to/documents",
    chunk_size=500,
    store_path="data/faiss_index",
    use_progress_bar=True
)

# Check results
print(f"Success rate: {report.success_rate:.1f}%")
print(f"Files processed: {report.processed_files}/{report.total_files}")

# Load detailed report later
report = get_ingestion_report("data/faiss_index")
```

### Success Metrics

- **Zero Silent Failures**: All processing failures are logged and reported
- **95% Document Extraction Success Rate**: Target success rate for document processing
- **Detailed Error Reporting**: Each failed file includes specific error messages and attempted methods
- AC-3 indexing stress benchmark (50 PDFs / 200 pages generated locally)

> Note: The validation suite is designed to confirm offline retrieval behavior without external model downloads during query execution. Ensure Ollama and required local models are already available before running.
