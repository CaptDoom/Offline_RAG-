"""Multi-format document loader for images, PDFs, and Git repositories.

Implements:
A. IMAGES (JPEG, PNG, TIFF):
   - Extract text via EasyOCR (handwriting) + Tesseract (print)
   - Auto-rotate and deskew using OpenCV
   - Display extracted text preview before indexing

B. PDFs (scanned + digital):
   - Scanned: pdf2image → OCR each page
   - Digital: PyPDF2 direct text extraction (5x faster)
   - Auto-detect type and choose method
   - Merge multi-page results into single document

C. Git REPOSITORIES:
   - Accept GitHub URL or local .git folder
   - Crawl and index: README.md, all .py, .ipynb, .txt files
   - Ignore: .git, __pycache__, node_modules, .env
   - Parse Jupyter notebooks (extract markdown + code cells separately)
   - For each repo, create a summary chunk

Common:
- Timeout: 120 seconds per operation
- Retry logic: 3 attempts with exponential backoff
- Validation: Skip corrupted files with warning (don't crash)
- Logging: Write all errors to local_archive.log with timestamps
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from local_archive_ai.logging_config import log
from local_archive_ai.services import (
    _sanitize_text,
    _split_with_tokens,
)

try:
    from PIL import Image, ImageFilter
except Exception:  # pragma: no cover
    Image = None
    ImageFilter = None

try:
    import cv2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    cv2 = None

try:
    import pytesseract  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    pytesseract = None

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT = 120
_MAX_RETRIES = 3
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_REPO_INDEX_EXTENSIONS = {".md", ".py", ".ipynb", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
_REPO_IGNORE_DIRS = {".git", "__pycache__", "node_modules", ".env", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build", ".tox", ".eggs"}
_SCANNED_PDF_TEXT_THRESHOLD = 50  # chars per page below which PDF is considered scanned


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LoadResult:
    """Result from loading a single document."""
    file_path: str
    file_name: str
    file_type: str  # 'image', 'pdf_digital', 'pdf_scanned', 'jupyter', 'text', 'repo_summary'
    text: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    preview: str = ""  # Short text preview
    status: str = "success"  # 'success', 'warning', 'error'
    error_message: str = ""
    processing_time: float = 0.0
    ocr_engine_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepoInfo:
    """Metadata about a Git repository."""
    name: str
    path: str
    url: str = ""
    main_language: str = ""
    file_count: int = 0
    description: str = ""


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def _retry(fn, max_retries: int = _MAX_RETRIES, operation: str = "operation"):
    """Execute fn with exponential backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            wait = min(2 ** attempt, 30)
            log.warning("%s failed (attempt %d/%d): %s", operation, attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
def _configure_tesseract() -> None:
    """Configure tesseract command path."""
    if pytesseract is None:
        return
    try:
        default_exe = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default_exe.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default_exe)
    except Exception:
        pass


def _auto_rotate_deskew(img: Any) -> Any:
    """Auto-rotate and deskew image using OpenCV.

    Steps:
    1. Convert to grayscale
    2. Detect skew angle using Hough line transform
    3. Rotate to correct skew
    4. Apply denoising
    """
    if img is None or cv2 is None:
        return img

    try:
        # Convert PIL Image to numpy array
        img_array = np.array(img)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()

        # Detect edges for skew detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use Hough lines to detect dominant angle
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines
                if abs(angle) < 45:
                    angles.append(angle)

            if angles:
                median_angle = float(np.median(angles))
                # Only correct if skew is significant (> 0.5 degrees) but not extreme
                if 0.5 < abs(median_angle) < 15:
                    h, w = gray.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img_array = cv2.warpAffine(
                        img_array, rotation_matrix, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

        # Denoise
        if len(img_array.shape) == 2:
            img_array = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
        elif len(img_array.shape) == 3:
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

        return Image.fromarray(img_array)
    except Exception as exc:
        log.debug("Auto-rotate/deskew failed: %s", exc)
        return img


def _preprocess_for_ocr(img: Any) -> Any:
    """Preprocess image for better OCR quality.

    Steps: resize to ~300 DPI equivalent, adaptive threshold, sharpen.
    """
    if img is None or Image is None:
        return img

    try:
        # Convert to grayscale
        if img.mode != "L":
            gray = img.convert("L")
        else:
            gray = img.copy()

        # Resize small images
        width, height = gray.size
        if width < 1500:
            scale = max(2, 3000 // max(width, 1))
            gray = gray.resize((width * scale, height * scale), Image.LANCZOS)

        # Sharpen
        if ImageFilter is not None:
            gray = gray.filter(ImageFilter.SHARPEN)

        # OpenCV adaptive threshold
        if cv2 is not None:
            arr = np.array(gray)
            arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
            gray = Image.fromarray(arr)

        return gray
    except Exception:
        return img


# ---------------------------------------------------------------------------
# MultiFormatLoader
# ---------------------------------------------------------------------------
class MultiFormatLoader:
    """Load documents from multiple formats: images, PDFs, Git repositories.

    Usage::

        loader = MultiFormatLoader()

        # Load an image
        result = loader.load_image(Path("scan.jpg"))

        # Load a PDF (auto-detect scanned vs digital)
        result = loader.load_pdf(Path("document.pdf"))

        # Load a Git repo
        results = loader.load_git_repo("https://github.com/user/repo")
    """

    def __init__(
        self,
        chunk_size: int = 500,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
        ocr_engine: str = "tesseract",
    ) -> None:
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.ocr_engine = ocr_engine
        _configure_tesseract()
        log.info("MultiFormatLoader initialised (chunk_size=%d, ocr=%s)", chunk_size, ocr_engine)

    # ==================================================================
    # A. IMAGE LOADING
    # ==================================================================
    def load_image(self, path: Path) -> LoadResult:
        """Extract text from image using EasyOCR (handwriting) + Tesseract (print).

        Steps:
        1. Open and validate image
        2. Auto-rotate and deskew with OpenCV
        3. OCR with primary engine (Tesseract for print)
        4. Fallback to EasyOCR for handwriting if primary fails
        5. Generate text preview
        6. Split into chunks
        """
        t0 = time.time()
        path = Path(path).resolve()

        if not path.exists():
            return LoadResult(
                file_path=str(path), file_name=path.name, file_type="image",
                text="", status="error", error_message="File not found",
                processing_time=time.time() - t0,
            )

        if path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return LoadResult(
                file_path=str(path), file_name=path.name, file_type="image",
                text="", status="error", error_message=f"Unsupported image format: {path.suffix}",
                processing_time=time.time() - t0,
            )

        ocr_engine_used = ""
        text = ""
        errors: list[str] = []

        try:
            if Image is None:
                raise RuntimeError("Pillow is required for image processing.")

            with Image.open(path) as img:
                # Auto-rotate and deskew
                processed = _auto_rotate_deskew(img.copy())
                preprocessed = _preprocess_for_ocr(processed)

                # Try Tesseract first (better for printed text)
                if pytesseract is not None:
                    try:
                        text = pytesseract.image_to_string(preprocessed, timeout=30)
                        ocr_engine_used = "tesseract"
                    except Exception as exc:
                        errors.append(f"tesseract: {exc}")
                        text = ""

                # Fallback to EasyOCR (better for handwriting)
                if not text.strip() and easyocr is not None:
                    try:
                        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                        result = reader.readtext(str(path), detail=0)
                        text = "\n".join(result)
                        ocr_engine_used = "easyocr"
                    except Exception as exc:
                        errors.append(f"easyocr: {exc}")

                # If both engines have results, combine them
                if not text.strip():
                    text = ""
                    if errors:
                        log.warning("OCR failed for %s: %s", path.name, "; ".join(errors))

        except Exception as exc:
            return LoadResult(
                file_path=str(path), file_name=path.name, file_type="image",
                text="", status="error", error_message=str(exc),
                processing_time=time.time() - t0,
            )

        text = _sanitize_text(text, max_length=50000)
        preview = text[:500] + ("..." if len(text) > 500 else "")

        # Split into chunks
        chunks = self._text_to_chunks(text, str(path), path.name, file_type="image")

        status = "success" if text.strip() else "warning"
        return LoadResult(
            file_path=str(path), file_name=path.name, file_type="image",
            text=text, chunks=chunks, preview=preview,
            status=status,
            error_message="" if text.strip() else "No text extracted from image",
            processing_time=time.time() - t0,
            ocr_engine_used=ocr_engine_used,
        )

    # ==================================================================
    # B. PDF LOADING
    # ==================================================================
    def _is_scanned_pdf(self, path: Path) -> bool:
        """Auto-detect if a PDF is scanned (image-based) vs digital (text-based).

        Heuristic: If average text per page is below threshold, it's likely scanned.
        """
        if PdfReader is None:
            return True  # Assume scanned if we can't check

        try:
            reader = PdfReader(str(path))
            if not reader.pages:
                return True

            total_text = 0
            pages_checked = min(len(reader.pages), 5)  # Check first 5 pages
            for page in reader.pages[:pages_checked]:
                page_text = page.extract_text() or ""
                total_text += len(page_text.strip())

            avg_chars = total_text / max(pages_checked, 1)
            return avg_chars < _SCANNED_PDF_TEXT_THRESHOLD
        except Exception:
            return True  # Assume scanned on error

    def _load_pdf_digital(self, path: Path) -> tuple[str, str]:
        """Extract text from digital PDF using PyPDF2 (fast path)."""
        errors: list[str] = []

        # Try pypdf first (fastest)
        if PdfReader is not None:
            try:
                reader = PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(pages)
                if text.strip():
                    return _sanitize_text(text, max_length=100000), "pypdf-digital"
            except Exception as exc:
                errors.append(f"pypdf: {exc}")

        # Fallback to pdfplumber
        if pdfplumber is not None:
            try:
                with pdfplumber.open(str(path)) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                text = "\n\n".join(pages)
                if text.strip():
                    return _sanitize_text(text, max_length=100000), "pdfplumber-digital"
            except Exception as exc:
                errors.append(f"pdfplumber: {exc}")

        raise RuntimeError(f"Digital PDF extraction failed: {'; '.join(errors)}")

    def _load_pdf_scanned(self, path: Path) -> tuple[str, str]:
        """Extract text from scanned PDF: convert pages to images → OCR each page."""
        page_texts: list[str] = []
        ocr_engine = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert PDF to images using pdftoppm
            try:
                result = subprocess.run(
                    ["pdftoppm", "-png", "-r", "200", str(path), f"{temp_dir}/page"],
                    capture_output=True, text=True, timeout=self.timeout,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"pdftoppm failed: {result.stderr[:200]}")

                png_files = sorted(Path(temp_dir).glob("page-*.png"))
                for png_file in png_files:
                    img_result = self.load_image(png_file)
                    if img_result.text.strip():
                        page_texts.append(img_result.text)
                        if not ocr_engine:
                            ocr_engine = img_result.ocr_engine_used

            except FileNotFoundError:
                # pdftoppm not available – try alternative with pypdf + OCR
                log.warning("pdftoppm not found, using fallback PDF-to-image conversion")
                try:
                    if PdfReader is not None:
                        reader = PdfReader(str(path))
                        for i, page in enumerate(reader.pages):
                            # Try extracting any text first
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                page_texts.append(page_text)
                except Exception as exc:
                    log.warning("Fallback PDF extraction failed: %s", exc)

            except subprocess.TimeoutExpired:
                raise RuntimeError("PDF-to-image conversion timed out")

        if not page_texts:
            raise RuntimeError("No text extracted from scanned PDF")

        merged = "\n\n".join(page_texts)
        return _sanitize_text(merged, max_length=100000), f"pdf2image-{ocr_engine or 'ocr'}"

    def load_pdf(self, path: Path) -> LoadResult:
        """Load PDF with auto-detection of scanned vs digital.

        Steps:
        1. Validate file
        2. Auto-detect: scanned or digital
        3. Digital: PyPDF2 direct extraction (5x faster)
        4. Scanned: pdf2image → OCR each page
        5. Merge multi-page results
        6. Split into chunks
        """
        t0 = time.time()
        path = Path(path).resolve()

        if not path.exists():
            return LoadResult(
                file_path=str(path), file_name=path.name, file_type="pdf_digital",
                text="", status="error", error_message="File not found",
                processing_time=time.time() - t0,
            )

        try:
            is_scanned = self._is_scanned_pdf(path)
            file_type = "pdf_scanned" if is_scanned else "pdf_digital"

            if is_scanned:
                log.info("PDF detected as scanned: %s", path.name)
                text, ocr_engine = _retry(
                    lambda: self._load_pdf_scanned(path),
                    max_retries=self.max_retries,
                    operation=f"scanned_pdf_{path.name}",
                )
            else:
                log.info("PDF detected as digital: %s", path.name)
                text, ocr_engine = _retry(
                    lambda: self._load_pdf_digital(path),
                    max_retries=self.max_retries,
                    operation=f"digital_pdf_{path.name}",
                )

        except Exception as exc:
            return LoadResult(
                file_path=str(path), file_name=path.name,
                file_type="pdf_scanned" if is_scanned else "pdf_digital",
                text="", status="error", error_message=str(exc),
                processing_time=time.time() - t0,
            )

        preview = text[:500] + ("..." if len(text) > 500 else "")
        chunks = self._text_to_chunks(text, str(path), path.name, file_type=file_type)

        return LoadResult(
            file_path=str(path), file_name=path.name, file_type=file_type,
            text=text, chunks=chunks, preview=preview,
            status="success" if text.strip() else "warning",
            error_message="" if text.strip() else "No text extracted from PDF",
            processing_time=time.time() - t0,
            ocr_engine_used=ocr_engine,
        )

    # ==================================================================
    # C. GIT REPOSITORY LOADING
    # ==================================================================
    def _clone_repo(self, url: str, dest: Path) -> Path:
        """Clone a Git repository to a temporary directory."""
        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", url, str(dest)],
                capture_output=True, text=True, timeout=self.timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed: {result.stderr[:300]}")
            return dest
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"git clone timed out after {self.timeout}s")
        except FileNotFoundError:
            raise RuntimeError("git is not installed")

    def _get_repo_info(self, repo_path: Path, url: str = "") -> RepoInfo:
        """Gather repository metadata."""
        name = repo_path.name

        # Count files by extension to determine main language
        ext_counts: dict[str, int] = {}
        file_count = 0
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in _REPO_IGNORE_DIRS]
            for fname in filenames:
                file_count += 1
                ext = Path(fname).suffix.lower()
                if ext:
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1

        # Determine main language from extensions
        lang_map = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".java": "Java", ".cpp": "C++", ".c": "C", ".go": "Go",
            ".rs": "Rust", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
        }
        main_lang = "Unknown"
        max_count = 0
        for ext, count in ext_counts.items():
            if ext in lang_map and count > max_count:
                max_count = count
                main_lang = lang_map[ext]

        # Try to read description from README
        description = ""
        for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = repo_path / readme_name
            if readme_path.exists():
                try:
                    content = readme_path.read_text(encoding="utf-8", errors="ignore")
                    # Extract first paragraph
                    lines = content.strip().split("\n")
                    desc_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#") and not stripped.startswith("="):
                            desc_lines.append(stripped)
                            if len(desc_lines) >= 3:
                                break
                    description = " ".join(desc_lines)[:200]
                except Exception:
                    pass
                break

        return RepoInfo(
            name=name, path=str(repo_path), url=url,
            main_language=main_lang, file_count=file_count,
            description=description,
        )

    def _parse_jupyter_notebook(self, path: Path) -> str:
        """Parse Jupyter notebook – extract markdown and code cells separately."""
        try:
            content = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            log.warning("Failed to parse notebook %s: %s", path.name, exc)
            return ""

        cells = content.get("cells", [])
        parts: list[str] = []

        for cell in cells:
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))

            if cell_type == "markdown":
                parts.append(f"[MARKDOWN]\n{source}")
            elif cell_type == "code":
                parts.append(f"[CODE]\n{source}")
            # Skip raw cells

        return "\n\n".join(parts)

    def _collect_repo_files(self, repo_path: Path) -> list[Path]:
        """Collect indexable files from a repository, respecting ignore patterns."""
        files: list[Path] = []
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in _REPO_IGNORE_DIRS]
            root_path = Path(root)
            for fname in filenames:
                fpath = root_path / fname
                suffix = fpath.suffix.lower()
                # Include README regardless of extension
                if fname.upper().startswith("README") or suffix in _REPO_INDEX_EXTENSIONS:
                    files.append(fpath)
        return sorted(files)

    def load_git_repo(self, url_or_path: str) -> list[LoadResult]:
        """Load a Git repository for indexing.

        Accepts:
        - GitHub URL (https://github.com/user/repo)
        - Local path to .git folder or repo directory

        Returns a list of LoadResult, one per file + a summary chunk.
        """
        t0 = time.time()
        results: list[LoadResult] = []
        cleanup_dir: str | None = None

        try:
            path = Path(url_or_path)
            is_url = url_or_path.startswith("http://") or url_or_path.startswith("https://") or url_or_path.startswith("git@")

            if is_url:
                # Clone to temp directory
                temp_dir = tempfile.mkdtemp(prefix="local_archive_repo_")
                cleanup_dir = temp_dir
                repo_path = self._clone_repo(url_or_path, Path(temp_dir) / "repo")
            elif path.is_dir():
                # Check if it's a git repo (has .git folder)
                if (path / ".git").is_dir():
                    repo_path = path
                elif path.name == ".git":
                    repo_path = path.parent
                else:
                    repo_path = path  # Use as-is even without .git
            else:
                return [LoadResult(
                    file_path=url_or_path, file_name=url_or_path,
                    file_type="repo_summary", text="",
                    status="error", error_message=f"Invalid repository path or URL: {url_or_path}",
                    processing_time=time.time() - t0,
                )]

            # Gather repo info
            repo_info = self._get_repo_info(repo_path, url=url_or_path if is_url else "")

            # Create summary chunk
            summary_text = (
                f"Repository: {repo_info.name}\n"
                f"URL: {repo_info.url or 'local'}\n"
                f"Main language: {repo_info.main_language}\n"
                f"Total files: {repo_info.file_count}\n"
                f"Description: {repo_info.description or 'N/A'}\n"
            )
            results.append(LoadResult(
                file_path=str(repo_path), file_name=f"{repo_info.name}_summary",
                file_type="repo_summary", text=summary_text,
                chunks=[{
                    "source_file": str(repo_path),
                    "file_path": str(repo_path),
                    "file_name": f"{repo_info.name}_summary",
                    "source_page": None,
                    "chunk_index": 1,
                    "text": summary_text,
                }],
                preview=summary_text[:500],
                status="success",
                processing_time=0,
                metadata={"repo_name": repo_info.name, "main_language": repo_info.main_language},
            ))

            # Collect and process files
            files = self._collect_repo_files(repo_path)
            log.info("Repository %s: found %d indexable files", repo_info.name, len(files))

            for file_path in files:
                try:
                    suffix = file_path.suffix.lower()

                    if suffix == ".ipynb":
                        text = self._parse_jupyter_notebook(file_path)
                        file_type = "jupyter"
                    elif suffix == ".pdf":
                        pdf_result = self.load_pdf(file_path)
                        results.append(pdf_result)
                        continue
                    elif suffix in _IMAGE_EXTENSIONS:
                        img_result = self.load_image(file_path)
                        results.append(img_result)
                        continue
                    else:
                        text = file_path.read_text(encoding="utf-8", errors="ignore")
                        file_type = "text"

                    text = _sanitize_text(text, max_length=50000)
                    if not text.strip():
                        continue

                    # Compute relative path for better file_name
                    try:
                        rel_path = str(file_path.relative_to(repo_path))
                    except ValueError:
                        rel_path = file_path.name

                    chunks = self._text_to_chunks(text, str(file_path), rel_path, file_type=file_type)

                    results.append(LoadResult(
                        file_path=str(file_path), file_name=rel_path,
                        file_type=file_type, text=text, chunks=chunks,
                        preview=text[:500] + ("..." if len(text) > 500 else ""),
                        status="success",
                        processing_time=0,
                    ))

                except Exception as exc:
                    log.warning("Skipping repo file %s: %s", file_path.name, exc)
                    results.append(LoadResult(
                        file_path=str(file_path), file_name=file_path.name,
                        file_type="text", text="",
                        status="error", error_message=str(exc),
                    ))

        except Exception as exc:
            log.error("Repository loading failed: %s", exc)
            results.append(LoadResult(
                file_path=url_or_path, file_name=url_or_path,
                file_type="repo_summary", text="",
                status="error", error_message=str(exc),
                processing_time=time.time() - t0,
            ))

        finally:
            # Cleanup cloned repo
            if cleanup_dir:
                try:
                    shutil.rmtree(cleanup_dir, ignore_errors=True)
                except Exception:
                    pass

        total_time = time.time() - t0
        log.info(
            "Repository loaded: %d results in %.1fs",
            len(results), total_time,
        )
        return results

    # ==================================================================
    # Utility: auto-detect file type and load
    # ==================================================================
    def load_file(self, path: Path) -> LoadResult:
        """Auto-detect file type and load accordingly."""
        path = Path(path).resolve()
        suffix = path.suffix.lower()

        if suffix in _IMAGE_EXTENSIONS:
            return self.load_image(path)
        elif suffix == ".pdf":
            return self.load_pdf(path)
        elif suffix == ".ipynb":
            t0 = time.time()
            text = self._parse_jupyter_notebook(path)
            text = _sanitize_text(text, max_length=50000)
            chunks = self._text_to_chunks(text, str(path), path.name, file_type="jupyter")
            return LoadResult(
                file_path=str(path), file_name=path.name, file_type="jupyter",
                text=text, chunks=chunks,
                preview=text[:500] + ("..." if len(text) > 500 else ""),
                status="success" if text.strip() else "warning",
                processing_time=time.time() - t0,
            )
        else:
            # Plain text / code
            t0 = time.time()
            try:
                text = _sanitize_text(
                    path.read_text(encoding="utf-8", errors="ignore"),
                    max_length=50000,
                )
            except Exception as exc:
                return LoadResult(
                    file_path=str(path), file_name=path.name, file_type="text",
                    text="", status="error", error_message=str(exc),
                    processing_time=time.time() - t0,
                )
            chunks = self._text_to_chunks(text, str(path), path.name, file_type="text")
            return LoadResult(
                file_path=str(path), file_name=path.name, file_type="text",
                text=text, chunks=chunks,
                preview=text[:500] + ("..." if len(text) > 500 else ""),
                status="success" if text.strip() else "warning",
                processing_time=time.time() - t0,
            )

    # ==================================================================
    # Chunk helper
    # ==================================================================
    def _text_to_chunks(
        self, text: str, file_path: str, file_name: str, file_type: str = "text",
    ) -> list[dict[str, Any]]:
        """Split text into indexed chunks."""
        if not text.strip():
            return []

        raw_chunks = _split_with_tokens(text, self.chunk_size)
        chunks: list[dict[str, Any]] = []
        for idx, chunk_text in enumerate(raw_chunks, start=1):
            chunk_text = _sanitize_text(chunk_text)
            if not chunk_text:
                continue
            chunks.append({
                "source_file": file_path,
                "file_path": file_path,
                "file_name": file_name,
                "source_page": None,
                "chunk_index": idx,
                "text": chunk_text,
                "file_type": file_type,
            })
        return chunks
