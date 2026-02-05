# DocuMind AI

An intelligent document processing system that extracts structured data from PDFs and images using OCR and NLP techniques.

## 🚀 Features

-   **Multi-Format Support**: Processes PDF documents and images (JPG, PNG).
-   **Intelligent OCR**: Utilizes **Tesseract OCR** for robust text extraction.
-   **Automatic Classification**: Categorizes documents (Invoices, Receipts, Resumes, etc.) using rule-based heuristics.
-   **Structured Extraction**: Extracts key fields including:
    -   Emails & Phone numbers
    -   Dates (Normalized to ISO format)
    -   monetary Amounts
    -   Invoice Numbers & URLs
-   **Interactive UI**: Streamlit-based frontend for easy file upload, visualization, and validation.
-   **Export**: Download extracted data as JSON or CSV.

## 🛠️ Architecture & Tech Stack

-   **Frontend**: Streamlit
-   **Backend**: FastAPI (In progress)
-   **Core Modules**:
    -   `src.preprocessing`: OpenCV-based image enhancement (deskewing, denoising).
    -   `src.ocr`: **Tesseract Engine** (chosen for stability).
    -   `src.classification`: Rule-based classifier (keyword/regex patterns).
    -   `src.extraction`: Regex-based field extractor with confidence scoring.

### 💡 Architectural Decision: OCR Engine Pivot

> **Note on OCR Engine:**
> Initially, this project utilized **EasyOCR**. However, we encountered persistent compatibility issues with `python-bidi` and newer Python versions (3.12+), leading to unavoidable `ModuleNotFoundError` crashes in production environments.
>
> **Decision:** We pivoted to **Tesseract OCR (`pytesseract`)**.
> **Reasoning:** Tesseract provides a more stable, system-level dependency that does not rely on the fragile `bidi` library, ensuring reliable deployments and easier CI/CD integration.

## 📦 Installation

### Prerequisites

You must have **Tesseract OCR** installed on your system.

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sam-Orion/DocuMind.ai.git
    cd DocuMind.ai
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### Running the Frontend (Streamlit)
Launch the interactive web interface:
```bash
streamlit run app.py
```
Upload a document to see the processing pipeline in action.

### Running Tests
Verify the installation and logic:
```bash
# Run unit tests (mocked Tesseract)
pytest tests/test_ocr.py tests/test_classification.py tests/test_extraction.py

# Run LIVE integration test (requires Tesseract binary installed)
pytest tests/test_tesseract_live.py
```

## 📂 Project Structure

```text
DocuMind.ai/
├── app.py                  # Streamlit Frontend
├── main.py                 # FastAPI Backend (stub)
├── requirements.txt        # Python dependencies
├── src/
│   ├── pipeline.py         # Main orchestration logic
│   ├── ocr/
│   │   └── tesseract_engine.py # Tesseract implementation
│   ├── processing/
│   │   └── image_processor.py  # Image enhancement
│   ├── classification/
│   │   └── rule_based.py       # Document classifier
│   └── extraction/
│       └── regex_extractor.py  # Data extractor
└── tests/                  # Unit and Integration tests
```
