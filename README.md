# DocuMind AI

An intelligent document processing system that extracts structured data from PDFs and images using OCR, NLP, and rule-based heuristics.

## 🚀 Features

-   **Multi-Format Support**: Processes PDF documents and images (JPG, PNG).
-   **Intelligent OCR**: Utilizes **Tesseract OCR (v5)** for robust text extraction suitable for diverse layouts.
-   **Specialized Extraction**:
    -   **Invoices**: Extracts Invoice IDs, Dates, Totals, Tax, Subtotals, and Vendor info using spatial context.
    -   **Receipts**: Captures Merchant names, Transaction Dates, Totals, and Auth Codes from noisy receipt images.
    -   **Resumes**: Parses contact info (Phone, Email, LinkedIn), Education, Work Experience, and Skills using section analysis and NER.
-   **Smart Validation & Correction**:
    -   **Auto-Correction**: Normalizes dates to ISO-8601, cleans up currency amounts (fixes common OCR errors like 'O' -> '0'), and formats phone numbers to E.164.
    -   **Logic Validation**: Cross-checks mathematical integrity (e.g., `Subtotal + Tax == Total`) and date logic (e.g., `Due Date >= Invoice Date`).
-   **Interactive UI**: Streamlit-based frontend for easy file upload, visualization, and real-time feedback.
-   **Export**: Download extracted and validated data as JSON.

## 🏗️ Architecture & Design Choices

### 1. Hybrid Extraction Strategy
We employ a tiered approach to extraction to maximize accuracy across different document types:
-   **Regex & Heuristics**: For highly structured fields like emails, dates, and phone numbers.
-   **Spatial Layout Analysis**: For documents like Invoices and Receipts where position matters (e.g., Total is usually at the bottom right).
-   **NLP / NER (spaCy)**: For unstructured text in Resumes to identify Organizations, Dates, and Person names.

### 2. OCR Engine: Tesseract 5
**Decision**: Pivoted from EasyOCR to Tesseract 5.
**Why**:
-   **Stability**: Tesseract is a mature, system-level dependency with predictable behavior, avoiding the "dependency hell" often seen with Python-only OCR wrappers.
-   **Performance**: FASTER processing for standard documents.
-   **Layout Analyis**: Tesseract's page segmentation modes (PSM) provide excellent block detection which is critical for parsing columns in resumes or tables in invoices.

### 3. Validation Layer
We believe extraction is only half the battle. Data must be *correct*.
-   **Field-Level**: Validates individual formats (Date ranges, Email structure).
-   **Cross-Field**: Ensures business logic holds true. If the extracted "Total" doesn't match the sum of line items, we flag it.

### 4. Auto-Corrector
OCR is never 100% perfect. Our `AutoCorrector` module handles common pitfalls:
-   **Confusables**: Replacing 'S' with '5' or 'O' with '0' in numeric fields.
-   **Format Normalization**: converting "Jan 5, 2023" to "2023-01-05".

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

    *Note: `en_core_web_lg` model for spaCy will be downloaded automatically if configured, otherwise run:*
    ```bash
    python -m spacy download en_core_web_lg
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
# Run full suite
pytest tests/
```

## 📂 Project Structure

```text
DocuMind.ai/
├── app.py                      # Streamlit Frontend
├── requirements.txt            # Python dependencies
├── src/
│   ├── pipeline.py             # Main Orchestration
│   ├── ocr/
│   │   └── tesseract_engine.py # Tesseract Wrapper
│   ├── preprocessing/
│   │   └── image_processor.py  # Image enhancement
│   ├── classification/
│   │   └── rule_based.py       # Document classifier
│   ├── extraction/
│   │   ├── regex_extractor.py               # Base regex tools
│   │   └── document_specific/
│   │       ├── invoice_extractor.py         # Invoice logic
│   │       ├── receipt_extractor.py         # Receipt logic
│   │       └── resume_extractor.py          # Resume logic (Hybrid)
│   └── validation/
│       ├── validators.py       # Validation Logic
│       └── auto_correct.py     # OCR Correction
└── tests/                      # Unit and Integration tests
```

## 🛡️ License
MIT
