# DocuMind AI

An intelligent document processing system that extracts structured data from PDFs and images.

## Project Structure

- `app.py`: Streamlit frontend application.
- `main.py`: FastAPI backend application.
- `src/`: Core logic modules.
- `data/`: Data storage.
- `notebooks/`: Jupyter notebooks for experimentation.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Backend:
   ```bash
   uvicorn main:app --reload
   ```

3. Run the Frontend:
   ```bash
   streamlit run app.py
   ```
