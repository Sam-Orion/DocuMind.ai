from fastapi import FastAPI

app = FastAPI(title="DocuMind AI API")

@app.get("/")
def read_root():
    return {"message": "Welcome to DocuMind AI API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
