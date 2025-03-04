import os
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline

load_dotenv()

app = FastAPI()

# Load summarization models
summarizer_1 = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer_2 = pipeline("summarization", model="t5-small")

@app.post("/summarize/")
async def summarize_pdf(file: UploadFile = File(...)):
    """Handles PDF uploads and returns a summary."""
    file_path = f"temp_{file.filename}"
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    
    max_chunk_length = 1024  
    chunks = [doc.page_content[:max_chunk_length] for doc in docs]

    summaries = []
    for chunk in chunks:
        input_length = len(chunk.split())  
        max_length = max(50, int(input_length * 0.5))
        min_length = max(25, int(max_length * 0.5))

        summary_1 = summarizer_1(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summary_2 = summarizer_2(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

        combined_summary = f"Model 1: {summary_1}\nModel 2: {summary_2}"
        summaries.append(combined_summary)

    os.remove(file_path)  # Clean up temp file

    return {"summary": "\n\n".join(summaries)}

@app.get("/")
def home():
    return {"message": "Welcome to the PDF Summarization API"}
