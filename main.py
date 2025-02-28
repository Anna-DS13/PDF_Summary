import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline

load_dotenv()

def summarize_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    max_chunk_length = 1024  
    chunks = [doc.page_content[:max_chunk_length] for doc in docs]

 
    summarizer_1 = pipeline("summarization", model="facebook/bart-large-cnn")
    summarizer_2 = pipeline("summarization", model="t5-small")  

    summaries = []
    for chunk in chunks:
        input_length = len(chunk.split())  
        max_length = max(50, int(input_length * 0.5))
        min_length = max(25, int(max_length * 0.5))

        summary_1 = summarizer_1(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summary_2 = summarizer_2(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

        combined_summary = f"Model 1: {summary_1}\nModel 2: {summary_2}"
        summaries.append(combined_summary)

    return "\n\n".join(summaries)

if __name__ == '__main__':
    summary = summarize_pdf('DS.pdf')

    print('Summary:')
    print(summary)
