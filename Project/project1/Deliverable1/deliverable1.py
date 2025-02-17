import os
import pandas as pd
import requests
import random
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

def rate_url_validity(user_query: str, url: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = " ".join([p.text for p in soup.find_all("p")])
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")  # Debug print
        return {  
            "Content Relevance": 0, 
            "Bias Score": 0, 
            "Final Validity Score": 0
        }

    if not page_text.strip():  # If no meaningful text is extracted
        print(f"Warning: No text extracted from {url}")  # Debug print
        return {  
            "Content Relevance": 0, 
            "Bias Score": 0, 
            "Final Validity Score": 0
        }

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    similarity_score = util.pytorch_cos_sim(model.encode(user_query), model.encode(page_text)).item() * 100

    sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    sentiment_result = sentiment_pipeline(page_text[:512])[0]
    bias_score = 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    final_score = (0.5 * similarity_score) + (0.5 * bias_score)

    print(f"Debug: Computed Final Score for {url} -> {final_score}")  # Debug print

    return {  
        "Content Relevance": similarity_score,
        "Bias Score": bias_score,
        "Final Validity Score": final_score
    }

# Ensure directory exists before saving CSV
desktop_path = os.path.expanduser("~/Desktop")
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

csv_file_path = os.path.join(desktop_path, "deliverable.csv")

sample_prompts = [
    "What are the symptoms of flu?",
    "How to bake a chocolate cake?",
    "Tell me about the history of Ancient Rome.",
    "What are the side effects of ibuprofen?",
    "Best exercises for weight loss?",
    "How do I improve my sleep quality?",
    "What are the latest trends in AI?",
    "How to prepare for a job interview?",
    "Explain the theory of relativity in simple terms.",
    "What are some beginner-friendly programming languages?",
    "How to save money effectively?",
    "Tell me about the benefits of meditation.",
    "What are the symptoms of COVID-19?",
    "How does blockchain technology work?",
    "What is quantum computing?"
]

sample_urls = [
    "https://www.bbc.com/news",
    "https://www.nytimes.com",
    "https://www.nature.com",
    "https://www.who.int/health-topics/coronavirus",
    "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
    "https://www.nasa.gov",
    "https://www.wikipedia.org",
    "https://www.python.org",
    "https://www.openai.com",
    "https://arxiv.org",
    "https://www.healthline.com",
    "https://www.sciencedaily.com",
    "https://www.nationalgeographic.com",
    "https://www.reuters.com",
    "https://www.economist.com"
]

num_rows = 10
new_data = {
    "user_prompt": random.sample(sample_prompts, num_rows),
    "url_to_check": random.sample(sample_urls, num_rows),
    "func_rating": [rate_url_validity(prompt, url).get("Final Validity Score", 0) for prompt, url in zip(sample_prompts[:num_rows], sample_urls[:num_rows])],
    "custom_rating": [random.randint(1, 5) for _ in range(num_rows)]
}

df = pd.DataFrame(new_data)

# Append new data to the uploaded CSV file
uploaded_csv_path = os.path.join(desktop_path, "simulated_deliverable.csv")
try:
    df_existing = pd.read_csv(uploaded_csv_path)
    df_combined = pd.concat([df_existing, df], ignore_index=True)
except FileNotFoundError:
    df_combined = df  # If file is missing, use new data only

df_combined.to_csv(csv_file_path, index=False)
print(f"Deliverable CSV file saved at: {csv_file_path}")
