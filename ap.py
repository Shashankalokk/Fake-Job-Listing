from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import os

# ========== Initialize Flask app ========== 
app = Flask(__name__)

# ========== Load Model ========== 
MODEL_PATH = "bert_fraud_detection.pth"
print(f"Looking for model at: {MODEL_PATH}")

try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    tokenizer = None

# ========== Helper Functions ========== 
def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def chunk_text(text, words_per_chunk=5):
    """Break text into chunks of approximately words_per_chunk words each."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk = ' '.join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    
    return chunks

def predict_overall(text, suspicious_chunks):
    """
    Return overall prediction based on presence of suspicious chunks.
    If there are no suspicious chunks, the job is considered real.
    """
    # For production, you might want to use model prediction here
    
    # Modified logic: If no suspicious chunks, the job is real
    if not suspicious_chunks:
        return "Real", 0.95
    else:
        # If there are suspicious chunks, classify as fake
        # You might want to adjust confidence based on number and scores of suspicious chunks
        return "Fake", 0.95

def get_suspicious_chunks(chunks, threshold=0.6):
    """Identify suspicious chunks in the text."""
    # For testing, return suspicious chunks containing certain keywords
    # In production, uncomment the code below
    """
    suspicious = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.split()) < 2:  # Skip very short chunks
            continue
            
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            fake_prob = probs[0][1].item()
        
        if fake_prob > threshold:
            suspicious.append({
                "chunk_id": i,
                "text": chunk, 
                "fake_score": round(fake_prob, 3)
            })
    """
    suspicious = []
    keywords = ["urgent", "immediate", "guaranteed", "bank account", "$5,000", "limited", "money", "western union", "commission"]
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        for keyword in keywords:
            if keyword.lower() in chunk_lower:
                suspicious.append({
                    "chunk_id": i,
                    "text": chunk,
                    "fake_score": round(0.7 + 0.2 * (i % 3), 2)  # Fake varying scores
                })
                break
    
    # Sort by fake score in descending order
    suspicious.sort(key=lambda x: x["fake_score"], reverse=True)
    return suspicious

# ========== API Routes ========== 
@app.route("/predict", methods=["POST"])
def predict_job_status():
    """
    Predict whether a job listing is fake or real, and identify suspicious chunks.
    """
    # Get the data from the request
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    # Clean and prepare the text
    text = clean_text(data["text"])
    if not text:
        return jsonify({"error": "Empty or invalid text provided"}), 400
    
    # Break text into chunks
    chunks = chunk_text(text, words_per_chunk=6)
    
    # Get suspicious chunks first
    sus_chunks = get_suspicious_chunks(chunks)
    
    # Get overall prediction based on suspicious chunks
    status, confidence = predict_overall(text, sus_chunks)
    
    return jsonify({
        "job_status": status,
        "confidence_score": round(confidence, 3),
        "all_chunks": chunks,
        "suspicious_chunks": sus_chunks
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "API is operational"})

# ========== Run the API ========== 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
