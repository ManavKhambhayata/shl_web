import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# Configure Gemini API
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyBxFG2RWw6yBa2_CIqTCrEXVfyMWfwBbZo")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Load data and index
try:
    df = pd.read_csv("shl_catalog_with_summaries.csv")
    index = faiss.read_index("shl_assessments_index.faiss")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("Models loaded successfully!")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# LLM preprocessing function
def llm_shorten_query(query):
    prompt = "Extract all technical skills from query as space-separated list, max 10: "
    try:
        response = model.generate_content(prompt + query)
        shortened = response.text.strip()
        words = shortened.split()
        return " ".join(words[:10]) if words else query
    except Exception as e:
        st.error(f"Query LLM error: {e}")
        return query

# Simplified retrieval function
def retrieve_assessments(query, k=10):
    processed_query = llm_shorten_query(query)
    # st.write(f"Processed Query: {processed_query}")  # Debug
    query_embedding = embedding_model.encode([processed_query], show_progress_bar=False)[0]
    query_embedding = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_embedding, k)
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = 1 - distances[0] / 2
    results = results.rename(columns={"Pre-packaged Job Solutions": "Assessment Name", 
                                      "Assessment Length": "Duration"})
    return results[["Assessment Name", "URL", "Remote Testing (y/n)", 
                    "Adaptive/IRT (y/n)", "Duration", "Test Type"]].head(k)

# Streamlit UI
st.title("SHL Assessment Recommendation Engine")
st.write("Enter a query (e.g., 'Java developers, 40 mins').")
query = st.text_input("Your Query", "")
if st.button("Get Recommendations"):
    if query:
        results = retrieve_assessments(query, k=10)
        st.write("### Recommended Assessments")
        st.table(results)
    else:
        st.warning("Please enter a query.")
