import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import re

# Configure Gemini API
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyBxFG2RWw6yBa2_CIqTCrEXVfyMWfwBbZo")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')  # Upgraded for better reasoning

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
        # Fallback: regex for skill-like words
        skills = re.findall(r'\b(?:Python|SQL|Java|JavaScript|Spring|Hibernate|C\+\+|C#|Ruby|PHP)\b', query, re.I)
        return " ".join(skills[:10]) if skills else query

# Retrieval function
def retrieve_assessments(query, k=10, max_duration=None):
    query_lower = query.lower()
    wants_flexible = any(x in query_lower for x in ["untimed", "variable", "flexible"])
    processed_query = llm_shorten_query(query)
    st.write(f"Processed Query: {processed_query}")  # Debug
    query_embedding = embedding_model.encode([processed_query], show_progress_bar=False)[0]
    query_embedding = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_embedding, k * 2)
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = 1 - distances[0] / 2
    if max_duration is not None or wants_flexible:
        filtered = []
        for _, row in results.iterrows():
            duration = row["Assessment Length Parsed"]
            # Loosen filter: include if duration parsing fails
            if pd.isna(duration) or duration == "flexible duration":
                filtered.append(row)
            elif isinstance(duration, str) and "flexible" in duration.lower():
                filtered.append(row)
            elif isinstance(duration, float) and max_duration is not None and duration <= max_duration:
                filtered.append(row)
        results = pd.DataFrame(filtered) if filtered else results
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
        max_duration = float(re.search(r'(\d+)\s*min', query).group(1)) if "min" in query else None
        results = retrieve_assessments(query, k=10, max_duration=max_duration)
        st.write("### Recommended Assessments")
        st.table(results)
    else:
        st.warning("Please enter a query.")
