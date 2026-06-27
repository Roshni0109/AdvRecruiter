import os
import sys
import tempfile
import pandas as pd
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import extract_candidate_features, JD_CORE_SKILLS
from src.feature_engineering import (
    load_embedding_model,
    compute_candidate_embeddings,
    compute_all_features,
    JD_TEXT
)
from src.ranker import compute_final_score

# --- Cache / State variables ---
embedding_model = None
candidate_pool = []
pool_embeddings = None
original_features_df = None

# --- Helper functions ---
def initialize_app():
    global embedding_model, candidate_pool, pool_embeddings, original_features_df
    
    # Load embedding model
    embedding_model = load_embedding_model()
    
    # Check if precomputed features exist
    features_path = os.path.join("outputs", "features.parquet")
    if os.path.exists(features_path):
        print(f"Loading precomputed features from {features_path}...")
        original_features_df = pd.read_parquet(features_path)
        original_features_df = original_features_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    else:
        print("Warning: No precomputed features found.")
        original_features_df = pd.DataFrame()
        
    # Load raw candidate records (up to 10k for memory limits on Hugging Face CPU Space)
    candidates_path = os.path.join("data", "candidates.jsonl")
    if not os.path.exists(candidates_path):
        # Fallback to absolute testing path
        candidates_path = "D:/AI Recruiter Hackathon/Testing Data/candidates.jsonl"
        
    if os.path.exists(candidates_path):
        import json
        print(f"Loading candidate records from: {candidates_path}...")
        with open(candidates_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                line = line.strip()
                if line:
                    candidate_pool.append(extract_candidate_features(json.loads(line)))
        print(f"Loaded {len(candidate_pool)} candidates for custom search.")
        
    # Check if 100k memmap embeddings exist, otherwise generate them for the pool on the fly
    mmap_path = os.path.join("outputs", "embeddings.mmap")
    if os.path.exists(mmap_path):
        print(f"Loading memmap embeddings from {mmap_path}...")
        pool_embeddings = np.memmap(mmap_path, dtype="float32", mode="r", shape=(100000, 384))
        if len(candidate_pool) < 100000:
            pool_embeddings = pool_embeddings[:len(candidate_pool)]
    elif len(candidate_pool) > 0:
        print("embeddings.mmap not found. Generating embeddings on the fly...")
        texts = [cand.get("career_text", "") for cand in candidate_pool]
        # Encode inputs (fast for 10k candidates on CPU, ~2 mins)
        pool_embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print("Embeddings generated successfully.")

# Run initialization
initialize_app()

def rank_existing_database(custom_jd):
    global pool_embeddings, candidate_pool, embedding_model
    
    if not candidate_pool or pool_embeddings is None:
        err_df = pd.DataFrame([{"Error": "Candidate database not loaded on Hugging Face Space."}])
        return err_df, None
        
    # Embed custom JD
    custom_jd_emb = embedding_model.encode(custom_jd, convert_to_numpy=True)
    
    # Recompute feature scores
    df = compute_all_features(candidate_pool, pool_embeddings, custom_jd_emb)
    ranked_df = compute_final_score(df)
    
    results_df = make_results_table(ranked_df)
    
    # Save a temporary CSV for download
    csv_path = "ranked_database_candidates.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    
    return results_df, csv_path

def rank_uploaded_resumes(file_obj, custom_jd):
    global embedding_model
    
    if file_obj is None:
        err_df = pd.DataFrame([{"Error": "Please upload a candidate JSON/JSONL file."}])
        return err_df, None
        
    import json
    candidates = []
    
    try:
        # Check file extension and read content
        with open(file_obj.name, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if file_obj.name.endswith(".jsonl"):
            for line in content.split("\n"):
                if line.strip():
                    candidates.append(extract_candidate_features(json.loads(line)))
        else: # assume regular JSON array
            raw_list = json.loads(content)
            if isinstance(raw_list, list):
                candidates = [extract_candidate_features(raw) for raw in raw_list]
            else:
                candidates = [extract_candidate_features(raw_list)]
    except Exception as e:
        err_df = pd.DataFrame([{"Error": f"Error reading file: {str(e)}"}])
        return err_df, None
        
    if not candidates:
        err_df = pd.DataFrame([{"Error": "No valid candidates found in file."}])
        return err_df, None
        
    # Generate text for embedding
    texts = []
    for cand in candidates:
        texts.append(cand.get("career_text", ""))
        
    # Embed uploaded candidates
    uploaded_embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    custom_jd_emb = embedding_model.encode(custom_jd, convert_to_numpy=True)
    
    df = compute_all_features(candidates, uploaded_embeddings, custom_jd_emb)
    ranked_df = compute_final_score(df)
    
    results_df = make_results_table(ranked_df)
    
    # Save a temporary CSV for download
    csv_path = "ranked_uploaded_candidates.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    
    return results_df, csv_path

from scripts.rank_candidates import generate_reasoning

def make_results_table(ranked_df):
    results = []
    for _, row in ranked_df.head(100).iterrows():
        # Use the spec reasoning generator function
        reasoning = generate_reasoning(row)
        
        results.append({
            "candidate_id": row['candidate_id'],
            "rank": int(row['rank']),
            "score": f"{row['final_score']:.4f}",
            "reasoning": reasoning
        })
    return pd.DataFrame(results)

# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AdvRecruiter - Candidate Ranking Dashboard")
    gr.Markdown("Rank candidates against any Job Description using semantic similarity and behavioral scoring.")
    
    with gr.Tab("Mode 1: Rank Existing Pool"):
        gr.Markdown("### Match the existing candidate database against a custom JD")
        with gr.Row():
            with gr.Column(scale=1):
                jd_input_1 = gr.Textbox(value=JD_TEXT, label="Job Description", lines=8)
                btn_db = gr.Button("Rank Database Candidates", variant="primary")
                download_db = gr.File(label="Download Ranked CSV Output")
            with gr.Column(scale=2):
                results_db = gr.Dataframe(label="Top Ranked Candidates (100k database)")
                
        btn_db.click(rank_existing_database, inputs=[jd_input_1], outputs=[results_db, download_db])
        
    with gr.Tab("Mode 2: Upload & Rank New Resumes"):
        gr.Markdown("### Upload a new JSON/JSONL candidate list and score them")
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload Candidate JSON/JSONL", file_count="single")
                jd_input_2 = gr.Textbox(value=JD_TEXT, label="Job Description", lines=5)
                btn_upload = gr.Button("Score & Rank Resumes", variant="primary")
                download_upload = gr.File(label="Download Ranked CSV Output")
            with gr.Column(scale=2):
                results_upload = gr.Dataframe(label="Ranked Results (New Uploads)")
                
        btn_upload.click(rank_uploaded_resumes, inputs=[file_input, jd_input_2], outputs=[results_upload, download_upload])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
