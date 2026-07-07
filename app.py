# app.py
# ─────────────────────────────────────────────────────────────────────
# PURPOSE: UI / API entrypoint for the AI Recruiter application
#
# This file is responsible ONLY for the Gradio UI interface, invoking
# normalization and feature scoring from the src/ modules.
# ─────────────────────────────────────────────────────────────────────

import os
import sys
import pandas as pd
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import extract_candidate_features, load_candidates
from src.normalization import normalize_job_description, read_jd_file
from src.feature_engineering import (
    load_embedding_model,
    compute_candidate_embeddings,
    compute_all_features,
    JD_TEXT
)
from src.ranker import compute_final_score
from scripts.rank_candidates import generate_reasoning

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
        
    # Load raw candidate records (up to 10k for memory limits)
    candidates_path = os.path.join("data", "candidates.jsonl")
    if not os.path.exists(candidates_path):
        candidates_path = "D:/AI Recruiter Hackathon/Testing Data/candidates.jsonl"
        
    if os.path.exists(candidates_path):
        print(f"Loading candidate records from: {candidates_path}...")
        candidate_pool = load_candidates(candidates_path, limit=10000)
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
        texts = [cand.get("full_text", "") for cand in candidate_pool]
        # Encode inputs (fast for 10k candidates on CPU, ~2 mins)
        pool_embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print("Embeddings generated successfully.")

# Run initialization
initialize_app()

# Preset Job Descriptions to simplify quick testing
PRESET_JDS = {
    "Senior AI Engineer (Default)": JD_TEXT,
    "Data Scientist": "Looking for a Data Scientist with 4+ years of experience in ML, statistical modeling, and predictive analytics. Proficient in Python, SQL, pandas, and scikit-learn. Experience with PyTorch or TensorFlow is preferred. Location: Noida/Pune.",
    "Backend Engineer": "We are seeking a Backend Engineer with 5+ years of experience. Must have strong Python (FastAPI/Django) or Go background, PostgreSQL/Redis experience, and familiarity with Docker/AWS. Location: Pune/Noida.",
    "Product Manager": "Product Manager role for AI Recruitment platform. 4+ years of product management experience. Strong analytical skills, user research experience, and product strategy background are required. Location: Pune/Noida."
}

def rank_existing_database(custom_jd, w_sem, w_title, w_exp, w_avail, top_n):
    global pool_embeddings, candidate_pool, embedding_model
    
    if not candidate_pool or pool_embeddings is None:
        err_df = pd.DataFrame([{"Error": "Candidate database not loaded."}])
        return err_df, None, None
        
    # Normalize weights so they sum to 1.0
    total = w_sem + w_title + w_exp + w_avail
    if total == 0:
        total = 1.0
    w_sem /= total
    w_title /= total
    w_exp /= total
    w_avail /= total
        
    # Parse Job Description into CanonicalJD
    jd_schema = normalize_job_description(custom_jd)
    # Apply slider weights overriding default ones
    jd_schema.weights = {
        "semantic_similarity": w_sem,
        "title_relevance": w_title,
        "experience_fit": w_exp,
        "behavioral": w_avail,
        "career_quality": 0.0,
        "skill_depth": 0.0,
        "location": 0.0
    }

    # Embed custom JD
    custom_jd_emb = embedding_model.encode(jd_schema.raw_text, convert_to_numpy=True)
    
    # Recompute feature scores
    df = compute_all_features(candidate_pool, pool_embeddings, custom_jd_emb, jd_schema)
    
    # Run scoring logic
    df = compute_final_score(df, jd_schema)
    
    results_df = make_results_table(df, int(top_n))
    
    # Save files for download
    csv_path = "ranked_database_candidates.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    
    xlsx_path = "ranked_database_candidates.xlsx"
    results_df.to_excel(xlsx_path, index=False)
    
    return results_df, csv_path, xlsx_path

def rank_uploaded_resumes(file_obj, jd_file, w_sem, w_title, w_exp, w_avail, top_n):
    global embedding_model
    
    if file_obj is None:
        err_df = pd.DataFrame([{"Error": "Please upload a candidate JSON/JSONL file."}])
        return err_df, None, None
        
    if jd_file is None:
        err_df = pd.DataFrame([{"Error": "Please upload a Job Description file (.txt, .pdf, or .docx)."}])
        return err_df, None, None
        
    custom_jd = read_jd_file(jd_file.name)
    if not custom_jd or not custom_jd.strip():
        err_df = pd.DataFrame([{"Error": "The uploaded Job Description file is empty or cannot be parsed."}])
        return err_df, None, None
            
    # Load uploaded candidates dynamically (JSON / JSONL format-agnostic)
    candidates = load_candidates(file_obj.name)
    if not candidates:
        err_df = pd.DataFrame([{"Error": "No valid candidates found in file."}])
        return err_df, None, None
        
    # Normalize weights so they sum to 1.0
    total = w_sem + w_title + w_exp + w_avail
    if total == 0:
        total = 1.0
    w_sem /= total
    w_title /= total
    w_exp /= total
    w_avail /= total
        
    # Parse Job Description into CanonicalJD
    jd_schema = normalize_job_description(custom_jd)
    jd_schema.weights = {
        "semantic_similarity": w_sem,
        "title_relevance": w_title,
        "experience_fit": w_exp,
        "behavioral": w_avail,
        "career_quality": 0.0,
        "skill_depth": 0.0,
        "location": 0.0
    }

    # Generate text for embedding
    texts = [cand.get("full_text", "") for cand in candidates]
        
    # Embed uploaded candidates
    uploaded_embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    custom_jd_emb = embedding_model.encode(jd_schema.raw_text, convert_to_numpy=True)
    
    df = compute_all_features(candidates, uploaded_embeddings, custom_jd_emb, jd_schema)
    
    # Run scoring logic
    df = compute_final_score(df, jd_schema)
    
    results_df = make_results_table(df, int(top_n))
    
    # Save files for download
    csv_path = "ranked_uploaded_candidates.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    
    xlsx_path = "ranked_uploaded_candidates.xlsx"
    results_df.to_excel(xlsx_path, index=False)
    
    return results_df, csv_path, xlsx_path

def make_results_table(ranked_df, top_n=100):
    results = []
    for _, row in ranked_df.head(top_n).iterrows():
        reasoning = generate_reasoning(row)
        
        results.append({
            "candidate_id": row['candidate_id'],
            "rank": int(row['rank']),
            "score": f"{row['final_score']:.4f}",
            "reasoning": reasoning
        })
    return pd.DataFrame(results)

# --- Gradio UI Layout with custom styling ---
theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_600, *secondary_600)",
    block_title_text_weight="600",
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# 🤖 AI Recruiter: Dynamic Resume Ranking Platform")
    
    with gr.Tab("Mode 1: Rank Existing Pool"):
        gr.Markdown("### Search the preloaded database of candidates using your Job Description.")
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 📥 1. Job Description")
                preset_dropdown_1 = gr.Dropdown(
                    choices=list(PRESET_JDS.keys()), 
                    value="Senior AI Engineer (Default)", 
                    label="Load a Preset Job Description"
                )
                jd_input_1 = gr.Textbox(
                    value=JD_TEXT, 
                    label="Job Description Text", 
                    placeholder="Paste the full job description here...", 
                    lines=8
                )
                
                gr.Markdown("### ⚙️ 2. Search Settings")
                top_n_1 = gr.Slider(
                    minimum=5, 
                    maximum=50, 
                    value=20, 
                    step=1, 
                    label="Number of Top Candidates to Show"
                )
                
                with gr.Accordion("Fine-Tune Ranking Weights (Advanced)", open=False):
                    w_sem_1 = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="Semantic Fit (Meaning)")
                    w_title_1 = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Title Relevance Match")
                    w_exp_1 = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Experience Fit (Years)")
                    w_avail_1 = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Availability & Notice period")
                
                btn_db = gr.Button("Find Top Candidates", variant="primary")
                with gr.Row():
                    download_db = gr.File(label="Download CSV Output")
                    download_db_xlsx = gr.File(label="Download Excel (XLSX) Output")
                
            with gr.Column(scale=6):
                gr.Markdown("### 🏆 3. Ranked Results")
                gr.Markdown("*Note: Higher scores indicate better semantic fit between the job description and candidate profiles.*")
                results_db = gr.Dataframe(label="Discovered Candidates")
                
        # Link callbacks
        preset_dropdown_1.change(lambda k: PRESET_JDS[k], inputs=[preset_dropdown_1], outputs=[jd_input_1])
        btn_db.click(
            rank_existing_database, 
            inputs=[jd_input_1, w_sem_1, w_title_1, w_exp_1, w_avail_1, top_n_1], 
            outputs=[results_db, download_db, download_db_xlsx]
        )
        
    with gr.Tab("Mode 2: Upload & Rank New Resumes"):
        gr.Markdown("### Upload a custom candidate database (JSON/JSONL) and rank them.")
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 📥 1. Upload Candidates & JD")
                file_input = gr.File(label="Upload Candidate JSON/JSONL Database", file_count="single")
                jd_file_input_2 = gr.File(
                    label="Upload Job Description File (.txt, .pdf, .docx)", 
                    file_count="single", 
                    file_types=[".txt", ".pdf", ".docx"]
                )
                
                gr.Markdown("### ⚙️ 2. Search Settings")
                top_n_2 = gr.Slider(
                    minimum=5, 
                    maximum=50, 
                    value=20, 
                    step=1, 
                    label="Number of Top Candidates to Show"
                )
                
                with gr.Accordion("Fine-Tune Ranking Weights (Advanced)", open=False):
                    w_sem_2 = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="Semantic Fit (Meaning)")
                    w_title_2 = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Title Relevance Match")
                    w_exp_2 = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Experience Fit (Years)")
                    w_avail_2 = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Availability & Notice period")
                
                btn_upload = gr.Button("Find Top Candidates", variant="primary")
                with gr.Row():
                    download_upload = gr.File(label="Download CSV Output")
                    download_upload_xlsx = gr.File(label="Download Excel (XLSX) Output")
                
            with gr.Column(scale=6):
                gr.Markdown("### 🏆 3. Ranked Results")
                gr.Markdown("*Note: Higher scores indicate better semantic fit between the job description and candidate profiles.*")
                results_upload = gr.Dataframe(label="Discovered Candidates")
                
        # Link callbacks
        btn_upload.click(
            rank_uploaded_resumes, 
            inputs=[file_input, jd_file_input_2, w_sem_2, w_title_2, w_exp_2, w_avail_2, top_n_2], 
            outputs=[results_upload, download_upload, download_upload_xlsx]
        )

if __name__ == "__main__":
    demo.launch()
