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

# --- File extraction helper functions ---
def extract_text_from_pdf(pdf_path):
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def extract_text_from_docx(docx_path):
    import docx
    doc = docx.Document(docx_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_file(file_path):
    if not file_path:
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return extract_text_from_docx(file_path)
        else: # assume text file/other
            return extract_text_from_txt(file_path)
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return ""

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

def rank_existing_database(custom_jd, w_sem, w_title, w_exp, w_avail):
    global pool_embeddings, candidate_pool, embedding_model
    
    if not candidate_pool or pool_embeddings is None:
        err_df = pd.DataFrame([{"Error": "Candidate database not loaded on Hugging Face Space."}])
        return err_df, None
        
    # Normalize weights so they sum to 1.0
    total = w_sem + w_title + w_exp + w_avail
    if total == 0:
        total = 1.0
    w_sem /= total
    w_title /= total
    w_exp /= total
    w_avail /= total
        
    # Embed custom JD
    custom_jd_emb = embedding_model.encode(custom_jd, convert_to_numpy=True)
    
    # Recompute feature scores
    df = compute_all_features(candidate_pool, pool_embeddings, custom_jd_emb)
    
    # Run custom weighted linear combination score logic
    base_score = (
        w_sem   * df["semantic_similarity"] +
        w_title * df["title_relevance"]     +
        w_exp   * df["experience_fit"]      +
        w_avail * df["behavioral"]
    )
    title_gate = 0.35 + 0.65 * df["title_relevance"]
    df["final_score"] = base_score * title_gate
    
    # Penalize honeypots
    df.loc[df["is_honeypot"] == True, "final_score"] = 0.0
    
    # Sort and rank
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    
    results_df = make_results_table(df)
    
    # Save a temporary CSV for download
    csv_path = "ranked_database_candidates.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    
    return results_df, csv_path

def rank_uploaded_resumes(file_obj, custom_jd, jd_file, w_sem, w_title, w_exp, w_avail):
    global embedding_model
    
    if file_obj is None:
        err_df = pd.DataFrame([{"Error": "Please upload a candidate JSON/JSONL file."}])
        return err_df, None
        
    # Check if a JD file was uploaded, extract text from it
    if jd_file is not None:
        extracted_jd = extract_text_from_file(jd_file.name)
        if extracted_jd.strip():
            custom_jd = extracted_jd
            
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
        
    # Normalize weights so they sum to 1.0
    total = w_sem + w_title + w_exp + w_avail
    if total == 0:
        total = 1.0
    w_sem /= total
    w_title /= total
    w_exp /= total
    w_avail /= total
        
    # Generate text for embedding
    texts = []
    for cand in candidates:
        texts.append(cand.get("career_text", ""))
        
    # Embed uploaded candidates
    uploaded_embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    custom_jd_emb = embedding_model.encode(custom_jd, convert_to_numpy=True)
    
    df = compute_all_features(candidates, uploaded_embeddings, custom_jd_emb)
    
    # Run custom weighted score
    base_score = (
        w_sem   * df["semantic_similarity"] +
        w_title * df["title_relevance"]     +
        w_exp   * df["experience_fit"]      +
        w_avail * df["behavioral"]
    )
    title_gate = 0.35 + 0.65 * df["title_relevance"]
    df["final_score"] = base_score * title_gate
    
    # Penalize honeypots
    df.loc[df["is_honeypot"] == True, "final_score"] = 0.0
    
    # Sort and rank
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    
    results_df = make_results_table(df)
    
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

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AdvRecruiter - Interactive Candidate Discovery Engine")
    
    with gr.Tab("Mode 1: Rank Existing Pool"):
        with gr.Row():
            with gr.Column(scale=4):
                jd_input_1 = gr.Textbox(value=JD_TEXT, label="Job Description", lines=5)
                with gr.Row():
                    w_sem_1 = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="Semantic (Meaning)")
                    w_title_1 = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Title Match")
                with gr.Row():
                    w_exp_1 = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Experience (Years)")
                    w_avail_1 = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Availability")
                with gr.Row():
                    btn_db = gr.Button("Rank Database Candidates", variant="primary")
                    download_db = gr.File(label="Download Ranked CSV Output", height=70)
            with gr.Column(scale=6):
                results_db = gr.Dataframe(label="Top Ranked Candidates (100k database)")
                
        btn_db.click(
            rank_existing_database, 
            inputs=[jd_input_1, w_sem_1, w_title_1, w_exp_1, w_avail_1], 
            outputs=[results_db, download_db]
        )
        
    with gr.Tab("Mode 2: Upload & Rank New Resumes"):
        with gr.Row():
            with gr.Column(scale=4):
                file_input = gr.File(label="Upload Candidate JSON/JSONL", file_count="single", height=90)
                jd_file_input_2 = gr.File(label="Upload Job Description (.txt, .pdf, .docx)", file_count="single", file_types=[".txt", ".pdf", ".docx"], height=90)
                jd_input_2 = gr.Textbox(value=JD_TEXT, label="Job Description (used if no file uploaded)", lines=4)
                with gr.Row():
                    w_sem_2 = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="Semantic (Meaning)")
                    w_title_2 = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Title Match")
                with gr.Row():
                    w_exp_2 = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Experience (Years)")
                    w_avail_2 = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Availability")
                with gr.Row():
                    btn_upload = gr.Button("Score & Rank Resumes", variant="primary")
                    download_upload = gr.File(label="Download Ranked CSV Output", height=70)
            with gr.Column(scale=6):
                results_upload = gr.Dataframe(label="Ranked Results (New Uploads)")
                
        btn_upload.click(
            rank_uploaded_resumes, 
            inputs=[file_input, jd_input_2, jd_file_input_2, w_sem_2, w_title_2, w_exp_2, w_avail_2], 
            outputs=[results_upload, download_upload]
        )

if __name__ == "__main__":
    demo.launch(theme=theme)
