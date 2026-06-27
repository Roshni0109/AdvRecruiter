# 🤖 AdvRecruiter - Project Explained

Welcome! This document explains the **AdvRecruiter** project in a very simple, clear way—as if explaining it to a 10-year-old. It contains everything you need to know to confidently discuss this project in an interview!

---

## 1. Project Overview

### What is this project?
Imagine a company wants to hire a new **Senior AI Engineer**. They post a job description, and suddenly **100,000 people** send in their resumes! 
No human recruiter has the time to read 100,000 resumes. 

**AdvRecruiter** is a smart robot assistant (an AI ranking engine) that reads all 100,000 resumes, compares them to the job description, and picks the **top 100 best candidates** in just a few seconds.

### What problem does it solve?
1. **Speed:** Reading resumes is slow. Our system does it in seconds.
2. **Beyond Keywords:** Simple search tools only look for exact words. If a resume says "Machine Learning" but the job description says "AI", simple tools might miss it. Our system understands that those two terms mean almost the same thing!
3. **Ghost Candidates:** Some candidates have great resumes but take 3 months to start or never reply to emails. Our system checks "behavioral signals" to prioritize active, responsive people.
4. **Honeypot Filtering:** Sometimes people put fake lists of hot keywords in their resumes to trick search engines. Our system automatically detects these "honeypots" and drops their score to zero.

### A Real-World Example
A recruiter at **Redrob** opens their dashboard. They paste in a job description for a "Senior NLP Engineer". The system scans the database of 100,000 candidates and displays a sorted table of the top matches, complete with a readable summary of *why* each person was chosen.

---

## 2. High-Level Architecture (The Big Picture)

Here is a short story of how the system works:

1. **The Library (Data Loading):** First, the system gathers all 100,000 resumes from a big file (`candidates.jsonl`). It cleans up the data, flattening nested details into simple boxes.
2. **The Translator (Embeddings):** The system translates the job description and the resumes into a special language of numbers (called **embeddings**). These numbers act like a unique fingerprint of the *meaning* of the text.
3. **The Matching Score (Similarity):** The system measures the angle between the Job Description's meaning-fingerprint and each candidate's resume-fingerprint. If they point in the same direction, they are a close match.
4. **The Feature Calculator (Feature Engineering):** Beside semantic meaning, the system calculates 7 special scores for every candidate, such as title fit, notice period, location preference, and years of experience.
5. **The Final Ranker (Ranking):** The system combines all 7 scores using weights into one final score between `0.0` and `1.0`. It drops fake profiles (honeypots) to `0.0`.
6. **The Output:** The top 100 candidates are written to a spreadsheet (`submission.csv`) with a clear sentence explaining why they are there.

---

## 3. Tech Stack and Libraries

* **Python:** The programming language we use to write all our logic.
* **pandas:** A library used to handle tables of data (like Excel spreadsheets) inside our code.
* **numpy:** A library used for fast mathematical operations on lists of numbers.
* **sentence-transformers (all-MiniLM-L6-v2):** A lightweight AI model that converts text sentences into 384-dimensional number lists (meanings). We use it because it is fast on normal CPUs without needing expensive graphics cards (GPUs).
* **scikit-learn:** Used to calculate `cosine_similarity` (measuring the angle of meaning between resumes and the JD).
* **Gradio:** A library that lets us build a beautiful web dashboard user interface using simple Python code.
* **Hugging Face Spaces:** A cloud platform where our Gradio app runs live so anyone can access it via a web link.

---

## 4. Project Structure (Folders and Files)

```
AdvRecruiter/
├── data/                    # Contains raw inputs (candidates.jsonl)
├── src/                     # Core logic modules
│   ├── data_loader.py       # Reads and cleans candidate profiles
│   ├── feature_engineering.py  # Computes the 7 individual feature scores
│   ├── ranker.py            # Combines scores to produce final ranks
│   └── utils.py             # Small helper functions
├── scripts/
│   ├── preprocess.py        # Runs the heavy embedding work once
│   └── rank_candidates.py   # Main script to output final submission.csv
├── outputs/                 # Stores output CSVs, Parquet files, and binary maps
├── requirements.txt         # List of libraries to install
└── README.md                # Project startup readme
```

### File Responsibilities
* [data_loader.py](file:///d:/AI%20Recruiter%20Hackathon/AdvRecruiter/src/data_loader.py): Reads the raw data file, detects fake profile tricks, and cleans the format.
* [feature_engineering.py](file:///d:/AI%20Recruiter%20Hackathon/AdvRecruiter/src/feature_engineering.py): Scores candidates individually on each of the 7 features.
* [ranker.py](file:///d:/AI%20Recruiter%20Hackathon/AdvRecruiter/src/ranker.py): The calculator that runs the final weighted formula and sorts everyone.
* [utils.py](file:///d:/AI%20Recruiter%20Hackathon/AdvRecruiter/src/utils.py): Holds common utility tools.
* [app.py](file:///d:/AI%20Recruiter%20Hackathon/AdvRecruiter/app.py): The web app code running the interactive Gradio UI.

---

## 5. Detailed File-by-File Explanation

### a) `src/data_loader.py`
This file is the "front door" of the project.
* **Input:** A JSONL file (`candidates.jsonl`) containing raw candidate records.
* **Key Functions:**
  * `load_candidates(filepath)`: Opens the large candidates file line-by-line (streaming) to save memory, and calls `extract_candidate_features`.
  * `extract_candidate_features(raw)`: Flattens the nested raw profile data. It converts timestamps to days elapsed, detects if a candidate is a fake profile ("honeypot"), and extracts career timelines.
* **Output:** A clean list of dictionaries containing flattened details (like `candidate_id`, `years_exp`, `notice_days`, `is_honeypot`).

### b) `src/feature_engineering.py`
This file calculates individual scores between `0.0` and `1.0` for 7 core criteria:
1. **Semantic Similarity:** Angle of meaning between the candidate's career text and the Job Description.
2. **Title Relevance:** Looks up keywords in current titles against a mapped list (AI/ML Engineer = 1.0, Backend Engineer = 0.45, HR Manager = 0.0).
3. **Experience Fit:** Uses a "tent-shaped" curve. Experience between 5-9 years gets `1.0`. Too junior (<2 years) or overqualified (>15 years) get lower scores.
4. **Location Fit:** Scores preferred cities (Noida/Pune) higher than others.
5. **Behavioral Availability:** Scores notice periods (shorter notice = higher score) and responsiveness.
6. **Career Quality:** Penalizes service-oriented consulting firms while boosting product-focused tenures.
7. **Skill Depth:** Matches resume skills with the Job Description's core toolsets.

### c) `src/ranker.py`
This file runs the final math:
* **The Base Score:** Sums the 7 features multiplied by their importance weights.
* **The Title Gate Multiplier:** Multiplies the base score by `(0.35 + 0.65 * Title_Relevance)`. If a candidate has a title score of `0.0` (like an HR Manager), their final score is slashed by 65%, ensuring non-engineers can never rank at the top.
* **Honeypot Filter:** Sets any flagged honeypot profiles' scores to exactly `0.0`.
* **Sort and Rank:** Sorts candidates from highest to lowest score and assigns ranks 1 to N.

### d) `app.py` (Hugging Face Gradio UI)
* **The UI:** Recruiter enters a Job Description on the left, clicks **"Rank"**, and views a table on the right showing the top 100 matches with their reasoning.
* **Behind the scenes:**
  * To run fast on Hugging Face CPU Spaces under memory limits, it maps the large 100k candidate embeddings directly to disk using `np.memmap` (a memory map that reads from disk without consuming RAM).
  * Computes the scores in real-time, displays the DataFrame, and creates a downloadable CSV file.

---

## 6. Data and Embeddings

### What is an Embedding?
Imagine every word or sentence can be mapped to coordinates in a giant "meaning map". 
* "Vector database" and "Pinecone" end up in the same corner of the map because they represent search systems.
* "Accounting" ends up in a completely different corner.
An **embedding** is simply the coordinate list (384 numbers in our case) of where a text lies on this map. We use `sentence-transformers` to generate these coordinates.

### Similarity Calculation
We calculate **Cosine Similarity**. If the coordinates of a candidate's resume point in the exact same direction as the Job Description's coordinates, the angle is 0° and the similarity score is `1.0`.

---

## 7. Limitations and Future Improvements

### Current Limitations:
1. **English Only:** The model is optimized for English profiles.
2. **Fixed Weights:** The feature weights (e.g., 35% semantic similarity) are hardcoded.

### Future Improvements:
* **Interactive Weight Sliders:** Let recruiters change weights (e.g., set location importance to 0% for remote roles).
* **Cross-Lingual Embeddings:** Use a multilingual model to rank resumes in French, German, or Hindi.
* **Deep Parsing:** Use an offline Local LLM (like Llama-3) to read the career text and extract bullet-point achievements.

---

## 8. How to Run the Project (For Beginners)

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Preprocess candidate data (Run once)
python scripts/preprocess.py

# 4. Generate final CSV ranking
python scripts/rank_candidates.py

# 5. Start Gradio locally
python app.py
```

---

## 9. Interview Cheat Sheet (Top 10 Key Points)

1. **The Core Goal:** AdvRecruiter ranks the top 100 candidates from a database of 100,000 for an AI Engineer role in under 5 minutes.
2. **The Challenge:** Processing 100k records quickly on cheap hardware without running out of RAM or using expensive GPUs.
3. **The Solution:** We split the pipeline: slow embedding work is done once in a pre-compute step, and the fast ranking step runs in less than 1 second.
4. **Memory Optimization:** We used `np.memmap` (memory mapping) to keep embeddings on disk. RAM usage remained flat instead of spiking.
5. **Beyond Keyword Search:** We used `sentence-transformers` (`all-MiniLM-L6-v2`) to capture semantic meaning.
6. **Honeypot Shield:** We designed algorithms to detect and penalize fake keyword-stuffed profiles.
7. **Title Gating:** Implemented a multiplier gate to prevent non-technical profiles from ranking high.
8. **Behavioral Scoring:** Weighted candidates by availability signals (e.g. shorter notice periods, high activity).
9. **Gradio UI:** Built a front-end dashboard allowing recruiters to paste custom JDs and download ranked CSV outputs.
10. **Validation:** The final submission CSV is validated and fully conforms to the hackathon's required schema.
