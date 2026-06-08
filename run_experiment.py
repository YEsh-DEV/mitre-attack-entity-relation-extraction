import os
import json
import re
import pandas as pd
import torch
from transformers import pipeline

# ==========================================
# 1. STRICT LOCAL ENVIRONMENT CONFIGURATION
# ==========================================
CACHE_DIR = "Y:/Reserchintern/Experiment1/.cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Force transformers to never check the internet globally
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

print("--- Starting Focused Cyber Entity-Relationship Extraction & Evaluation ---")
print("Enforcing Strict Local Folder Routing (Zero Internet Access)")

device = 0 if torch.cuda.is_available() else -1
print(f"Hardware Status: Using {'GPU' if device == 0 else 'CPU (Fallback)'}")

STOPWORDS = {"the", "and", "a", "of", "to", "in", "is", "for", "with", "on", "at", "by", "an", "this", "that", "attack", "tactic"}

# ==========================================
# 2. AUTOMATED LOCAL PATH RESOLVER UTILITY
# ==========================================
def resolve_absolute_local_path(repo_id, base_cache=CACHE_DIR):
    """
    Bypasses broken Windows symlinks by finding the actual absolute directory 
    containing the local model configurations and weights.
    """
    hf_folder_format = "models--" + repo_id.replace("/", "--")
    snapshots_path = os.path.join(base_cache, "hub", hf_folder_format, "snapshots")
    
    if os.path.exists(snapshots_path):
        subfolders = os.listdir(snapshots_path)
        if subfolders:
            absolute_path = os.path.join(snapshots_path, subfolders[0])
            if os.path.exists(os.path.join(absolute_path, "config.json")):
                return absolute_path
                
    repo_keyword = repo_id.split("/")[-1].lower()
    for root, dirs, files in os.walk(base_cache):
        if "config.json" in files and repo_keyword in root.lower():
            return root
            
    return repo_id

# Resolve exact disk directory targets for the 2 functioning models
path_securebert = resolve_absolute_local_path("ehsanaghaei/SecureBERT")
path_ctibert    = resolve_absolute_local_path("ibm-research/CTI-BERT")

# ==========================================
# 3. LOADING & PREPROCESSING DATASETS 
# ==========================================
print("\nLoading and cleaning datasets from Excel files...")
try:
    attack_df = pd.read_excel("Attackmitre.xlsx")
    enterprise_df = pd.read_excel("MitreEnterprise.xlsx")
except FileNotFoundError as e:
    print(f"Error: Could not find Excel files! Details: {e}")
    exit(1)

attack_df.columns = attack_df.columns.str.strip()
enterprise_df.columns = enterprise_df.columns.str.strip()

attack_df['Group Techniques'] = attack_df['Group Techniques'].fillna('').astype(str)
attack_df['Group Techniques'] = attack_df['Group Techniques'].apply(
    lambda x: [i.strip() for i in x.split(';') if i.strip() and i.strip().lower() != 'nan']
)
attack_df = attack_df.explode('Group Techniques')
attack_df = attack_df[attack_df['Group Techniques'].astype(str).str.strip() != '']

print("Merging MITRE datasets on 'Group Techniques' -> 'Tactic ID'...")
merged_df = pd.merge(attack_df, enterprise_df, left_on="Group Techniques", right_on="Tactic ID")
print(f"Successfully merged! Total interactive rows to process: {len(merged_df)}")

# ==========================================
# 4. INITIALIZE PIPELINES DIRECTLY FROM DISK
# ==========================================
print("\nInitializing Local NLP Pipelines from Absolute Disk Folders...")

# We NO LONGER pass `model_kwargs={"local_files_only": True}` because `TRANSFORMERS_OFFLINE=1` 
# already handles this globally. This prevents the "multiple values" crash.

try:
    print(f"-> Loading SecureBERT from: {path_securebert}")
    secure_bert_pipe = pipeline("fill-mask", model=path_securebert, tokenizer=path_securebert, device=device)
except Exception as e:
    print(f"Skipping SecureBERT due to loading constraints: {e}")
    secure_bert_pipe = None

try:
    print(f"-> Loading CTI-BERT from: {path_ctibert}")
    cti_bert_pipe = pipeline("fill-mask", model=path_ctibert, tokenizer=path_ctibert, device=device)
except Exception as e:
    print(f"Skipping CTI-BERT due to loading constraints: {e}")
    cti_bert_pipe = None



