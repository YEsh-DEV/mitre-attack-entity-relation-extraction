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

=======\n")

df_results.to_csv("model_evaluation_table.csv")
print("Metrics successfully compiled and saved to 'model_evaluation_table.csv'!")
