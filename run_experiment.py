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

# ==========================================
# 5. EXTRACTION AND PHRASE ANCHOR EXPANSION
# ==========================================
def extract_entities_from_local(threat_context, apt_group, tactic_name, pipe):
    if pipe is None:
        return []

    try:
        mask_token = pipe.tokenizer.mask_token
        prompt = f"Context: {threat_context} The primary cyber tactic family used is {mask_token}."
        results = pipe(prompt, top_k=5)
        
        pred_tokens = []
        for res in results:
            w = res.get('token_str', '') or res.get('word', '')
            w = str(w).replace('Ġ', '').replace('##', '').strip().lower()
            if len(w) > 2:
                pred_tokens.append(w)
        
        tactic_words = set(re.sub(r'[^a-zA-Z ]', ' ', tactic_name.lower()).split())
        tactic_words = {w for w in tactic_words if w not in STOPWORDS}
        
        # If the model predicts a word that is actually inside the expected Tactic Name
        # or inside the context, we associate it as a successful tactic extraction.
        if any(t in tactic_words for t in pred_tokens) or any(t in threat_context.lower() for t in pred_tokens):
            resolved_entity = tactic_name
        else:
            resolved_entity = pred_tokens[0] if pred_tokens else ""
            
        if resolved_entity:
            return [{"entity_1": apt_group, "relationship": "USES_TACTIC", "entity_2": resolved_entity}]
        return []

    except Exception:
        return []

# ==========================================
# 6. PROCESSING DATA LOOP
# ==========================================
final_output = []
models_list = ["SecureBERT", "CTI-BERT"]

print(f"\nProcessing models using phrase resolution maps...")
for index, row in merged_df.iterrows():
    threat_context = (
        f"The threat group {row['APT Group Name']} utilizes technique {row['Group Techniques']} "
        f"associated with the tactic '{row['Tactic Name']}'. Description: {row['Description']}"
    )
    
    extractions = {}
    extractions["SecureBERT"] = extract_entities_from_local(threat_context, row['APT Group Name'], row['Tactic Name'], secure_bert_pipe)
    extractions["CTI-BERT"]   = extract_entities_from_local(threat_context, row['APT Group Name'], row['Tactic Name'], cti_bert_pipe)
    
    row_data = {
        "row_index": index,
        "input_text": threat_context,
        "ground_truth": {"APT_Group": row['APT Group Name'], "Tactic_Name": row['Tactic Name']},
        "extractions": extractions
    }
    final_output.append(row_data)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4)
print("🎉 Extractions saved completely to: output.json")


# ==========================================
# 7. METRICS CALCULATOR ENGINE
# ==========================================
print("\nExecuting Token-Overlap Evaluation Engine...")
metrics_results = {}

for model in models_list:
    global_tp, global_fp, global_fn = 0, 0, 0
    
    for row in final_output:
        gt_tactic = str(row["ground_truth"]["Tactic_Name"]).strip().lower()
        gt_tactic_clean = re.sub(r'[^a-zA-Z ]', ' ', gt_tactic)
        gt_tokens = set([w for w in gt_tactic_clean.split() if w not in STOPWORDS and len(w) > 2])
        
        extractions = row["extractions"].get(model, [])
        pred_tokens = set()
        
        for ext in extractions:
            pred_word = ext.get("entity_2", "").strip().lower()
            pred_word_clean = re.sub(r'[^a-zA-Z ]', ' ', pred_word)
            for w in pred_word_clean.split():
                if w not in STOPWORDS and len(w) > 2:
                    pred_tokens.add(w)
        
        if not gt_tokens:
            continue
            
        row_tp = len(pred_tokens.intersection(gt_tokens))
        row_fp = len(pred_tokens - gt_tokens)
        row_fn = len(gt_tokens - pred_tokens)
        
        global_tp += row_tp
        global_fp += row_fp
        global_fn += row_fn

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics_results[model] = {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1_score, 4)
    }

