"""
MITRE ATT&CK Entity & Relation Extraction
Generalised Multi-Model Benchmarking Framework
Supports: SecureBERT | CTI-BERT | BERT-BiLSTM-CRF | LLaMA-3.1-8B-Instruct
Usage:
    # Interactive model selection menu
    python main.py
    # Pass model names directly as arguments
    python main.py --models SecureBERT CTI-BERT1
    # Run all available/loadable models
    python main.py --all
Outputs -> Y:/Reserchintern/Experiment1/
    entity_relationship_output.json
    evaluation_results.csv
"""

import os
import sys
import json
import re
import time
import argparse
import pandas as pd
import torch
from transformers import pipeline as hf_pipeline

# ============================================================
# 0.  PATHS & ENVIRONMENT
# ============================================================
BASE_DIR  = r"Y:/Reserchintern/Experiment1"
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
JSON_OUT  = os.path.join(BASE_DIR, "entity_relationship_output.json")
CSV_OUT   = os.path.join(BASE_DIR, "evaluation_results.csv")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ============================================================
# 1.  MODEL REGISTRY
#     Add any new model here — nothing else needs to change.
##    Fields:
#       hub_id      : Hugging Face repo ID (used as fallback if snap absent)
#       type        : "mlm"  -> fill-mask pipeline  (SecureBERT, CTI-BERT)
#                     "ner"  -> token-classification  (BERT-BiLSTM-CRF)
#                     "gen"  -> text-generation       (LLaMA)
#       mask_token  : mask placeholder for MLM probes ("<mask>" or "[MASK]")
#       snap        : absolute path to local snapshot folder (auto-resolved
#                     if set to None using resolve_snap())
#       top_k       : top-K predictions to retrieve (MLM only)
#       max_new_tok : max tokens to generate (gen only)
# ============================================================
def resolve_snap(hub_id: str) -> str | None:
    """
    Scan the local cache snapshots folder and return the first available
    snapshot path. Bypasses Windows symlink restrictions.
    Returns None if model is not cached locally.
    """
    folder     = "models--" + hub_id.replace("/", "--")
    snaps_dir  = os.path.join(CACHE_DIR, "hub", folder, "snapshots")
    if os.path.isdir(snaps_dir):
        entries = [e for e in os.listdir(snaps_dir)
                   if os.path.isdir(os.path.join(snaps_dir, e))]
        if entries:
            return os.path.join(snaps_dir, entries[0])
    return None


MODEL_REGISTRY = {
    "SecureBERT": {
        "hub_id":      "ehsanaghaei/SecureBERT",
        "type":        "mlm",
        "mask_token":  "<mask>",        # RoBERTa-based
        "snap":        None,            # auto-resolved at startup
        "top_k":       10,
    },
    "CTI-BERT": {
        "hub_id":      "ibm-research/CTI-BERT",
        "type":        "mlm",
        "mask_token":  "[MASK]",        # BERT-based
        "snap":        None,
        "top_k":       10,
    },
    "BERT-BiLSTM-CRF": {
        "hub_id":      "gcelikmasat/BERT-biLSTM-CRF",
        "type":        "ner",
        "mask_token":  "[MASK]",
        "snap":        None,
        "top_k":       10,
    },
    "LLaMA": {
        "hub_id":      "meta-llama/Llama-3.1-8B-Instruct",
        "type":        "gen",
        "mask_token":  None,
        "snap":        None,
        "top_k":       None,
        "max_new_tok": 64,
    },
}

# Auto-resolve snapshot paths at import time
for _cfg in MODEL_REGISTRY.values():
    _cfg["snap"] = resolve_snap(_cfg["hub_id"])

# ============================================================
# 2.  DATA LOADING & PREPROCESSING
# ============================================================
STOP = {
    "with","have","from","their","that","this","they","used","which","such",
    "these","into","been","also","more","than","other","both","system","systems",
    "network","information","data","access","using","target","attack","process",
    "service","software","file","user","files","users","based","local","remote",
    "within","discovery",
}

def load_dataset() -> pd.DataFrame:
    """
    Load, explode, merge and deduplicate the two source Excel files.
    Returns a DataFrame with columns:
        APT Group Name | Tactic ID | Tactic Name | Description | gt_pairs
    """
    print("[1/4] Loading and merging datasets...")
    attack_df     = pd.read_excel(os.path.join(BASE_DIR, "Attackmitre.xlsx"))
    enterprise_df = pd.read_excel(os.path.join(BASE_DIR, "MitreEnterprise.xlsx"))
    
    attack_df.columns     = attack_df.columns.str.strip()
    enterprise_df.columns = enterprise_df.columns.str.strip()
    
    # Explode semicolon-separated Group Techniques (e.g. "T1057; T1135; T1003")
    exp = attack_df.copy()
    exp["Group Techniques"] = exp["Group Techniques"].fillna("").astype(str).str.split(";")
    exp = exp.explode("Group Techniques")
    exp["Group Techniques"] = exp["Group Techniques"].str.strip()
    exp = exp[exp["Group Techniques"].notna() &
              (exp["Group Techniques"] != "") &
              (exp["Group Techniques"].str.lower() != "nan")]
              
    merged = pd.merge(
        exp,
        enterprise_df[["Tactic ID", "Tactic Name", "Description"]],
        left_on="Group Techniques", right_on="Tactic ID", how="inner"
    )
    merged = merged[merged["Description"].notna()]
    
    # Deduplicate on unique APT + Tactic pair (removes software-ID duplicates)
    df = (merged[["APT Group Name", "Tactic ID", "Tactic Name", "Description"]]
          .drop_duplicates()
          .reset_index(drop=True))
    df["gt_pairs"] = df["Tactic Name"].apply(_get_gt_pairs)
    
    print(f"    Unique APT+Tactic pairs : {len(df)}")
    print(f"    APT groups              : {df['APT Group Name'].nunique()}")
    print(f"    Unique tactics          : {df['Tactic ID'].nunique()}")
    total_probes = sum(len(p) for p in df["gt_pairs"])
    print(f"    Total evaluation probes : {total_probes}")
    return df

def _get_gt_pairs(tactic_name: str) -> list:
    """
    Returns [(clean_token, original_word), ...] for each meaningful word
    in the tactic name after stopword filtering.
    clean_token   = alphanumeric lowercase form  (used for prediction comparison)
    original_word = exact word as it appears     (used for regex masking in probe)
    """
    pairs = []
    for word in tactic_name.split():
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if len(clean) >= 4 and clean not in STOP and clean.isalpha():
            pairs.append((clean, word))
    return pairs

# ============================================================
# 3.  MODEL LOADER
# ============================================================
def load_model(name: str) -> dict | None:
    """
    Load the pipeline for a registered model.
    Returns the registry config dict with an added 'pipe' key,
    or None if loading fails.
    """
    cfg    = MODEL_REGISTRY[name]
    source = cfg["snap"] if cfg["snap"] and os.path.isdir(cfg["snap"]) else cfg["hub_id"]
    device = 0 if torch.cuda.is_available() else -1
    print(f"    Loading {name}  [{cfg['type'].upper()}]  from: {source}")
    try:
        if cfg["type"] == "mlm":
            pipe = hf_pipeline("fill-mask", model=source, tokenizer=source, device=device)
        elif cfg["type"] == "ner":
            pipe = hf_pipeline(
                "token-classification",
                model=source, tokenizer=source,
                aggregation_strategy="simple",
                device=device
            )
        elif cfg["type"] == "gen":
            pipe = hf_pipeline(
                "text-generation",
                model=source, tokenizer=source,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            print(f"    [SKIP] Unknown model type: {cfg['type']}")
            return None
        print(f"    [OK] {name}")
        return {**cfg, "pipe": pipe, "name": name}
    except Exception as e:
        print(f"    [FAIL] {name}: {e}")
        return None

# ============================================================
# 4.  EXTRACTION DISPATCHERS
#     One function per model type, all return the same schema:
#       results  : list of {gt, top1, hit, skipped}
#       triplets : list of {entity_1, relationship, entity_2, gt_entity, correct}
# ============================================================
def _clean_pred(raw: str) -> str:
    """Normalise a predicted token to alphanumeric lowercase."""
    return re.sub(r"[^a-z]", "", str(raw).replace("Ġ", "").replace("##", "").lower())

def _extract_mlm(model_cfg: dict, apt: str, tactic_name: str,
                 desc: str, gt_pairs: list) -> tuple:
    """
    Cloze-task extraction for Masked Language Models (fill-mask).
    For each GT (clean_token, original_word):
      1. Mask the original_word inside the tactic name.
      2. Build a single-mask probe: "{APT} performs {masked_tactic}. {desc[:100]}"
      3. Predict top-K tokens at the mask position.
      4. Check if clean_token is in the predicted set.
    """
    pipe       = model_cfg["pipe"]
    mask_token = model_cfg["mask_token"]
    top_k      = model_cfg["top_k"]
    results, triplets = [], []
    for clean_gt, orig_word in gt_pairs:
        # Replace original word in tactic name with mask (one occurrence)
        masked_tactic = re.sub(
            re.escape(orig_word), mask_token,
            tactic_name, flags=re.IGNORECASE, count=1
        )
        desc_snippet = re.sub(r"\s+", " ", desc[:100]).strip()
        probe = f"{apt} performs {masked_tactic}. {desc_snippet}"
        
        # Enforce exactly one mask token
        if probe.count(mask_token) != 1:
            probe = f"The attack technique {masked_tactic} is used by {apt}."
        if probe.count(mask_token) != 1:
            results.append({"gt": clean_gt, "top1": None, "hit": False, "skipped": True})
            continue
            
        try:
            raw_preds = pipe(probe, top_k=top_k)
        except Exception as e:
            print(f"      [pipe error] {e}")
            results.append({"gt": clean_gt, "top1": None, "hit": False, "skipped": True})
            continue
            
        # Guard: single-mask returns list[dict]; multi-mask returns list[list[dict]]
        if raw_preds and isinstance(raw_preds[0], list):
            raw_preds = raw_preds[0]
            
        predicted = [
            _clean_pred(p.get("token_str", "") or p.get("word", ""))
            for p in raw_preds if isinstance(p, dict)
        ]
        top1 = predicted[0] if predicted else None
        hit  = clean_gt in predicted
        results.append({"gt": clean_gt, "top1": top1, "hit": hit, "skipped": False})
        
        if top1 and len(top1) >= 3 and top1.isalpha():
            triplets.append({
                "entity_1":     apt,
                "relationship": "USES",
                "entity_2":     top1,
                "gt_entity":    clean_gt,
                "correct":      hit,
            })
    return results, triplets

def _extract_ner(model_cfg: dict, apt: str, tactic_name: str,
                 desc: str, gt_pairs: list) -> tuple:
    """
    Token-classification NER extraction (BERT-BiLSTM-CRF style).
    Feeds the description text directly into the NER pipeline.
    Extracted entity spans are compared against GT tokens via
    token-overlap matching to compute TP/FP/FN.
    """
    pipe = model_cfg["pipe"]
    results, triplets = [], []
    input_text = f"{apt} uses {tactic_name}. {desc}"
    try:
        ner_output = pipe(input_text)
    except Exception as e:
        print(f"      [ner error] {e}")
        for clean_gt, _ in gt_pairs:
            results.append({"gt": clean_gt, "top1": None, "hit": False, "skipped": True})
        return results, triplets
        
    # Collect all predicted entity words (normalised)
    predicted_set = set()
    for ent in ner_output:
        word = _clean_pred(ent.get("word", ""))
        if len(word) >= 3:
            predicted_set.add(word)
            # Also add individual sub-tokens for partial matching
            for tok in word.split():
                if len(tok) >= 3:
                    predicted_set.add(tok)
                    
    top1_global = list(predicted_set)[0] if predicted_set else None
    for clean_gt, _ in gt_pairs:
        hit = clean_gt in predicted_set
        results.append({"gt": clean_gt, "top1": top1_global, "hit": hit, "skipped": False})
        if top1_global and len(top1_global) >= 3:
            triplets.append({
                "entity_1":     apt,
                "relationship": "USES",
                "entity_2":     top1_global,
                "gt_entity":    clean_gt,
                "correct":      hit,
            })
    return results, triplets

def _extract_gen(model_cfg: dict, apt: str, tactic_name: str,
                 desc: str, gt_pairs: list) -> tuple:
    """
    Generative extraction for instruction-tuned LLMs (LLaMA style).
    Builds an instruction prompt asking the model to identify the
    tactic keyword. Parses the generated text and checks for GT tokens.
    """
    pipe         = model_cfg["pipe"]
    max_new_tok  = model_cfg.get("max_new_tok", 64)
    results, triplets = [], []
    for clean_gt, orig_word in gt_pairs:
        prompt = (
            f"<|system|>You are a cybersecurity analyst. "
            f"Identify the key tactic term in the following and respond with only that single word.\n"
            f"<|user|>APT group: {apt}\nTechnique description: {desc[:200]}\n"
            f"What single word best describes the tactic '{tactic_name}'?\n<|assistant|>"
        )
        try:
            output    = pipe(prompt, max_new_tokens=max_new_tok,
                             do_sample=False, temperature=1.0)
            gen_text  = output[0]["generated_text"].split("<|assistant|>")[-1].strip()
            predicted = [_clean_pred(w) for w in gen_text.split()
                         if len(_clean_pred(w)) >= 3]
        except Exception as e:
            print(f"      [gen error] {e}")
            results.append({"gt": clean_gt, "top1": None, "hit": False, "skipped": True})
            continue
            
        top1 = predicted[0] if predicted else None
        hit  = clean_gt in predicted
        results.append({"gt": clean_gt, "top1": top1, "hit": hit, "skipped": False})
        if top1 and len(top1) >= 3 and top1.isalpha():
            triplets.append({
                "entity_1":     apt,
                "relationship": "USES",
                "entity_2":     top1,
                "gt_entity":    clean_gt,
                "correct":      hit,
            })
    return results, triplets

# Dispatcher: routes each model to the correct extraction function
_DISPATCHER = {
    "mlm": _extract_mlm,
    "ner": _extract_ner,
    "gen": _extract_gen,
}

# ============================================================
# 5.  METRIC COMPUTATION
# ============================================================
def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p + r) if (p + r) > 0  else 0.0
    return {
        "Precision": round(p,  4),
        "Recall":    round(r,  4),
        "F1":        round(f1, 4),
        "TP":        tp, "FP": fp, "FN": fn
    }

# ============================================================
# 6.  USER MODEL SELECTION
# ============================================================
def select_models_interactive() -> list:
    """
    Show a numbered menu and let the user pick one or more models.
    Returns a list of selected model name strings.
    """
    names = list(MODEL_REGISTRY.keys())
    print("\n" + "=" * 55)
    print("  MITRE ATT&CK Extraction Framework")
    print("  Available Models:")
    print("=" * 55)
    for i, name in enumerate(names, 1):
        cfg  = MODEL_REGISTRY[name]
        snap = cfg["snap"]
        status = "[cached]" if snap and os.path.isdir(snap) else "[download required]"
        print(f"  {i}. {name:<25} {cfg['type'].upper():<5} {status}")
    print("=" * 55)
    print("  Enter model numbers separated by spaces (e.g. 1 2)")
    print("  Or press Enter to run all cached models")
    print("-" * 55)
    raw = input("  Selection: ").strip()
    
    if not raw:
        selected = [n for n in names
                    if MODEL_REGISTRY[n]["snap"] and os.path.isdir(MODEL_REGISTRY[n]["snap"])]
        if not selected:
            print("  No cached models found. Please download at least one model.")
            sys.exit(1)
        print(f"  Auto-selected cached models: {', '.join(selected)}")
        return selected
        
    selected = []
    for token in raw.split():
        try:
            idx = int(token) - 1
            if 0 <= idx < len(names):
                selected.append(names[idx])
        except ValueError:
            # Accept model names directly too
            if token in MODEL_REGISTRY:
                selected.append(token)
                
    if not selected:
        print("  Invalid selection. Exiting.")
        sys.exit(1)
    return selected

def parse_args() -> list:
    """Parse CLI arguments. Returns list of model names to run."""
    parser = argparse.ArgumentParser(
        description="MITRE ATT&CK Entity-Relation Extraction Framework"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        metavar="MODEL",
        help=f"One or more model names: {', '.join(MODEL_REGISTRY.keys())}"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run all models that are available locally"
    )
    args = parser.parse_args()
    
    if args.all:
        return [n for n in MODEL_REGISTRY
                if MODEL_REGISTRY[n]["snap"] and os.path.isdir(MODEL_REGISTRY[n]["snap"])]
    if args.models:
        return args.models
        
    # No CLI args: fall through to interactive menu
    return None

# ============================================================
# 7.  MAIN EXECUTION
# ============================================================
def main():
    print("\n" + "=" * 55)
    print("  MITRE ATT&CK Entity-Relation Extraction")
    print("  Generalised Multi-Model Benchmarking Framework")
    print("=" * 55)
    
    # --- Determine which models to run ---
    selected_names = parse_args()
    if selected_names is None:
        selected_names = select_models_interactive()
    print(f"\n  Models to run: {', '.join(selected_names)}\n")
    
    # --- Hardware ---
    device = 0 if torch.cuda.is_available() else -1
    print(f"[HW] {'GPU' if device == 0 else 'CPU'}")
    
    # --- Load dataset ---
    df = load_dataset()
    
    # --- Load selected models ---
    print(f"\n[2/4] Loading {len(selected_names)} model(s)...")
    loaded_models = {}
    for name in selected_names:
        if name not in MODEL_REGISTRY:
            print(f"    [SKIP] Unknown model: {name}")
            continue
        result = load_model(name)
        if result is not None:
            loaded_models[name] = result
            
    if not loaded_models:
        print("\n[ERROR] No models loaded successfully. Exiting.")
        sys.exit(1)
    print(f"\n  Loaded: {', '.join(loaded_models.keys())}")
    
    # --- Initialise metric accumulators ---
    counters = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in loaded_models
    }
    
    # --- Main extraction loop ---
    print(f"\n[3/4] Running extraction on {len(df)} unique APT+Tactic pairs...\n")
    records = []
    for i, row in df.iterrows():
        apt    = row["APT Group Name"]
        tid    = row["Tactic ID"]
        tname  = row["Tactic Name"]
        desc   = str(row["Description"])
        pairs  = row["gt_pairs"]
        if not pairs:
            continue
            
        print(f"  [{i+1}/{len(df)}] {apt} | {tid} | {tname}")
        record = {
            "apt_group":   apt,
            "tactic_id":   tid,
            "tactic_name": tname,
            "description": desc,
            "gt_tokens":   [p[0] for p in pairs],
            "extractions": {}
        }
        
        for name, model_cfg in loaded_models.items():
            extract_fn = _DISPATCHER[model_cfg["type"]]
            res, trips = extract_fn(model_cfg, apt, tname, desc, pairs)
            
            # Update metric counters
            for r in res:
                if r["hit"]:
                    counters[name]["tp"] += 1
                else:
                    counters[name]["fn"] += 1
                    if r["top1"] and not r.get("skipped", False):
                        counters[name]["fp"] += 1
                        
            record["extractions"][name] = {
                "cloze_results": res,
                "triplets":      trips,
            }
        records.append(record)
        time.sleep(0.01)
        
    # --- Compute and display metrics ---
    print(f"\n[4/4] Results:")
    print("=" * 65)
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("=" * 65)
    
    results_rows = []
    for name in loaded_models:
        c = counters[name]
        m = compute_metrics(c["tp"], c["fp"], c["fn"])
        print(f"{name:<25} {m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>10.4f}")
        results_rows.append({"Model": name, **m})
    print("=" * 65)
    
    print("\nDetailed TP / FP / FN:")
    for name in loaded_models:
        c = counters[name]
        print(f"  {name:<25}  TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  (TP+FN={c['tp']+c['fn']})")
        
    # --- Save outputs ---
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    pd.DataFrame(results_rows).to_csv(CSV_OUT, index=False)
    
    print(f"\nSaved -> {JSON_OUT}")
    print(f"Saved -> {CSV_OUT}\n")

if __name__ == "__main__":
    main()