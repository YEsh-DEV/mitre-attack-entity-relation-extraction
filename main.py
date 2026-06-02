"""
MITRE ATT&CK Entity & Relation Extraction
SecureBERT + CTI-BERT | Cloze-Task Evaluation
All outputs -> Y:/Reserchintern/Experiment1/
"""

import os, json, re, time
import pandas as pd
import torch
from transformers import pipeline

# ==========================================
# PATHS
# ==========================================
BASE_DIR  = r"Y:/Reserchintern/Experiment1"
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
JSON_OUT  = os.path.join(BASE_DIR, "entity_relationship_output.json")
CSV_OUT   = os.path.join(BASE_DIR, "evaluation_results.csv")

os.environ["HF_HOME"]                        = CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SECURE_SNAP = os.path.join(CACHE_DIR, "hub/models--ehsanaghaei--SecureBERT/snapshots/3a47918dd874e5c769efd152b51c5756a953fb67")
CTI_SNAP    = os.path.join(CACHE_DIR, "hub/models--ibm-research--CTI-BERT/snapshots/4cd0a0edf150e063811e6d2f5022fa6b3c9cc9e3")

# ==========================================
# 1. LOAD & MERGE
# ==========================================
print("[1/5] Loading and merging datasets...")
attack_df     = pd.read_excel(os.path.join(BASE_DIR, "Attackmitre.xlsx"))
enterprise_df = pd.read_excel(os.path.join(BASE_DIR, "MitreEnterprise.xlsx"))
attack_df.columns     = attack_df.columns.str.strip()
enterprise_df.columns = enterprise_df.columns.str.strip()

exp = attack_df.copy()
exp["Group Techniques"] = exp["Group Techniques"].str.split(";")
exp = exp.explode("Group Techniques")
exp["Group Techniques"] = exp["Group Techniques"].str.strip()
exp = exp[exp["Group Techniques"].notna() & (exp["Group Techniques"] != "")]

merged = pd.merge(
    exp,
    enterprise_df[["Tactic ID", "Tactic Name", "Description"]],
    left_on="Group Techniques", right_on="Tactic ID", how="inner"
)
merged = merged[merged["Description"].notna()]
df = merged[["APT Group Name", "Tactic ID", "Tactic Name", "Description"]].drop_duplicates().reset_index(drop=True)
print(f"    Unique APT+Tactic pairs: {len(df)}  |  APT groups: {df['APT Group Name'].nunique()}  |  Tactics: {df['Tactic ID'].nunique()}")

# ==========================================
# 2. GROUND TRUTH TOKENS
# ==========================================
# Returns list of (clean_form, original_word) tuples.
# clean_form  = alphanumeric lowercase, used for predicted-token comparison.
# original_word = exact word from tactic name, used for regex masking in probe.
#
# STOP list excludes:
#   - generic English words (with, have, ...)
#   - ultra-common tactic suffixes that ANY cybersecurity model trivially predicts
#     ('discovery' appears in 13% of all rows -> inflates scores unfairly)
STOP = {
    "with","have","from","their","that","this","they","used","which","such",
    "these","into","been","also","more","than","other","both","system","systems",
    "network","information","data","access","using","target","attack","process",
    "service","software","file","user","files","users","based","local","remote",
    "within","discovery",
}

def get_gt_pairs(tactic_name: str) -> list:
    """Returns [(clean_token, original_word), ...] for each meaningful word."""
    pairs = []
    for word in tactic_name.split():
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if len(clean) >= 4 and clean not in STOP and clean.isalpha():
            pairs.append((clean, word))
    return pairs

df["gt_pairs"] = df["Tactic Name"].apply(get_gt_pairs)

# ==========================================
# 3. LOAD MODELS
# ==========================================
device = 0 if torch.cuda.is_available() else -1
print(f"[2/5] Hardware: {'GPU' if device == 0 else 'CPU'}")

def load_pipe(snap, hub_id, label):
    name = snap if os.path.exists(snap) else hub_id
    print(f"    Loading {label} from: {name}")
    try:
        p = pipeline("fill-mask", model=name, tokenizer=name, device=device)
        print(f"    [OK] {label}")
        return p
    except Exception as e:
        print(f"    [FAIL] {label}: {e}")
        return None

print("[3/5] Loading models...")
sb_pipe  = load_pipe(SECURE_SNAP, "ehsanaghaei/SecureBERT", "SecureBERT")
cti_pipe = load_pipe(CTI_SNAP,    "ibm-research/CTI-BERT",  "CTI-BERT")

# ==========================================
# 4. CLOZE-TASK EXTRACTION
#
# For each GT (clean_tok, original_word) pair:
#   - Mask the ORIGINAL word in the tactic name (handles hyphenated words correctly)
#   - Build a single-mask probe using description for context (harder, more accurate eval)
#   - Predict top-K; check if clean_tok appears in predicted tokens
#
# Metrics:
#   TP = GT clean token found in model's top-K predictions
#   FP = model produced a top-1 prediction that is NOT the GT
#   FN = GT not found in top-K predictions
#   Skipped probes (mask count != 1) counted as FN only (model failed)
# ==========================================
TOP_K = 10

def clean_pred(raw: str) -> str:
    """Normalise predicted token to alphanumeric lowercase."""
    return re.sub(r"[^a-z]", "", str(raw).replace("Ġ", "").replace("##", "").lower())

def run_cloze(pipe, mask_token, apt, tactic_name, desc, gt_pairs):
    results  = []
    triplets = []

    for clean_gt, orig_word in gt_pairs:
        # Mask the original word form (fixes hyphenated words like 'Command-Line')
        masked_tactic = re.sub(
            re.escape(orig_word), mask_token,
            tactic_name, flags=re.IGNORECASE, count=1
        )

        # SINGLE-MASK probe: tactic context + first 100 chars of description
        desc_snippet = re.sub(r"\s+", " ", desc[:100]).strip()
        probe = f"{apt} performs {masked_tactic}. {desc_snippet}"

        # Safety: ensure exactly one mask token
        if probe.count(mask_token) != 1:
            # Fallback: minimal probe
            probe = f"The attack technique {masked_tactic} is used by {apt}."

        if probe.count(mask_token) != 1:
            # Still broken (e.g. special chars prevented masking) -> FN
            results.append({"gt": clean_gt, "top1": None, "hit": False, "skipped": True})
            continue

        try:
            raw_preds = pipe(probe, top_k=TOP_K)
        except Exception as e:
            print(f"      [pipe error] {e}")
            results.append({"gt": clean_gt, "top1": None, "hit": False, "skipped": True})
            continue

        # Pipeline with single mask -> list of dicts. Guard against nested list.
        if raw_preds and isinstance(raw_preds[0], list):
            raw_preds = raw_preds[0]

        predicted = [
            clean_pred(p.get("token_str", "") or p.get("word", ""))
            for p in raw_preds if isinstance(p, dict)
        ]

        top1 = predicted[0] if predicted else None
        hit  = clean_gt in predicted  # GT found in top-K

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


print(f"[4/5] Running cloze extraction on {len(df)} unique APT+Tactic pairs...")

records = []
sb_tp  = sb_fp  = sb_fn  = 0
cti_tp = cti_fp = cti_fn = 0

for i, row in df.iterrows():
    apt     = row["APT Group Name"]
    tid     = row["Tactic ID"]
    tname   = row["Tactic Name"]
    desc    = str(row["Description"])
    pairs   = row["gt_pairs"]

    if not pairs:
        continue

    print(f"  [{i+1}/{len(df)}] {apt} | {tid} | {tname}")

    sb_res,  sb_trips  = run_cloze(sb_pipe,  "<mask>", apt, tname, desc, pairs) if sb_pipe  else ([], [])
    cti_res, cti_trips = run_cloze(cti_pipe, "[MASK]", apt, tname, desc, pairs) if cti_pipe else ([], [])

    for r in sb_res:
        if r["hit"]:
            sb_tp += 1
        else:
            sb_fn += 1
            if r["top1"] and not r["skipped"]:
                sb_fp += 1

    for r in cti_res:
        if r["hit"]:
            cti_tp += 1
        else:
            cti_fn += 1
            if r["top1"] and not r["skipped"]:
                cti_fp += 1

    records.append({
        "apt_group":   apt,
        "tactic_id":   tid,
        "tactic_name": tname,
        "description": desc,
        "gt_tokens":   [p[0] for p in pairs],
        "SecureBERT":  {"cloze_results": sb_res,  "triplets": sb_trips},
        "CTI-BERT":    {"cloze_results": cti_res, "triplets": cti_trips},
    })
    time.sleep(0.01)

# ==========================================
# 5. METRICS & SAVE
# ==========================================
def compute_metrics(tp, fp, fn):
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p+r)  if (p  + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)

sb_p,  sb_r,  sb_f1  = compute_metrics(sb_tp,  sb_fp,  sb_fn)
cti_p, cti_r, cti_f1 = compute_metrics(cti_tp, cti_fp, cti_fn)

print("\n[5/5] Results:")
print("=" * 55)
print(f"{'Model':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("=" * 55)
print(f"{'SecureBERT':<15} {sb_p:>10.4f} {sb_r:>10.4f} {sb_f1:>10.4f}")
print(f"{'CTI-BERT':<15}   {cti_p:>10.4f} {cti_r:>10.4f} {cti_f1:>10.4f}")
print("=" * 55)
print(f"\nTP/FP/FN  SecureBERT: {sb_tp}/{sb_fp}/{sb_fn}  |  CTI-BERT: {cti_tp}/{cti_fp}/{cti_fn}")

os.makedirs(BASE_DIR, exist_ok=True)

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4)

pd.DataFrame([
    {"Model": "SecureBERT", "Precision": sb_p,  "Recall": sb_r,  "F1": sb_f1,
     "TP": sb_tp,  "FP": sb_fp,  "FN": sb_fn},
    {"Model": "CTI-BERT",   "Precision": cti_p, "Recall": cti_r, "F1": cti_f1,
     "TP": cti_tp, "FP": cti_fp, "FN": cti_fn},
]).to_csv(CSV_OUT, index=False)

print(f"\nSaved -> {JSON_OUT}")
print(f"Saved -> {CSV_OUT}")