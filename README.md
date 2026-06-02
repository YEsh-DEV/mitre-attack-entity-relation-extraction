# MITRE ATT&CK Entity Relation Extraction

Entity and Relationship Extraction from the MITRE ATT&CK Dataset using Cyber Threat Intelligence Language Models.

## Overview

This project implements an automated evaluation framework for extracting cybersecurity entities and relationships from MITRE ATT&CK data using domain-specific NLP models.

The system combines MITRE ATT&CK datasets, generates structured evaluation probes, runs masked language model inference, extracts entity relationships, and computes performance metrics including Precision, Recall, and F1-Score.

The primary objective is to benchmark cybersecurity language models on their ability to recover MITRE ATT&CK tactic information from contextual threat intelligence text.

---

## Features

- Automated MITRE ATT&CK dataset processing
- Entity and relationship extraction pipeline
- Cloze-task based evaluation methodology
- Model benchmarking framework
- Precision, Recall, and F1-Score calculation
- JSON and CSV result export
- Offline Hugging Face model support
- Reproducible evaluation workflow

---

## Models Evaluated

### SecureBERT

- Model: `ehsanaghaei/SecureBERT`
- Architecture: RoBERTa-based Masked Language Model
- Domain: Cybersecurity Text

### CTI-BERT

- Model: `ibm-research/CTI-BERT`
- Architecture: BERT-based Masked Language Model
- Domain: Cyber Threat Intelligence (CTI)

### Models Not Evaluated

#### BERT-BiLSTM-CRF

Could not be evaluated because the publicly available repository did not contain the required trained model weights.

#### LLaMA-3.1-8B-Instruct

Could not be evaluated due to network restrictions preventing model download and hardware limitations for local inference.

---

## Dataset

The experiment uses two MITRE ATT&CK derived datasets.

### Attackmitre.xlsx

Contains:

- APT Group Name
- Group Techniques
- Software ID
- Software Techniques
- Software References

### MitreEnterprise.xlsx

Contains:

- Tactic ID
- Tactic Name
- Description
- Mitigation Steps
- Examples

### Dataset Source

https://ieee-dataport.org/documents/mitre-attack-dataset-knowledge-graph-enhanced-rag-cyber-threat-intelligence

---

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── README.md
├── Attackmitre.xlsx
├── MitreEnterprise.xlsx
├── entity_relationship_output.json
└── evaluation_results.csv
```

---

## Methodology

### Data Preparation

The `Group Techniques` column contains semicolon-separated ATT&CK technique IDs.

Example:

```text
T1057;T1135;T1003
```

The pipeline splits and explodes these values so each ATT&CK technique becomes an individual row.

Example:

```python
attack_df["Group Techniques"] = attack_df["Group Techniques"].str.split(";")
attack_df = attack_df.explode("Group Techniques")
attack_df["Group Techniques"] = attack_df["Group Techniques"].str.strip()
```

The resulting dataset is merged with MITRE ATT&CK descriptions using ATT&CK technique IDs.

---

### Cloze-Task Evaluation

The evaluated models are Masked Language Models (MLMs).

Instead of performing direct Named Entity Recognition (NER), the experiment evaluates them using a cloze-task methodology.

Example:

```text
APT1 performs Pass the [MASK].
```

Expected prediction:

```text
Hash
```

If the correct token appears within the model's Top-10 predictions, the prediction is considered successful.

---

## Evaluation Metrics

Metrics are calculated using standard information retrieval formulas.

```text
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × Precision × Recall / (Precision + Recall)
```

Where:

- TP = True Positives
- FP = False Positives
- FN = False Negatives

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YEsh-DEV/mitre-attack-entity-relation-extraction.git

cd mitre-attack-entity-relation-extraction
```

### Create Virtual Environment

#### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Experiment

Execute:

```bash
python main.py
```

The script performs:

1. Loads Attackmitre.xlsx and MitreEnterprise.xlsx
2. Explodes semicolon-separated ATT&CK technique IDs
3. Merges ATT&CK techniques with descriptions
4. Generates cloze-task evaluation probes
5. Runs SecureBERT and CTI-BERT inference
6. Extracts entity relationships
7. Computes Precision, Recall, and F1-Score
8. Saves evaluation outputs

---

## Generated Outputs

After execution, the following files are generated:

### entity_relationship_output.json

Contains:

- Extracted entity relationships
- Model predictions
- Ground-truth tokens
- Evaluation metadata

Example:

```json
{
  "entity_1": "APT1",
  "relationship": "USES",
  "entity_2": "hash",
  "correct": true
}
```

### evaluation_results.csv

Contains:

- Model
- Precision
- Recall
- F1-Score
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)

---

## Results

### Performance Comparison

| Model | Precision | Recall | F1-Score |
|---------|---------|---------|---------|
| SecureBERT | 0.6904 | 0.6900 | 0.6902 |
| CTI-BERT | 0.7549 | 0.7459 | 0.7504 |

### Detailed Counts

| Model | TP | FP | FN |
|---------|---------|---------|---------|
| SecureBERT | 1271 | 570 | 571 |
| CTI-BERT | 1374 | 446 | 468 |

### Best Performing Model

CTI-BERT achieved the highest performance:

```text
Precision : 0.7549
Recall    : 0.7459
F1 Score  : 0.7504
```

The results indicate that CTI-specific pretraining provides a measurable advantage for MITRE ATT&CK entity prediction tasks.

---

## Challenges Encountered

### Semicolon-Separated ATT&CK IDs

The ATT&CK techniques were stored as semicolon-separated values, preventing direct joins.

**Solution**

- Split values
- Explode rows
- Merge on individual ATT&CK IDs

### Circular Ground Truth

An early implementation leaked answers into prompts, producing unrealistically perfect scores.

**Solution**

- Build ground truth from ATT&CK tactic keywords
- Mask only target tokens

### Multiple Mask Tokens

MLM pipelines fail when multiple mask tokens exist in a single prompt.

**Solution**

- Enforce exactly one mask token per evaluation probe

### Windows Symlink Restrictions

Offline Hugging Face cache resolution failed because Windows blocked symbolic links.

**Solution**

- Implement direct snapshot-folder path resolution

---

## Requirements

Main libraries used:

```text
transformers
torch
pandas
numpy
openpyxl
scikit-learn
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## References

### MITRE ATT&CK

https://attack.mitre.org

### SecureBERT

https://huggingface.co/ehsanaghaei/SecureBERT

### CTI-BERT

https://huggingface.co/ibm-research/CTI-BERT

### Dataset Source

https://ieee-dataport.org/documents/mitre-attack-dataset-knowledge-graph-enhanced-rag-cyber-threat-intelligence

---

## Author

**Yesh**

Research Project: MITRE ATT&CK Entity and Relationship Extraction using Cyber Threat Intelligence Language Models.
