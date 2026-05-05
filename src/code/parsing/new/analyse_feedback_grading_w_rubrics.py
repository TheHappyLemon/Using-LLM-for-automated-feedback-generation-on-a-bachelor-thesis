import pandas as pd
from collections import defaultdict
import unicodedata
import re
from src.data.constants import BASE_PATH
import os

FILE_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "gemma4_negative_feedback_analysis", "01_w_rubrics", "grading_with_rubrics_artkuc.csv") 

# Normalize text (remove accents, lowercase, strip punctuation)
def normalize(text):
    text = str(text).strip().lower()
    
    # Remove accents (e.g., "dalēji" → "daleji")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    return text

# Map normalized values to canonical answers
ANSWER_MAP = {
    "ja": "Ja",
    "j": "Ja",
    
    "daleji": "Daleji",
    "dal": "Daleji",
    
    "ne": "Ne",
    "n": "Ne"
}

# Load data
df = pd.read_csv(FILE_PATH)

# Drop first row (rubric labels)
df = df.iloc[1:].reset_index(drop=True)

# Drop ID column
df_data = df.drop(columns=["ID"])

# rubric_index - answer - count
stats = {i: defaultdict(int) for i in range(1, 6)}

for _, row in df_data.iterrows():
    values = row.tolist()

    for i in range(0, len(values), 5):
        chunk = values[i:i+5]

        if len(chunk) < 5:
            continue

        for rubric_idx, raw_answer in enumerate(chunk, start=1):
            if pd.isna(raw_answer):
                continue

            norm = normalize(raw_answer)

            if norm in ANSWER_MAP:
                answer = ANSWER_MAP[norm]
                stats[rubric_idx][answer] += 1
            else:
                pass

# Print results
for rubric_idx in range(1, 6):
    print(f"Rubric {rubric_idx}:")
    total = sum(stats[rubric_idx].values())

    for answer in ["Ja", "Daleji", "Ne"]:
        count = stats[rubric_idx][answer]
        pct = (count / total * 100) if total else 0
        print(f"  {answer}: {count} ({pct:.2f}%)")

    print()

output_rows = []

for rubric_idx in range(1, 6):
    total = sum(stats[rubric_idx].values())

    row = {"rubric": rubric_idx}

    for answer in ["Ja", "Daleji", "Ne"]:
        count = stats[rubric_idx][answer]
        pct = (count / total * 100) if total else 0

        #row[f"{answer}_count"] = count
        row[f"{answer}_pct"] = round(pct, 2)

    output_rows.append(row)

output_df = pd.DataFrame(output_rows)

OUTPUT_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "gemma4_negative_feedback_analysis", "01_w_rubrics", "rubric_summary.csv")
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved results to {OUTPUT_PATH}")