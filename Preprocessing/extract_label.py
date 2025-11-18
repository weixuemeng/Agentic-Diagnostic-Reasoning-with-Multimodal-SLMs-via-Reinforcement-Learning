import pandas as pd

CHEXPERT_CSV = "../Data/mimic-cxr-2.0.0-chexpert.csv"

LABEL_COLS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "No Finding","Pleural Effusion","Pleural Other","Pneumonia",
    "Pneumothorax","Support Devices"
]

def normalize_label(v):
    if pd.isna(v):
        return 0
    if v == -1.0: # uncertain
        return 0   # or 0.5 if you want soft reward
    return int(v)

def load_chexpert_labels(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    lookup = {}
    for _, row in df.iterrows():
        sid = int(row["study_id"])
        labels = {col: normalize_label(row[col]) for col in LABEL_COLS}
        lookup[sid] = labels
    return lookup

# Build the dictionary once
chexpert_lookup = load_chexpert_labels(CHEXPERT_CSV)

# Example usage:
# study_id = 50414267
# print(chexpert_lookup[study_id])