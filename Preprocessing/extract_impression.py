import re
import pandas as pd, ast
from pathlib import Path

def extract_impression(text: str) -> str:
    # Use regex to find the "Impression" section
    match = re.search(r'(?i)(impression:)(.*?)(\n[A-Z][a-z]+:|\Z)', text, re.DOTALL)
    if match:
        impression = match.group(2).strip()
        return impression
    else:
        # fallback to conclusion 
        findings = re.search(r'(?i)(findings:\s*:?\s*)([\s\S]*?)(?=(\n[A-Z][A-Z ]{2,}:|$))', text)
        if findings:
            return findings.group(2).strip()

        return text.strip()

def main():
    df = pd.read_csv("subset_study_manifest.csv")
    records = []
    for _, row in df.iterrows():
        report_path = row['path']
        text = Path("../Data/Reports").joinpath(report_path).read_text()
        impression = extract_impression(text)
        images = ast.literal_eval(row["image_paths"])
        images = [s.split(".")[0]+".jpg" for s in images]
        print("Converted images:", images)
        
        records.append({
            "study_id": row["study_id"],
            "image_paths": images,
            "impression": impression,
            "split": row["split"]
        })
        
    pd.DataFrame(records).to_csv("mimic_jpg_impression_subset.csv", index=False)


if __name__ == "__main__":
    main()