import re
import pandas as pd, ast
from pathlib import Path

def clean(text):
    """Standardize whitespace / remove weird tokens."""
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"\b_+\b", "", text)

    # Remove sequences like: ___-year-old, ___cm, ___ mm
    text = re.sub(r"_+(?=[A-Za-z0-9])", "", text)

    # Remove formats like: (_, ___), ____, ____, 
    text = re.sub(r"_+", "", text)

    # Cleanup extra spaces after removal
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text.strip()

def extract_section(text, header):
    """
    Extract section that starts with `header:` until next ALL-CAPS header
    (e.g., IMPRESSION, FINDINGS, etc.)
    """
    pattern = rf"{header}\s*:?(.*?)(?=(?:[A-Z ]{{3,}}:)|$)"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return clean(m.group(1).strip())
    return None

def extract_last_paragraph(text):
    """Fallback: return last paragraph of report."""
    paras = [p.strip() for p in text.split("\n") if p.strip() != ""]
    if len(paras) >= 1:
        return clean(paras[-1])
    return ""

def extract_impression(text: str) -> str:
    # Use regex to find the "Impression" section
    text = clean(text)

    # if impression exists 
    imp = extract_section(text, "IMPRESSION")
    if imp:
        return imp
    
    # impression -> conclusion 
    conclusion = extract_section(text, "CONCLUSION")
    if conclusion:
        return conclusion
    
    # last paragraph(not tagged but usually impression)
    last = extract_last_paragraph(text)
    if len(last.split(".")) <= 10 :
        return last
    
    # fallback to conclusion 
    findings = extract_section(text, "FINDINGS")
    if findings:
        return findings

    return text.strip()

def main():
    df = pd.read_csv("subset_study_manifest.csv")
    records = []
    for _, row in df.iterrows():
        report_path = row['path']
        text = Path("../Data/Reports").joinpath(report_path).read_text()
        impression = extract_impression(text)
        images = ast.literal_eval(row["image_paths"])
        #images = [s.split(".")[0]+".jpg" for s in images] for jpg
        #print("Converted images:", images) for jpg
        
        records.append({
            "study_id": row["study_id"],
            "image_paths": images,
            "impression": impression,
            "split": row["split"]
        })
        
    pd.DataFrame(records).to_csv("mimic_dicom_impression_subset.csv", index=False)


if __name__ == "__main__":
    main()
