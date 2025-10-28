import pandas as pd

def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(name)

def main():
    cxr_records = pd.read_csv("../Data/cxr-record-list.csv")
    cxr_studies = pd.read_csv("../Data/cxr-study-list.csv")
    print(cxr_records.head())
    print(cxr_records.columns)
    print(cxr_studies.head())
    print(cxr_studies.columns)
    # record image path by study as a list [path1, path2,...]
    image_by_study = (cxr_records.groupby("study_id")["path"].apply(list).reset_index(name = "image_paths"))
    merged_records = pd.merge(cxr_studies, image_by_study, on="study_id", how="inner")

    print("Total studies:", len(merged_records))
    print(merged_records.head())

    merged_records.to_csv("../Data/study_manifest.csv", index=False)
    print("Save study_manifest.csv...")

if __name__ == "__main__":
    main()