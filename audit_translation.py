import json
import pandas as pd
from pathlib import Path


def main() -> None:
    path = Path("bluesky_data/bluesky_all_cleaned.csv")
    df = pd.read_csv(path)
    src = pd.read_csv("bluesky_data/bluesky_all.csv", usecols=["text"])

    out = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "has_text_en": "text_en" in df.columns,
        "has_lang": "lang" in df.columns,
    }

    if "lang" in df.columns:
        out["lang_top15"] = {k: int(v) for k, v in df["lang"].value_counts().head(15).items()}
        non_en_mask = (df["lang"] != "en") & (df["lang"] != "und")
        out["non_en_rows"] = int(non_en_mask.sum())
    else:
        non_en_mask = pd.Series([False] * len(df))

    if {"text_en", "text_clean", "lang"}.issubset(df.columns):
        if non_en_mask.any():
            same = df.loc[non_en_mask, "text_en"].fillna("") == df.loc[non_en_mask, "text_clean"].fillna("")
            out["non_en_same_as_text_clean"] = int(same.sum())
            out["non_en_changed"] = int((~same).sum())

    if "text_en" in df.columns:
        s = df["text_en"].fillna("")
        patterns = {
            "cjk": r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]",
            "arabic": r"[\u0600-\u06ff]",
            "cyrillic": r"[\u0400-\u04ff]",
            "devanagari": r"[\u0900-\u097f]",
            "thai": r"[\u0e00-\u0e7f]",
            "hebrew": r"[\u0590-\u05ff]",
            "greek": r"[\u0370-\u03ff]",
        }
        out["text_en_script_counts"] = {
            name: int(s.str.contains(pat, regex=True).sum()) for name, pat in patterns.items()
        }

        src_s = src["text"].fillna("").astype(str)
        out["source_text_script_counts"] = {
            name: int(src_s.str.contains(pat, regex=True).sum()) for name, pat in patterns.items()
        }

    # Example rows that still contain non-latin scripts after supposed translation.
    if {"text_en"}.issubset(df.columns):
        example_rows = []
        combo_pat = r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3\u0600-\u06ff\u0400-\u04ff\u0900-\u097f\u0e00-\u0e7f\u0590-\u05ff\u0370-\u03ff]"
        mask = df["text_en"].fillna("").str.contains(combo_pat, regex=True)
        cols = [c for c in ["lang", "text_clean", "text_en"] if c in df.columns]
        for _, row in df.loc[mask, cols].head(20).iterrows():
            example_rows.append({c: str(row[c]) for c in cols})
        out["examples_non_latin_in_text_en"] = example_rows

        source_examples = []
        src_mask = src["text"].fillna("").astype(str).str.contains(combo_pat, regex=True)
        for text in src.loc[src_mask, "text"].head(20):
            source_examples.append(str(text))
        out["examples_non_latin_in_source_text"] = source_examples

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
