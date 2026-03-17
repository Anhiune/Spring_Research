#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("test_100rows_cleaned.csv")

# Check translation coverage for non-English rows
non_en = df[df['lang'] != 'en']
print(f"Total rows: {len(df)}")
print(f"Non-English (detected): {len(non_en)}")

if len(non_en) > 0:
    # Check if they were actually translated (text_en differs from text_clean)
    non_en = non_en.copy()
    non_en['was_translated'] = non_en['text_en'] != non_en['text_clean']
    translated_count = non_en['was_translated'].sum()
    not_translated = len(non_en) - translated_count
    
    print(f"\nTranslation results:")
    print(f"  Translated (text_en != text_clean): {translated_count}")
    print(f"  NOT translated (text_en = text_clean): {not_translated}")
    
    if translated_count > 0:
        print(f"\n=== SAMPLE TRANSLATED ROWS ===")
        for idx, (i, row) in enumerate(non_en[non_en['was_translated']].head(3).iterrows()):
            print(f"\n[Row {i}] Language: {row['lang']}")
            print(f"  Original: {repr(row['text_clean'][:100])}")
            print(f"  English:  {repr(row['text_en'][:100])}")
            print(f"  Translated? {row['text_en'] != row['text_clean']}")
