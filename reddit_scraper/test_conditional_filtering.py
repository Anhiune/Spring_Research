import os
import sys
import pandas as pd
import spacy
from datetime import datetime

# Add the project directory to sys.path to import from the scraper script
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Note: For testing purposes, we define the core logic here or import it
# Since the scraper is a standalone script, we'll re-define the logic to test it independently
# or import functions if needed. 

# Let's import the actual functions from the script to ensure we're testing the real logic
import scrape_reddit_pullpush as scraper

def test_load_keywords():
    print("Testing keyword loading...")
    general = scraper.GENERAL_KEYWORDS
    ontopic = scraper.ONTOPIC_KEYWORDS
    print(f"Loaded {len(general)} general keywords.")
    print(f"Loaded {len(ontopic)} on-topic keywords.")
    assert len(general) > 0
    assert len(ontopic) > 0
    print("✅ Keyword loading test passed.")

def test_filtering_logic():
    print("\nTesting conditional filtering logic...")
    
    # Enable conditional filtering
    scraper.USE_CONDITIONAL_FILTERING = True
    
    test_cases = [
        {
            "text": "NVIDIA faces new export controls from China.",
            "expected": True,
            "desc": "Both policy (export controls/China) and company (NVIDIA)"
        },
        {
            "text": "NVIDIA released new GPU today.",
            "expected": False,
            "desc": "Only company (NVIDIA/GPU), no policy"
        },
        {
            "text": "New export controls announced by the BIS today.",
            "expected": False,
            "desc": "Only policy (export controls/BIS), no company"
        },
        {
            "text": "The price of eggs in China is rising.",
            "expected": False,
            "desc": "Only policy keyword (China) but irrelevent context"
        },
        {
            "text": "SMIC is being added to the entity list by the US government.",
            "expected": True,
            "desc": "Both policy (entity list) and company keyword (SMIC)"
        }
    ]
    
    for i, case in enumerate(test_cases):
        sentences = [case["text"]]
        filtered = scraper.filter_sentences_for_keywords(sentences, scraper.GENERAL_KEYWORDS, scraper.ONTOPIC_KEYWORDS)
        result = len(filtered) > 0
        if result == case["expected"]:
            print(f"Case {i+1}: ✅ Passed - {case['desc']}")
        else:
            print(f"Case {i+1}: ❌ Failed - {case['desc']} (Expected: {case['expected']}, Got: {result})")

def test_company_detection():
    print("\nTesting company detection logic...")
    test_cases = [
        ("NVIDIA and AMD are competing in the AI chip market.", ["NVDA", "AMD"]),
        ("Intel is building a new fab in Ohio.", ["Intel"]),
        ("TSMC is at the center of geopolitical tension.", ["TSMC"]),
        ("New sanctions on China affecting semiconductor production.", ["Other"]),
        ("ASML lithography machines are restricted.", ["ASML"])
    ]
    
    for i, (text, expected) in enumerate(test_cases):
        detected = scraper.detect_companies(text)
        if set(detected) == set(expected):
            print(f"Case {i+1}: ✅ Passed - '{text}' -> {detected}")
        else:
            print(f"Case {i+1}: ❌ Failed - '{text}' -> Expected {expected}, got {detected}")

def test_excel_creation():
    print("\nTesting Excel file creation...")
    # Mock some data
    scraper.rows = [
        {
            'submission_id': 's1', 'comment_id': None, 'timestamp': 123456, 
            'subreddit': 'stocks', 'text_type': 'title', 
            'text': 'NVIDIA faces new export controls from China.'
        },
        {
            'submission_id': 's2', 'comment_id': None, 'timestamp': 123457, 
            'subreddit': 'AMD_Stock', 'text_type': 'title', 
            'text': 'AMD restricted from selling to SMIC.'
        }
    ]
    
    # Save results (this should create CSV and Excel)
    scraper.save_results()
    
    # Check if files were created
    output_dir = "output"
    files = os.listdir(output_dir)
    excel_files = [f for f in files if f.endswith(".xlsx") and f.startswith("reddit_")]
    csv_files = [f for f in files if f.endswith(".csv") and f.startswith("reddit_")]
    
    if excel_files:
        print(f"✅ Excel file created: {excel_files[-1]}")
        # Test if we can read the sheets
        xlsx_path = os.path.join(output_dir, excel_files[-1])
        with pd.ExcelFile(xlsx_path) as xls:
            sheets = xls.sheet_names
            print(f"Sheets found: {sheets}")
            if 'All_Data' in sheets and 'NVDA' in sheets and 'AMD' in sheets:
                print("✅ All expected sheets found in Excel file.")
            else:
                print("❌ Some sheets missing in Excel file.")
    else:
        print("❌ Excel file not found.")

if __name__ == "__main__":
    if not os.path.exists("output"):
        os.makedirs("output")
    
    try:
        test_load_keywords()
        test_filtering_logic()
        test_company_detection()
        test_excel_creation()
        print("\nAll tests completed.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
