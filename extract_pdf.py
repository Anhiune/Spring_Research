from pypdf import PdfReader
import os

pdf_path = r"c:/Users/hoang/OneDrive - University of St. Thomas/Anh Bui UROP Research Report for Spring Semester 2025.pdf"
output_path = "extracted_content.txt"

reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n\n"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Extracted text to {output_path}")
