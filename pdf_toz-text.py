# Install dependency:
# pip install PyPDF2

from PyPDF2 import PdfReader

def pdf_to_text(pdf_path, output_txt):
    reader = PdfReader(pdf_path)
    
    with open(output_txt, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            f.write(f"--- Page {i} ---\n")
            f.write(text.strip() if text else "")  # write text or blank if none
            f.write("\n\n")  # separate pages

    print(f"✅ All pages processed. Text saved to '{output_txt}'")

# Example usage
if __name__ == "__main__":
    pdf_to_text("input.pdf", "output.txt")
