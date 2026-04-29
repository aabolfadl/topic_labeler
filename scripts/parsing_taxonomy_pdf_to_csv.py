import os
import pandas as pd
from pypdf import PdfReader


def parse_pdf_to_csv(pdf_filename):
    # Check if file exists
    if not os.path.exists(pdf_filename):
        print(
            f"Error: The file '{pdf_filename}' was not found in the current directory."
        )
        return

    print(f"Reading {pdf_filename}...")

    # Initialize the PDF reader
    try:
        reader = PdfReader(pdf_filename)
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return

    extracted_data = []

    # Iterate through every page in the PDF
    for page in reader.pages:
        text = page.extract_text()

        if text:
            # Split the page text into individual lines
            lines = text.split("\n")

            for line in lines:
                # User Requirement: Only process lines containing a colon
                if ":" in line:
                    # Split by colon and strip whitespace from each part
                    parts = [part.strip() for part in line.split(":")]

                    # Ensure we have exactly 3 columns (Granularity Levels)
                    # If line is "A: B", result is [A, B, None]
                    # If line is "A: B: C", result is [A, B, C]

                    level_1 = parts[0] if len(parts) > 0 else None
                    level_2 = parts[1] if len(parts) > 1 else None
                    level_3 = parts[2] if len(parts) > 2 else None

                    extracted_data.append([level_1, level_2, level_3])

    # Create a DataFrame
    df = pd.DataFrame(extracted_data, columns=["Level 1", "Level 2", "Level 3"])

    # Clean data: Drop empty rows if any were created effectively
    df.dropna(how="all", inplace=True)

    # Generate Output Filename (same name as PDF but with .csv extension)
    output_filename = os.path.splitext(pdf_filename)[0] + ".csv"

    # Save to CSV in the same directory
    df.to_csv(output_filename, index=False, encoding="utf-8-sig")

    print(f"Success! Processed {len(df)} rows.")
    print(f"Saved to: {os.path.abspath(output_filename)}")


# --- EXECUTION ---
if __name__ == "__main__":
    # REPLACE THIS WITH YOUR PDF NAME
    target_pdf = "Taxonomy.pdf"

    parse_pdf_to_csv(target_pdf)
