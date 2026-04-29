import fitz  # PyMuPDF

# Open PDF
pdf_path = "final_results.pdf"
doc = fitz.open(pdf_path)

# Select 8th page (index 7)
page = doc[7]

# Image path
img_path = "vertax stats.png"

# Get page size
page_width = page.rect.width
page_height = page.rect.height

# Desired image size
img_width = 614
img_height = 117

# Compute centered position
x0 = (page_width - img_width) / 2
y0 = page_height - img_height - 100
x1 = x0 + img_width
y1 = y0 + img_height

rect = fitz.Rect(x0, y0, x1, y1)

# Insert image
page.insert_image(rect, filename=img_path)

# Save PDF
doc.save("final_results_formatted.pdf")
doc.close()
