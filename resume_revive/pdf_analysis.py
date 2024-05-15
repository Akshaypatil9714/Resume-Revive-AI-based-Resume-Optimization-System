import fitz  # PyMuPDF, a Python library for working with PDF files (reading, writing, modifying).
import language_tool_python # Library for performing grammar checks using LanguageTool.
import PyPDF2 # Python library to work with PDF files: extracting text, metadata, etc.

# Initialize the grammar checker tool
tool = language_tool_python.LanguageTool('en-US')

def count_pdf_pages(pdf_path):
    document = fitz.open(pdf_path) # Open the PDF file using fitz (PyMuPDF).
    count = document.page_count # Retrieve the total count of pages in the PDF.
    document.close() # Close the PDF document to free resources.
    return count

# def extract_text_from_pdf(pdf_path):
#     document = fitz.open(pdf_path)
#     text = ''
#     for page_num in range(document.page_count):
#         page = document.load_page(page_num)
#         text += page.get_text("text")
#     document.close()
#     return text



def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file: # Open the PDF file in binary mode.
        reader = PyPDF2.PdfReader(file) # Initialize a PDF reader object.
        text = ''
        for page in reader.pages: # Iterate through each page in the PDF file.
            text += page.extract_text() # Extract and concatenate text from each page.
    return text # Return the concatenated text from all pages.


def check_grammar(text):
    matches = tool.check(text) # Use LanguageTool to find grammar mistakes in the text.
    return len(matches), matches # Return the number of mistakes and the details of these mistakes.
