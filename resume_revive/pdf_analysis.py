import fitz  # PyMuPDF
import language_tool_python
import PyPDF2

# Initialize the grammar checker tool
tool = language_tool_python.LanguageTool('en-US')

def count_pdf_pages(pdf_path):
    document = fitz.open(pdf_path)
    count = document.page_count
    document.close()
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
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


def check_grammar(text):
    matches = tool.check(text)
    return len(matches), matches
