from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            full_text += f"\n\n[Page {page_number}]\n{page_text}"

    return full_text