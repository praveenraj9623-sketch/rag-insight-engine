import re


def clean_text(text):
    """
    Basic NLP cleaning:
    - Remove extra spaces
    - Remove special characters
    """
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'[^\w\s.,!?]', '', text)  # keep basic punctuation
    return text.strip()


def get_document_stats(text):
    """
    Returns basic stats for document
    """
    words = text.split()
    sentences = re.split(r'[.!?]', text)

    return {
        "total_words": len(words),
        "total_sentences": len([s for s in sentences if s.strip()]),
        "total_characters": len(text)
    }