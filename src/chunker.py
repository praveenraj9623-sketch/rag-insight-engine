from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_chunks(text):
    """
    Splits text into chunks for vector database
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    documents = []

    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"chunk_id": i}
        )
        documents.append(doc)

    return documents