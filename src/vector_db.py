from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_faiss_vector_store(documents):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )

    return vector_store


def save_vector_store(vector_store):
    vector_store.save_local("vector_store/faiss_index")