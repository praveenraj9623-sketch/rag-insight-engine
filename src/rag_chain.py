import google.generativeai as genai
from src.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash-lite")


RETRIEVAL_THRESHOLD = 1.20


def keyword_search_all_chunks(vector_store, question):
    matched_docs = []

    try:
        all_docs = list(vector_store.docstore._dict.values())

        for doc in all_docs:
            if question.lower() in doc.page_content.lower():
                matched_docs.append((doc, 0.0))

    except Exception:
        pass

    return matched_docs


def retrieve_docs_with_scores(vector_store, question, k=6):
    keyword_docs = keyword_search_all_chunks(vector_store, question)

    if keyword_docs:
        return keyword_docs[:k]

    docs_with_scores = vector_store.similarity_search_with_score(question, k=k)

    filtered_docs = [
        (doc, score)
        for doc, score in docs_with_scores
        if score <= RETRIEVAL_THRESHOLD
    ]

    return filtered_docs if filtered_docs else docs_with_scores[:3]


def calculate_confidence(docs_with_scores):
    if not docs_with_scores:
        return "Low"

    best_score = min(score for _, score in docs_with_scores)

    if best_score <= 1.05:
        return "High"
    elif best_score <= 1.20:
        return "Medium"
    else:
        return "Low"


def format_docs(docs_with_scores):
    context = ""

    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        context += f"""
Source Chunk {i}
Similarity Distance: {score}
Content:
{doc.page_content}
"""

    return context


def answer_question(vector_store, question, k=6):
    docs_with_scores = retrieve_docs_with_scores(vector_store, question, k=k)

    confidence = calculate_confidence(docs_with_scores)

    context = format_docs(docs_with_scores)

    prompt = f"""
You are an AI document assistant.

Answer the user question using ONLY the provided context.
Do not hallucinate.
If the answer is not present in the context, say:
"I could not find this information in the uploaded document."

Give a clear, professional answer.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)

    return {
        "answer": response.text.strip(),
        "source_documents": docs_with_scores,
        "confidence": confidence
    }