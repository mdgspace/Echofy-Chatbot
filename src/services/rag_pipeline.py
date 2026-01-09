import json
import os

from config import CHROMA_PATH, DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_RETRIEVAL_K
from src.data.loader import load_qa_pairs
from src.retrieval.chroma_service import load_chroma, save_to_chroma


def initialize_database(embedding_function):
    """Initialize the vector database if it doesn't exist."""
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
        documents = load_qa_pairs()
        save_to_chroma(persist_directory=CHROMA_PATH, chunks=documents, embedding_function=embedding_function)


def retrieve_contexts(query_text, embedding_function, k=DEFAULT_RETRIEVAL_K, threshold=DEFAULT_RELEVANCE_THRESHOLD):
    """Retrieve and filter relevant contexts for the query."""

    # Load ChromaDB and Fetch Relevant Results
    db = load_chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=k)

    # Return empty list in case of no results found
    if len(results) == 0 :
        print(f"\nUnable to find matching results for '{query_text}'")
        return []
    
    contexts = []
    # Filter by threshold
    for doc, _score in results:
        if _score >= threshold:
            contexts.append({
                "context": doc.page_content,
                "relevance_score": str(_score)
            })

    return contexts


def create_prompt(query_text, contexts):
    """Create the prompt with retrieved contexts."""
    prompt = f"""
        You are an intelligent language model designed to answer questions using only a list of retrieved text passages, each tagged with a relevance score.

        Your objective is to:
        - **Carefully read** the provided contexts and select only the most relevant one(s) to answer the question.
        - **Do not rely on prior knowledge** or external information. Your answer must be grounded strictly in the given contexts.

        Guidelines:
        - The **higher the score**, the more importance that context holds in shaping your answer.
        - **Do not use prior knowledge** or any information outside the provided contexts.
        - **Respond in 2–3 short, clear sentences** forming a brief paragraph.
        - Your answer should be plain text—**never include the contexts or scores** in your response.
        - If the question is unrelated to MDG Space, respond with: "I can only provide information related to MDG Space. Please ask a question within this scope."
        - If the question is related to MDG Space, the answer **must directly address it using the provided context**—no exceptions.
        - The response **must always be confident** and **must never suggest uncertainty** or imply that it is "trying" to answer the question.
        - **Hallucination is strictly prohibited**—only use the information available in the retrieved context.
        - Treat similar names as the same entity. For example:
            - "mdg", "mdg group", "mdg space", "mdgspace", "MDG Space" all refer to the same entity.
            - Similarly, for projects: "Security app project", "security project", "security app" all refer to the same project.
        - If the user greets (e.g., "hi", "hello"), respond with a friendly greeting like: "Hello! How can I assist you today?"

        Here is the list of contexts with relevance scores:

        {json.dumps(contexts, indent=4)}

        Now, answer the following question based only on the valid context(s):

        Question: {query_text}
    """
    return prompt
