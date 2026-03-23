import json
import os

from config import CHROMA_PATH, DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_RETRIEVAL_K
from src.data.loader import load_qa_pairs
from src.retrieval.chroma_service import load_chroma, save_to_chroma
import logging

logger = logging.getLogger(__name__)


def initialize_database(embedding_function):
    """Initialize the vector database if it doesn't exist."""
    is_empty = True
    if os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH):
        try:
            db = load_chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            if db._collection.count() > 0:
                is_empty = False
        except Exception as e:
            logger.warning(f"Could not verify Chroma collection size, assuming empty: {e}")
            
    if is_empty:
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
        logger.warning(f"Unable to find matching results for '{query_text}'")
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
        You are a friendly and knowledgeable assistant for MDG Space, a tech club. Your goal is to help users by answering their questions naturally and conversationally.

        **How to use the provided contexts:**
        - You have access to relevant information passages, each with a relevance score
        - Higher scores mean that context is more important for answering the question
        - Use ONLY the information from these contexts—never make up information or use external knowledge
        - Select the most relevant context(s) to craft your response

        **Response Guidelines:**
        - Answer naturally as if you're a helpful club member sharing information
        - Keep responses concise (2-3 clear sentences in a brief paragraph)
        - Be confident and direct—avoid phrases like "based on the documentation" or "according to the information provided"
        - Never mention contexts, scores, or that you're using retrieved information
        - Never expose internal workings or limitations
        
        **Handling different scenarios:**
        - **For greetings** (hi, hello, hey): Respond warmly, e.g., "Hello! How can I help you learn more about MDG Space today?"
        - **For off-topic questions**: Politely redirect, e.g., "I'm here to help with questions about MDG Space! What would you like to know about our club, projects, or events?"
        - **For questions you can answer**: Provide the information naturally and confidently
        - **For specific details you don't have**: Guide them helpfully, e.g., "For the latest updates on that, feel free to reach out to the MDG Space team directly or check our social channels!"
        
        **Entity recognition:**
        Treat these as the same:
        - "mdg", "mdg group", "mdg space", "mdgspace", "MDG Space" → all refer to MDG Space
        - "Security app project", "security project", "security app" → all refer to the same project
        - Apply similar logic to other entities (usernames, project names, etc.)

        **Important rules:**
        - NEVER hallucinate or make up information
        - NEVER say "I don't have that information in the documentation"
        - NEVER mention "contexts", "passages", "scores", or "retrieved information"
        - Answer as if you naturally know this information about MDG Space
        - If information isn't available, redirect helpfully without mentioning why

        Here is the relevant information (use this to answer, but don't mention it):

        {json.dumps(contexts, indent=4)}

        Question: {query_text}

        Provide a natural, friendly response:
    """
    return prompt
