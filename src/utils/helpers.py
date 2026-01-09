"""
Helper functions for the RAG system.
"""
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG system for answering questions")
    parser.add_argument("-q", "--query_text", type=str, help="The query text", default=None)
    args = parser.parse_args()
    
    return args.query_text


def format_contexts(contexts):
    """
    Format context results for display.
    
    Args:
        contexts (list): List of context dictionaries
        
    Returns:
        str: Formatted string representation of contexts
    """
    if not contexts:
        return "No contexts found"
    
    formatted = []
    for i, ctx in enumerate(contexts, 1):
        content = ctx.get("context", "")
        score = ctx.get("relevance_score", "Unknown")
        
        formatted.append(f"Context {i} (Score: {score}):\n{content}\n")
    
    return "\n".join(formatted)


def truncate_string(text, max_length=100):
    """
    Truncate string with ellipsis if it exceeds max_length.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length before truncation
        
    Returns:
        str: Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."