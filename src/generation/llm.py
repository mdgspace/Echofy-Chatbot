"""
LLM interaction for generating responses.
"""

from groq import Groq
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P, GROQ_API_KEY
import logging

logger = logging.getLogger(__name__)

def generate_response(prompt):
    """
    Generate a response from the LLM using the given prompt.
    
    Args:
        prompt (str): The prompt to send to the LLM
        
    Returns:
        str: The generated response
    """
    try:
        # Initialize Groq client (assumes GROQ_API_KEY is in os.environ)
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
        )
        
        response_text = completion.choices[0].message.content
        
        if not response_text:
            logger.warning("Empty response received from LLM")
            
        return response_text
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"