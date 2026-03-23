"""
Embedding models for text representation.
"""

from sentence_transformers import SentenceTransformer
from config import DEFAULT_EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)


class StellaEmbeddings:
    """
    Text embedding model using Sentence Transformers.
    
    This class provides methods to encode documents and queries into embedding vectors.
    """
    
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL, **kwargs):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): The name of the model to use
            **kwargs: Additional arguments for the model initialization
        """
        self.model_name = model_name
        try:
            self.client = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise

    def embed_documents(self, texts):
        """
        Embed a list of document texts.
        
        Args:
            texts (list): List of strings to embed
            
        Returns:
            list: List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            return self.client.encode(texts, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            return []

    def embed_query(self, text):
        """
        Embed a single query text.
        
        Args:
            text (str): Query string to embed
            
        Returns:
            list: Embedding vector
        """
        if not text:
            return []
            
        try:
            return self.client.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            return []
