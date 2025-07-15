import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Optional, Union
import logging
from ..utils.embedding_config import get_embedding_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}

def get_cached_model(model_name: Optional[str] = None, cache_dir: Optional[str] = None, use_offline_fallback: bool = True) -> Optional[SentenceTransformer]:
    """
    Get a cached SentenceTransformer model with fallback options.
    
    Args:
        model_name: Name of the model to load
        cache_dir: Directory to cache the model
        use_offline_fallback: Whether to use offline fallback if model loading fails
    
    Returns:
        SentenceTransformer model or None if loading fails
    """
    global _model_cache
    
    if model_name is None:
        config = get_embedding_config()
        model_name = config.model_name
    
    # Check if model is already cached
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    try:
        # Set cache directory if provided
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        logger.info(f"Successfully loaded model: {model_name}")
        return model
        
    except Exception as e:
        logger.warning(f"Failed to load SentenceTransformer model {model_name}: {e}")
        
        if use_offline_fallback:
            logger.info("Using offline fallback embedding method")
            return None
        else:
            raise e

def simple_embedding_fallback(text: str, embedding_dim: int = 384) -> np.ndarray:
    """
    Simple fallback embedding method that doesn't require network access.
    Uses basic text features to create embeddings.
    
    Args:
        text: Input text to embed
        embedding_dim: Dimension of the embedding vector
    
    Returns:
        Embedding vector as numpy array
    """
    # Simple hash-based embedding as fallback
    import hashlib
    
    # Create a hash of the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to numbers and create embedding
    embedding = np.zeros(embedding_dim)
    for i, char in enumerate(text_hash):
        if i >= embedding_dim:
            break
        embedding[i] = ord(char) / 255.0  # Normalize to [0, 1]
    
    # Add some text length and character diversity features
    embedding[0] = len(text) / 1000.0  # Normalize text length
    embedding[1] = len(set(text)) / len(text) if text else 0  # Character diversity
    
    # Fill remaining dimensions with hash-based values
    for i in range(2, embedding_dim):
        if i < len(text_hash):
            embedding[i] = (ord(text_hash[i % len(text_hash)]) + i) % 256 / 255.0
    
    return embedding

def get_sentence_embedding(sentence: str, 
                          model_name: Optional[str] = None, 
                          cache_dir: Optional[str] = None,
                          use_offline_fallback: bool = True,
                          embedding_dim: int = 384) -> np.ndarray:
    """
    Get sentence embedding with robust error handling and offline fallback.
    
    Args:
        sentence: Input sentence to embed
        model_name: Name of the SentenceTransformer model to use
        cache_dir: Directory to cache the model
        use_offline_fallback: Whether to use offline fallback if model loading fails
        embedding_dim: Dimension for fallback embedding
    
    Returns:
        Embedding vector as numpy array
    """
    try:
        model = get_cached_model(model_name, cache_dir, use_offline_fallback)
        
        if model is not None:
            # Use the SentenceTransformer model
            embeddings = model.encode(sentence)
            # Convert to numpy array if it's a tensor
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            return embeddings
        else:
            # Use fallback method
            logger.info("Using fallback embedding method")
            return simple_embedding_fallback(sentence, embedding_dim)
            
    except Exception as e:
        logger.error(f"Error in get_sentence_embedding: {e}")
        
        if use_offline_fallback:
            logger.info("Falling back to simple embedding method due to error")
            return simple_embedding_fallback(sentence, embedding_dim)
        else:
            raise e

def clear_model_cache():
    """Clear the global model cache."""
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")

def set_cache_directory(cache_dir: str):
    """Set the cache directory for Hugging Face models."""
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    logger.info(f"Cache directory set to: {cache_dir}")
