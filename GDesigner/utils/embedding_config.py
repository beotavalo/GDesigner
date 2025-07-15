import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    """Configuration for sentence embedding functionality."""
    
    # Model settings
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dim: int = 384
    
    # Caching settings
    cache_dir: Optional[str] = None
    use_cache: bool = True
    
    # Fallback settings
    use_offline_fallback: bool = True
    fallback_embedding_dim: int = 384
    
    # Network settings
    timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        """Set default cache directory if not provided."""
        if self.cache_dir is None:
            # Use user's home directory for cache
            home_dir = os.path.expanduser("~")
            self.cache_dir = os.path.join(home_dir, ".cache", "gdesigner", "models")
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

# Global configuration instance
embedding_config = EmbeddingConfig()

def get_embedding_config() -> EmbeddingConfig:
    """Get the global embedding configuration."""
    return embedding_config

def update_embedding_config(**kwargs):
    """Update the global embedding configuration."""
    global embedding_config
    
    for key, value in kwargs.items():
        if hasattr(embedding_config, key):
            setattr(embedding_config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    
    # Re-initialize to ensure cache directory is created
    embedding_config.__post_init__()

def set_cache_directory(cache_dir: str):
    """Set the cache directory for embeddings."""
    update_embedding_config(cache_dir=cache_dir)

def enable_offline_mode():
    """Enable offline mode with fallback embeddings."""
    update_embedding_config(use_offline_fallback=True)

def disable_offline_mode():
    """Disable offline mode - will raise errors if model can't be loaded."""
    update_embedding_config(use_offline_fallback=False) 