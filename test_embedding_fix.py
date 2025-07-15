#!/usr/bin/env python3
"""
Test script to verify the embedding solution works in both online and offline modes.
"""

import sys
import os

# Add the GDesigner directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GDesigner'))

from GDesigner.llm.profile_embedding import get_sentence_embedding, clear_model_cache
from GDesigner.utils.embedding_config import get_embedding_config, enable_offline_mode, disable_offline_mode

def test_embedding_functionality():
    """Test the embedding functionality with different scenarios."""
    
    print("=== Testing Embedding Functionality ===\n")
    
    # Test sentences
    test_sentences = [
        "Hello world",
        "This is a test sentence for embedding",
        "Mathematics is the language of the universe",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    # Test 1: Normal mode (with fallback)
    print("1. Testing normal mode with fallback enabled:")
    config = get_embedding_config()
    print(f"   Cache directory: {config.cache_dir}")
    print(f"   Model name: {config.model_name}")
    print(f"   Offline fallback: {config.use_offline_fallback}")
    
    for i, sentence in enumerate(test_sentences):
        try:
            embedding = get_sentence_embedding(sentence)
            print(f"   Sentence {i+1}: {sentence[:30]}... -> Embedding shape: {embedding.shape}")
        except Exception as e:
            print(f"   Sentence {i+1}: Error - {e}")
    
    print()
    
    # Test 2: Offline mode only
    print("2. Testing offline mode only:")
    enable_offline_mode()
    
    for i, sentence in enumerate(test_sentences):
        try:
            embedding = get_sentence_embedding(sentence)
            print(f"   Sentence {i+1}: {sentence[:30]}... -> Embedding shape: {embedding.shape}")
        except Exception as e:
            print(f"   Sentence {i+1}: Error - {e}")
    
    print()
    
    # Test 3: Clear cache and test again
    print("3. Testing after clearing cache:")
    clear_model_cache()
    
    for i, sentence in enumerate(test_sentences):
        try:
            embedding = get_sentence_embedding(sentence)
            print(f"   Sentence {i+1}: {sentence[:30]}... -> Embedding shape: {embedding.shape}")
        except Exception as e:
            print(f"   Sentence {i+1}: Error - {e}")
    
    print("\n=== Test completed ===")

def test_configuration():
    """Test configuration management."""
    print("\n=== Testing Configuration Management ===\n")
    
    # Get current config
    config = get_embedding_config()
    print(f"Current cache directory: {config.cache_dir}")
    print(f"Current model name: {config.model_name}")
    print(f"Offline fallback enabled: {config.use_offline_fallback}")
    
    # Test configuration update
    print("\nUpdating configuration...")
    from GDesigner.utils.embedding_config import update_embedding_config
    update_embedding_config(embedding_dim=512, fallback_embedding_dim=512)
    
    config = get_embedding_config()
    print(f"Updated embedding dimension: {config.embedding_dim}")
    print(f"Updated fallback dimension: {config.fallback_embedding_dim}")

if __name__ == "__main__":
    test_embedding_functionality()
    test_configuration() 