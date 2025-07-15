#!/usr/bin/env python3
"""
Setup script for GDesigner embedding system.
This script helps configure the embedding system and test connectivity.
"""

import sys
import os
import requests
from pathlib import Path

# Add the GDesigner directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GDesigner'))

def test_network_connectivity():
    """Test network connectivity to required services."""
    print("=== Testing Network Connectivity ===\n")
    
    test_urls = [
        "https://huggingface.co",
        "https://api.openai.com",
        "https://www.google.com"  # General connectivity test
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"✓ {url} - Status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"✗ {url} - Connection failed (DNS/Network issue)")
        except requests.exceptions.Timeout:
            print(f"✗ {url} - Timeout")
        except Exception as e:
            print(f"✗ {url} - Error: {e}")
    
    print()

def setup_cache_directory():
    """Set up cache directory for models."""
    print("=== Setting Up Cache Directory ===\n")
    
    from GDesigner.utils.embedding_config import get_embedding_config, set_cache_directory
    
    # Get current config
    config = get_embedding_config()
    current_cache = config.cache_dir
    
    print(f"Current cache directory: {current_cache}")
    
    # Check if directory exists and is writable
    if current_cache is None:
        print("✗ Cache directory is None")
        return
        
    cache_path = Path(current_cache)
    if cache_path.exists():
        print(f"✓ Cache directory exists: {current_cache}")
        if os.access(current_cache, os.W_OK):
            print("✓ Cache directory is writable")
        else:
            print("✗ Cache directory is not writable")
    else:
        print(f"Creating cache directory: {current_cache}")
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            print("✓ Cache directory created successfully")
        except Exception as e:
            print(f"✗ Failed to create cache directory: {e}")
    
    print()

def test_embedding_system():
    """Test the embedding system with different scenarios."""
    print("=== Testing Embedding System ===\n")
    
    from GDesigner.llm.profile_embedding import get_sentence_embedding
    from GDesigner.utils.embedding_config import enable_offline_mode, disable_offline_mode
    
    test_sentence = "This is a test sentence for the embedding system."
    
    # Test 1: Try with online model first
    print("1. Testing with online model (if available):")
    try:
        embedding = get_sentence_embedding(test_sentence)
        print(f"   ✓ Success! Embedding shape: {embedding.shape}")
        print(f"   ✓ Using SentenceTransformer model")
    except Exception as e:
        print(f"   ✗ Failed to load online model: {e}")
        print("   → Will use fallback method")
    
    print()
    
    # Test 2: Force offline mode
    print("2. Testing offline fallback mode:")
    enable_offline_mode()
    try:
        embedding = get_sentence_embedding(test_sentence)
        print(f"   ✓ Success! Embedding shape: {embedding.shape}")
        print(f"   ✓ Using fallback embedding method")
    except Exception as e:
        print(f"   ✗ Fallback also failed: {e}")
    
    print()

def provide_recommendations():
    """Provide recommendations based on test results."""
    print("=== Recommendations ===\n")
    
    print("If you're experiencing network connectivity issues:")
    print("1. Check your internet connection")
    print("2. Check if you're behind a corporate firewall")
    print("3. Try using a VPN if needed")
    print("4. The system will automatically use fallback embeddings if online models fail")
    print()
    
    print("To improve performance:")
    print("1. The system caches models automatically")
    print("2. You can set a custom cache directory using:")
    print("   from GDesigner.utils.embedding_config import set_cache_directory")
    print("   set_cache_directory('/path/to/your/cache')")
    print()
    
    print("To force offline mode:")
    print("   from GDesigner.utils.embedding_config import enable_offline_mode")
    print("   enable_offline_mode()")
    print()

def main():
    """Main setup function."""
    print("GDesigner Embedding System Setup")
    print("=" * 40)
    print()
    
    test_network_connectivity()
    setup_cache_directory()
    test_embedding_system()
    provide_recommendations()
    
    print("Setup complete! You can now run your GDesigner experiments.")
    print("The system will automatically handle network issues and use fallback methods when needed.")

if __name__ == "__main__":
    main() 