#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

from multimodal_rag import MultimodalEcommerceRAG

async def debug_image_search():
    """Debug image search functionality"""
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Initialize RAG system
    print("🔄 Initializing RAG system...")
    rag = MultimodalEcommerceRAG(openai_api_key)
    
    # Test with a sample image (you can replace this path)
    test_image_path = "temp_uploads"  # Check if there are any uploaded images
    
    # Look for uploaded images
    temp_dir = Path("temp_uploads")
    if temp_dir.exists():
        image_files = list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.png"))
        if image_files:
            test_image_path = str(image_files[0])
            print(f"📷 Found test image: {test_image_path}")
        else:
            print("❌ No test images found in temp_uploads/")
            return
    else:
        print("❌ temp_uploads directory not found")
        return
    
    # Test image analysis
    print("\n🔍 Testing image analysis...")
    image_analysis = rag.analyze_uploaded_image(test_image_path)
    print(f"Image Analysis Result:\n{image_analysis}\n")
    
    # Test CLIP embedding generation
    print("🔄 Testing CLIP embedding generation...")
    query_embedding = rag.get_clip_query_embedding("", test_image_path)
    print(f"Embedding shape: {query_embedding.shape}")
    print(f"Embedding sample: {query_embedding[:5]}")
    
    # Test product retrieval
    print("\n🔍 Testing product retrieval...")
    results = rag.retrieve_similar_products(query_embedding, "image", top_k=5)
    
    print(f"\nFound {len(results)} products:")
    for i, product in enumerate(results, 1):
        print(f"\n[{i}] {product.get('Product Name', 'Unknown')}")
        print(f"    Category: {product.get('Category', 'Unknown')}")
        print(f"    Similarity: {product.get('similarity_score', 0):.3f}")
        print(f"    About: {str(product.get('About Product', ''))[:100]}...")
    
    # Test full query processing
    print("\n🔄 Testing full query processing...")
    result = await rag.process_query(
        text_query="Tell me about this product",
        image_path=test_image_path,
        top_k=3
    )
    
    print(f"\nFull Query Result:")
    print(f"Response: {result.get('response', '')[:200]}...")
    print(f"Query Analysis: {result.get('query_analysis', '')[:200]}...")
    print(f"Search Mode: {result.get('search_mode', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(debug_image_search()) 