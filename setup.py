#!/usr/bin/env python3
"""
Setup script for Multimodal E-commerce Conversational AI
This script helps initialize the system and process the Amazon dataset with CLIP embeddings.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if environment is properly set up"""
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=sk-your_api_key_here")
        return False
        
    print("✅ Environment variables configured")
    return True

def check_gpu_availability():
    """Check if GPU is available for CLIP processing"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
            return True
        else:
            print("⚠️ No GPU detected. Processing will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    print("This may take a while as it includes PyTorch and transformers...")
    os.system("pip install -r requirements.txt")
    print("✅ Dependencies installed")

async def process_dataset(max_products=None):
    """Process the Amazon dataset with CLIP embeddings"""
    if max_products:
        print(f"🔄 Processing Amazon dataset (max {max_products} products)...")
    else:
        print("🔄 Processing FULL Amazon dataset (this may take several hours)...")
        print("💡 Tip: Use --max-products 1000 for faster testing")
    
    try:
        from data_processor import AmazonDataProcessor
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        processor = AmazonDataProcessor(openai_api_key)
        
        # Process dataset
        await processor.process_dataset(max_products=max_products)
        print("✅ Dataset processing completed!")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return False
        
    return True

def check_data_status():
    """Check if processed data exists"""
    data_dir = Path("data")
    processed_file = data_dir / "processed_products.csv"
    clip_embeddings_file = data_dir / "embeddings" / "clip_embeddings.json"
    numpy_embeddings = data_dir / "embeddings" / "multimodal_embeddings.npy"
    
    if processed_file.exists() and (clip_embeddings_file.exists() or numpy_embeddings.exists()):
        print("✅ Processed data with CLIP embeddings found")
        
        # Show data statistics
        try:
            import pandas as pd
            df = pd.read_csv(processed_file)
            print(f"📊 Dataset contains {len(df)} products with valid images")
        except:
            pass
            
        return True
    else:
        print("⚠️ Processed data not found")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup Multimodal E-commerce AI with CLIP")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--process-data", action="store_true", help="Process dataset")
    parser.add_argument("--max-products", type=int, default=None, help="Maximum products to process (default: all)")
    parser.add_argument("--check", action="store_true", help="Check system status")
    parser.add_argument("--full-setup", action="store_true", help="Run full setup")
    parser.add_argument("--test-setup", action="store_true", help="Setup with limited data for testing")
    
    args = parser.parse_args()
    
    print("🛍️ Multimodal E-commerce Conversational AI Setup (CLIP-based)")
    print("=" * 60)
    
    if args.test_setup:
        print("\n🧪 Running test setup with limited data...")
        args.full_setup = True
        args.max_products = 500
    
    if args.check or args.full_setup:
        print("\n📋 Checking system status...")
        env_ok = check_environment()
        gpu_available = check_gpu_availability()
        data_ok = check_data_status()
        
        if env_ok and data_ok:
            print("✅ System is ready!")
            if not args.full_setup:
                return
        elif not env_ok:
            print("❌ Environment setup required")
            if not args.full_setup:
                return
    
    if args.install or args.full_setup:
        install_dependencies()
        # Check GPU again after installation
        check_gpu_availability()
    
    if args.process_data or args.full_setup:
        if not check_environment():
            print("❌ Cannot process data without proper environment setup")
            return
            
        max_products = args.max_products
        if args.full_setup and max_products is None:
            # Ask user for confirmation for full dataset
            print("\n⚠️ WARNING: Processing the full dataset may take 4-8 hours and use significant API credits.")
            print("💰 Estimated cost: $50-100 in OpenAI API calls for ~10,000 products")
            print("🔧 For testing, consider using --test-setup instead")
            
            response = input("\nDo you want to continue with full dataset? (y/N): ").lower()
            if response != 'y':
                print("Setup cancelled. Use --test-setup for a smaller dataset.")
                return
            
        asyncio.run(process_dataset(max_products))
    
    if args.full_setup or args.test_setup:
        print("\n🎉 Setup completed!")
        print("\nNext steps:")
        print("1. Test the system: python test_system.py")
        print("2. Run the application: chainlit run app/main.py -w --port 8005")
        print("3. Open your browser to http://localhost:8005")
        print("4. Start chatting with your AI shopping assistant!")
        print("\n💡 Features:")
        print("- Upload images for visual product search")
        print("- Ask questions in natural language")
        print("- Get multimodal product recommendations")

if __name__ == "__main__":
    main() 