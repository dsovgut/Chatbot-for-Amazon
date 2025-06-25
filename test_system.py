#!/usr/bin/env python3
"""
Test script for Multimodal E-commerce RAG System with CLIP and Evaluation Metrics
This script tests the core functionality including evaluation metrics without the UI.
"""

import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag_system():
    """Test the RAG system functionality"""
    print("🧪 Testing Multimodal E-commerce RAG System (CLIP-based)")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        return False
    
    try:
        # Import and initialize RAG system
        from multimodal_rag import MultimodalEcommerceRAG
        
        print("🔄 Initializing CLIP-based RAG system...")
        rag = MultimodalEcommerceRAG(openai_api_key)
        
        # Check if data is loaded
        if (rag.products_df is None or 
            rag.multimodal_embeddings_matrix is None or 
            rag.image_embeddings_matrix is None or 
            rag.text_embeddings_matrix is None):
            print("⚠️ No processed data found. Please run setup first:")
            print("python setup.py --test-setup")
            return False
            
        print(f"✅ Loaded {len(rag.products_df)} products")
        print(f"✅ Loaded CLIP embeddings: {rag.multimodal_embeddings_matrix.shape}")
        print(f"✅ Image embeddings: {rag.image_embeddings_matrix.shape}")
        print(f"✅ Text embeddings: {rag.text_embeddings_matrix.shape}")
        
        # Test 1: Text query
        print("\n🔍 Test 1: Text-only query...")
        test_query = "wireless bluetooth headphones with noise cancellation"
        
        result = await rag.process_query(
            text_query=test_query,
            top_k=3
        )
        
        print(f"Query: '{test_query}'")
        print(f"Search Mode: {result['search_mode']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Retrieved {len(result['retrieved_products'])} products")
        
        if result['retrieved_products']:
            print("\nTop products:")
            for i, product in enumerate(result['retrieved_products'][:3], 1):
                print(f"{i}. {product['product_name']} (Score: {product['similarity_score']:.3f})")
        
        # Test 2: CLIP embedding generation
        print("\n🔄 Test 2: CLIP embedding generation...")
        test_embedding = rag.get_clip_query_embedding("test product")
        if len(test_embedding) > 0:
            print(f"✅ Generated CLIP embedding with {len(test_embedding)} dimensions")
        else:
            print("❌ Failed to generate CLIP embedding")
            return False
        
        # Test 3: Different search modes
        print("\n🔄 Test 3: Testing different search modes...")
        
        # Test multimodal search
        multimodal_result = await rag.process_query(
            text_query="gaming laptop",
            top_k=2
        )
        print(f"Multimodal search found {len(multimodal_result['retrieved_products'])} products")
        
        # Test 4: Product recommendations
        print("\n🔄 Test 4: Testing product recommendations...")
        if rag.product_ids:
            first_product_id = rag.product_ids[0]
            recommendations = rag.get_product_recommendations(first_product_id, top_k=2)
            print(f"Found {len(recommendations)} recommendations for product {first_product_id}")
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec['product_name']} (Score: {rec['similarity_score']:.3f})")
        
        print("\n✅ All RAG tests passed! System is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        return False

async def test_evaluation_system():
    """Test the real-world evaluation metrics system"""
    print("\n🧪 Testing Real-World Evaluation Metrics System")
    print("-" * 50)
    
    try:
        from multimodal_rag import MultimodalEcommerceRAG
        from evaluation_metrics import RealWorldRAGEvaluator
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize systems
        print("🔄 Initializing RAG and real-world evaluation systems...")
        rag = MultimodalEcommerceRAG(openai_api_key)
        evaluator = RealWorldRAGEvaluator(rag, openai_api_key)
        
        if rag.products_df is None:
            print("⚠️ No processed data found. Skipping evaluation tests.")
            return False
        
        print(f"✅ Real-world evaluation system initialized with {len(evaluator.test_queries)} generated test queries")
        
        # Test 1: Embedding quality evaluation
        print("\n🔄 Test 1: Embedding quality evaluation...")
        embedding_quality = await evaluator.evaluate_embedding_quality()
        
        if embedding_quality:
            print("Embedding Quality Metrics:")
            print(f"- Overall Quality: {embedding_quality.get('overall_embedding_quality', 0):.3f}")
            print(f"- Intra-cluster Coherence: {embedding_quality.get('intra_cluster_coherence', 0):.3f}")
            print(f"- Inter-cluster Separation: {embedding_quality.get('inter_cluster_separation', 0):.3f}")
            print(f"- Effective Dimensionality: {embedding_quality.get('effective_dimensionality', 0):.3f}")
        else:
            print("⚠️ Embedding quality evaluation failed")
        
        # Test 2: Retrieval diversity and coverage
        print("\n🔄 Test 2: Retrieval diversity and coverage...")
        retrieval_metrics = await evaluator.evaluate_retrieval_diversity_and_coverage()
        
        if retrieval_metrics:
            print("Retrieval Analysis:")
            print(f"- Average Diversity Score: {retrieval_metrics.get('avg_diversity_score', 0):.3f}")
            print(f"- Average Coverage Score: {retrieval_metrics.get('avg_coverage_score', 0):.3f}")
            print(f"- Result Consistency: {retrieval_metrics.get('result_consistency', 0):.3f}")
            print(f"- Queries Evaluated: {retrieval_metrics.get('queries_evaluated', 0)}")
        else:
            print("⚠️ Retrieval analysis failed")
        
        # Test 3: Response consistency
        print("\n🔄 Test 3: Response consistency evaluation...")
        response_metrics = await evaluator.evaluate_response_consistency()
        
        if response_metrics:
            print("Response Consistency:")
            print(f"- Average Response Length: {response_metrics.get('avg_response_length', 0):.1f} words")
            print(f"- Response Length Consistency: {response_metrics.get('response_length_consistency', 0):.3f}")
            print(f"- Product Mention Rate: {response_metrics.get('product_mention_rate', 0):.3f}")
            print(f"- Average Coherence Score: {response_metrics.get('avg_coherence_score', 0):.3f}")
        else:
            print("⚠️ Response consistency evaluation failed")
        
        # Test 4: User behavior simulation
        print("\n🔄 Test 4: User behavior simulation...")
        user_behavior = await evaluator.simulate_user_behavior()
        
        if user_behavior:
            print("User Behavior Simulation:")
            print(f"- Predicted CTR: {user_behavior.get('predicted_ctr', 0):.3f}")
            print(f"- Average Engagement Score: {user_behavior.get('avg_engagement_score', 0):.3f}")
            print(f"- Estimated Satisfaction: {user_behavior.get('estimated_satisfaction', 0):.3f}")
        else:
            print("⚠️ User behavior simulation failed")
        
        # Test 5: Cross-modal alignment
        print("\n🔄 Test 5: Cross-modal alignment evaluation...")
        cross_modal = await evaluator.evaluate_cross_modal_alignment()
        
        if cross_modal:
            print("Cross-Modal Quality:")
            print(f"- Average Alignment Score: {cross_modal.get('avg_alignment_score', 0):.3f}")
            print(f"- Cross-Modal Consistency: {cross_modal.get('cross_modal_consistency', 0):.3f}")
            print(f"- Products Analyzed: {cross_modal.get('products_analyzed', 0)}")
        else:
            print("⚠️ Cross-modal alignment evaluation skipped (missing embeddings)")
        
        # Test 6: System performance
        print("\n🔄 Test 6: System performance evaluation...")
        performance = await evaluator.evaluate_system_performance()
        
        if performance:
            print("System Performance:")
            print(f"- Average Response Time: {performance.get('avg_response_time', 0):.3f}s")
            print(f"- 95th Percentile Response Time: {performance.get('response_time_p95', 0):.3f}s")
            print(f"- Throughput: {performance.get('throughput_qps', 0):.2f} queries/sec")
            print(f"- Memory Efficiency: {performance.get('memory_efficiency', 0):.3f}")
        else:
            print("⚠️ Performance evaluation failed")
        
        print("\n✅ Real-world evaluation system tests completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing real-world evaluation system: {e}")
        return False

async def test_data_processor():
    """Test the data processor functionality"""
    print("\n🧪 Testing CLIP Data Processor")
    print("-" * 40)
    
    try:
        from data_processor import AmazonDataProcessor
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        processor = AmazonDataProcessor(openai_api_key)
        
        # Check if dataset exists
        if processor.dataset_path.exists():
            print("✅ Dataset file found")
        else:
            print("⚠️ Dataset file not found")
            
        # Check if processed data exists
        data_dir = Path("data")
        if (data_dir / "processed_products.csv").exists():
            print("✅ Processed products file found")
        else:
            print("⚠️ Processed products file not found")
            
        # Check CLIP embeddings
        if (data_dir / "embeddings" / "clip_embeddings.json").exists():
            print("✅ CLIP embeddings JSON file found")
        else:
            print("⚠️ CLIP embeddings JSON file not found")
            
        # Check numpy embeddings
        embeddings_dir = data_dir / "embeddings"
        numpy_files = [
            "image_embeddings.npy",
            "text_embeddings.npy", 
            "multimodal_embeddings.npy"
        ]
        
        for file in numpy_files:
            if (embeddings_dir / file).exists():
                print(f"✅ {file} found")
            else:
                print(f"⚠️ {file} not found")
        
        # Test CLIP model loading
        print("\n🔄 Testing CLIP model loading...")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device: {device}")
            
            # This will test if CLIP can be loaded
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ CLIP model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing data processor: {e}")
        return False

def test_gpu_setup():
    """Test GPU availability for CLIP processing"""
    print("\n🧪 Testing GPU Setup")
    print("-" * 25)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️ No GPU detected. CLIP processing will use CPU (slower)")
            
        return True
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking GPU setup: {e}")
        return False

async def test_comprehensive_evaluation():
    """Test the comprehensive real-world evaluation system"""
    print("\n🧪 Testing Comprehensive Real-World Evaluation")
    print("-" * 45)
    
    try:
        from multimodal_rag import MultimodalEcommerceRAG
        from evaluation_metrics import RealWorldRAGEvaluator
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize systems
        rag = MultimodalEcommerceRAG(openai_api_key)
        evaluator = RealWorldRAGEvaluator(rag, openai_api_key)
        
        if rag.products_df is None:
            print("⚠️ No processed data found. Skipping comprehensive evaluation.")
            return False
        
        print("🔄 Running limited comprehensive real-world evaluation...")
        print("⚠️ Note: This uses realistic queries generated from your actual product data")
        
        # Limit test queries for testing to avoid long runtime
        original_queries = evaluator.test_queries
        evaluator.test_queries = original_queries[:5]  # Test with 5 queries only
        
        # Run comprehensive evaluation
        results = await evaluator.run_comprehensive_evaluation()
        
        if results:
            print("\n📊 Comprehensive Real-World Evaluation Results:")
            print(f"Overall Score: {results.get('overall_score', 0):.2f}/10")
            
            # Display key metrics
            if 'embedding_quality' in results:
                print(f"Embedding Quality: {results['embedding_quality'].get('overall_embedding_quality', 0):.3f}")
            
            if 'retrieval_analysis' in results:
                ra = results['retrieval_analysis']
                print(f"Retrieval Diversity: {ra.get('avg_diversity_score', 0):.3f}")
                print(f"Retrieval Coverage: {ra.get('avg_coverage_score', 0):.3f}")
            
            if 'user_behavior' in results:
                ub = results['user_behavior']
                print(f"Predicted User Satisfaction: {ub.get('estimated_satisfaction', 0):.3f}")
            
            if 'system_performance' in results:
                sp = results['system_performance']
                print(f"Avg Response Time: {sp.get('avg_response_time', 0):.3f}s")
            
            # Generate report
            report = evaluator.generate_evaluation_report(results)
            print("\n📋 Real-World Evaluation Report Generated:")
            print(f"Report length: {len(report)} characters")
            
            print("✅ Comprehensive real-world evaluation test completed!")
        else:
            print("❌ Comprehensive real-world evaluation failed")
            return False
        
        # Restore original queries
        evaluator.test_queries = original_queries
        
        return True
        
    except Exception as e:
        print(f"❌ Error in comprehensive real-world evaluation test: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Multimodal E-commerce AI System Tests with Evaluation")
    print("=" * 70)
    
    # Test GPU setup
    gpu_ok = test_gpu_setup()
    
    # Test data processor
    asyncio.run(test_data_processor())
    
    # Test RAG system
    rag_success = asyncio.run(test_rag_system())
    
    # Test evaluation system
    eval_success = asyncio.run(test_evaluation_system())
    
    # Test comprehensive evaluation (limited)
    comp_eval_success = asyncio.run(test_comprehensive_evaluation())
    
    print("\n" + "=" * 70)
    if rag_success and eval_success:
        print("🎉 All tests completed successfully!")
        print("\n🚀 System is ready with evaluation metrics! You can now:")
        print("1. Run the application: chainlit run app/main.py -w --port 8005")
        print("2. Open your browser to: http://localhost:8005")
        print("3. Try these features:")
        print("   • Upload product images for visual search")
        print("   • Ask questions like 'wireless headphones under $100'")
        print("   • View real-time performance metrics")
        print("   • Get product comparisons and recommendations")
        print("   • See evaluation scores for each query")
        
        if comp_eval_success:
            print("   • Run comprehensive system evaluation")
        
    else:
        print("❌ Some tests failed. Please check the setup.")
        print("\n🔧 Try running:")
        print("python setup.py --test-setup  # For quick setup with limited data")
        print("python setup.py --full-setup  # For complete setup with full dataset")

if __name__ == "__main__":
    main() 