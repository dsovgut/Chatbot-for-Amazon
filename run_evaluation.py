#!/usr/bin/env python3
"""
Standalone Real-World Evaluation Script for Multimodal E-commerce RAG System

This script runs comprehensive real-world evaluation metrics without requiring ground truth data.
It evaluates embedding quality, retrieval diversity, response consistency, user behavior simulation,
cross-modal alignment, and system performance using practical metrics.
"""

import os
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive real-world evaluation of the multimodal RAG system")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory to save results")
    parser.add_argument("--save-report", action="store_true", help="Save evaluation report to file")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (fewer queries)")
    parser.add_argument("--detailed", action="store_true", help="Run detailed evaluation with all metrics")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found. Please set it in .env file.")
        return
    
    print("🔍 Real-World Multimodal E-commerce RAG System Evaluation")
    print("=" * 60)
    
    asyncio.run(run_evaluation(args, openai_api_key))

async def run_evaluation(args, openai_api_key):
    """Run the real-world evaluation process"""
    try:
        # Import systems
        from multimodal_rag import MultimodalEcommerceRAG
        from evaluation_metrics import RealWorldRAGEvaluator
        
        print("🔄 Initializing systems...")
        
        # Initialize RAG system
        rag_system = MultimodalEcommerceRAG(openai_api_key)
        
        if rag_system.products_df is None:
            print("❌ No processed data found. Please run setup first:")
            print("python setup.py --test-setup")
            return
        
        print(f"✅ Loaded {len(rag_system.products_df)} products")
        
        # Initialize evaluator
        evaluator = RealWorldRAGEvaluator(rag_system, openai_api_key, args.output_dir)
        
        # Adjust evaluation scope based on arguments
        if args.quick:
            print("⚡ Running quick real-world evaluation...")
            # Limit to fewer queries for quick evaluation
            evaluator.test_queries = evaluator.test_queries[:5]
        elif args.detailed:
            print("🔬 Running detailed real-world evaluation...")
            # Use all generated queries for detailed evaluation
            pass
        else:
            print("📊 Running standard real-world evaluation...")
            # Use moderate number of queries
            evaluator.test_queries = evaluator.test_queries[:10]
        
        print(f"📋 Evaluating with {len(evaluator.test_queries)} realistic test queries")
        print("💡 Note: Queries are automatically generated from your actual product data")
        
        # Run comprehensive evaluation
        print("\n🚀 Starting real-world evaluation process...")
        results = await evaluator.run_comprehensive_evaluation()
        
        if not results:
            print("❌ Real-world evaluation failed")
            return
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 REAL-WORLD EVALUATION RESULTS")
        print("=" * 60)
        
        # Overall score
        overall_score = results.get('overall_score', 0)
        print(f"\n🎯 OVERALL PERFORMANCE SCORE: {overall_score:.2f}/10")
        
        # Performance grade
        if overall_score >= 8.5:
            grade = "🏆 EXCELLENT"
        elif overall_score >= 7.0:
            grade = "🥇 VERY GOOD"
        elif overall_score >= 5.5:
            grade = "🥈 GOOD"
        elif overall_score >= 4.0:
            grade = "🥉 FAIR"
        else:
            grade = "⚠️ NEEDS IMPROVEMENT"
        
        print(f"Grade: {grade}")
        
        # Detailed metrics
        print("\n📈 DETAILED METRICS:")
        print("-" * 30)
        
        # 1. Embedding Quality
        if 'embedding_quality' in results:
            eq = results['embedding_quality']
            print("\n🧠 Embedding Quality Assessment:")
            print(f"  • Overall Quality: {eq.get('overall_embedding_quality', 0):.3f}")
            print(f"  • Intra-cluster Coherence: {eq.get('intra_cluster_coherence', 0):.3f}")
            print(f"  • Inter-cluster Separation: {eq.get('inter_cluster_separation', 0):.3f}")
            print(f"  • Effective Dimensionality: {eq.get('effective_dimensionality', 0):.3f}")
        
        # 2. Retrieval Analysis
        if 'retrieval_analysis' in results:
            ra = results['retrieval_analysis']
            print("\n🔍 Retrieval Diversity & Coverage:")
            print(f"  • Average Diversity Score: {ra.get('avg_diversity_score', 0):.3f}")
            print(f"  • Average Coverage Score: {ra.get('avg_coverage_score', 0):.3f}")
            print(f"  • Result Consistency: {ra.get('result_consistency', 0):.3f}")
            print(f"  • Queries Evaluated: {ra.get('queries_evaluated', 0)}")
        
        # 3. Response Consistency
        if 'response_consistency' in results:
            rc = results['response_consistency']
            print("\n💬 Response Consistency:")
            print(f"  • Response Length Consistency: {rc.get('response_length_consistency', 0):.3f}")
            print(f"  • Product Mention Rate: {rc.get('product_mention_rate', 0):.3f}")
            print(f"  • Average Coherence Score: {rc.get('avg_coherence_score', 0):.3f}")
            print(f"  • Average Response Length: {rc.get('avg_response_length', 0):.1f} words")
        
        # 4. User Behavior Simulation
        if 'user_behavior' in results:
            ub = results['user_behavior']
            print("\n👤 User Behavior Simulation:")
            print(f"  • Predicted Click-Through Rate: {ub.get('predicted_ctr', 0):.3f}")
            print(f"  • Average Engagement Score: {ub.get('avg_engagement_score', 0):.3f}")
            print(f"  • Estimated User Satisfaction: {ub.get('estimated_satisfaction', 0):.3f}")
            print(f"  • Queries Simulated: {ub.get('queries_simulated', 0)}")
        
        # 5. Cross-Modal Quality
        if 'cross_modal_quality' in results:
            cmq = results['cross_modal_quality']
            print("\n🖼️ Cross-Modal Alignment:")
            print(f"  • Average Alignment Score: {cmq.get('avg_alignment_score', 0):.3f}")
            print(f"  • Cross-Modal Consistency: {cmq.get('cross_modal_consistency', 0):.3f}")
            print(f"  • Products Analyzed: {cmq.get('products_analyzed', 0)}")
        
        # 6. System Performance
        if 'system_performance' in results:
            sp = results['system_performance']
            print("\n⚡ System Performance:")
            print(f"  • Average Response Time: {sp.get('avg_response_time', 0):.3f}s")
            print(f"  • 95th Percentile Response Time: {sp.get('response_time_p95', 0):.3f}s")
            print(f"  • Throughput: {sp.get('throughput_qps', 0):.2f} queries/second")
            print(f"  • Memory Efficiency: {sp.get('memory_efficiency', 0):.3f}")
        
        # Generate and display recommendations
        print("\n💡 ACTIONABLE INSIGHTS:")
        print("-" * 25)
        
        recommendations = generate_recommendations(results)
        for rec in recommendations:
            print(f"  • {rec}")
        
        # Save report if requested
        if args.save_report:
            report = evaluator.generate_evaluation_report(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path(args.output_dir) / f"realworld_evaluation_report_{timestamp}.md"
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Display file locations
        print(f"\n📁 Results saved to: {args.output_dir}/")
        print("   • realworld_evaluation_*.json - Summary metrics")
        print("   • realworld_detailed_*.json - Detailed results")
        if args.save_report:
            print("   • realworld_evaluation_report_*.md - Human-readable report")
        
        print("\n✅ Real-world evaluation completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during real-world evaluation: {e}")

def generate_recommendations(results):
    """Generate actionable recommendations based on real-world evaluation results"""
    recommendations = []
    
    overall_score = results.get('overall_score', 0)
    
    # Overall performance recommendations
    if overall_score < 6.0:
        recommendations.append("System needs significant improvement across multiple areas")
        recommendations.append("Consider reprocessing dataset with better quality control")
    elif overall_score < 7.5:
        recommendations.append("Good performance with room for optimization")
    else:
        recommendations.append("Excellent performance - focus on maintaining quality")
    
    # Embedding quality recommendations
    if 'embedding_quality' in results:
        eq_score = results['embedding_quality'].get('overall_embedding_quality', 0)
        if eq_score < 0.6:
            recommendations.append("Improve embedding quality through better CLIP preprocessing")
            recommendations.append("Consider using higher resolution images for better embeddings")
        
        coherence = results['embedding_quality'].get('intra_cluster_coherence', 0)
        if coherence < 0.7:
            recommendations.append("Enhance intra-cluster coherence through better product categorization")
    
    # Retrieval diversity recommendations
    if 'retrieval_analysis' in results:
        diversity = results['retrieval_analysis'].get('avg_diversity_score', 0)
        if diversity < 0.4:
            recommendations.append("Increase retrieval diversity to provide more varied results")
            recommendations.append("Implement diversity-aware ranking algorithms")
        
        consistency = results['retrieval_analysis'].get('result_consistency', 0)
        if consistency < 0.8:
            recommendations.append("Improve result consistency through deterministic ranking")
    
    # Response consistency recommendations
    if 'response_consistency' in results:
        mention_rate = results['response_consistency'].get('product_mention_rate', 0)
        if mention_rate < 0.6:
            recommendations.append("Improve product mention rate in responses")
            recommendations.append("Enhance LLM prompting to reference retrieved products")
    
    # User behavior recommendations
    if 'user_behavior' in results:
        satisfaction = results['user_behavior'].get('estimated_satisfaction', 0)
        if satisfaction < 0.6:
            recommendations.append("Focus on improving user satisfaction through better results")
            recommendations.append("Optimize ranking to surface most relevant products first")
        
        ctr = results['user_behavior'].get('predicted_ctr', 0)
        if ctr < 0.3:
            recommendations.append("Improve predicted click-through rates through better relevance")
    
    # Cross-modal recommendations
    if 'cross_modal_quality' in results:
        alignment = results['cross_modal_quality'].get('avg_alignment_score', 0)
        if alignment < 0.5:
            recommendations.append("Enhance cross-modal alignment through better image descriptions")
            recommendations.append("Consider fine-tuning CLIP on domain-specific data")
    
    # Performance recommendations
    if 'system_performance' in results:
        response_time = results['system_performance'].get('avg_response_time', 0)
        if response_time > 2.0:
            recommendations.append("Optimize response time through caching and model optimization")
            recommendations.append("Consider using GPU acceleration for faster processing")
        
        memory_efficiency = results['system_performance'].get('memory_efficiency', 0)
        if memory_efficiency < 0.5:
            recommendations.append("Improve memory efficiency through embedding compression")
    
    # Default recommendations if none specific
    if not recommendations:
        recommendations.append("System shows strong performance across all metrics")
        recommendations.append("Continue monitoring and maintain current quality standards")
        recommendations.append("Consider A/B testing for incremental improvements")
    
    return recommendations

if __name__ == "__main__":
    main() 