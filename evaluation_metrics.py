#!/usr/bin/env python3
"""
Real-World Evaluation Metrics for Multimodal E-commerce RAG System

This module implements practical evaluation metrics that work without ground truth:
1. Embedding Quality Assessment (intra-cluster coherence, inter-cluster separation)
2. Retrieval Diversity and Coverage Analysis
3. Response Consistency and Coherence
4. User Behavior Simulation (click-through prediction, engagement metrics)
5. System Performance and Scalability Metrics
6. Cross-Modal Alignment Quality (without labels)
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
import statistics
from openai import OpenAI
import random
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class RealWorldRAGEvaluator:
    def __init__(self, rag_system, openai_api_key: str, results_dir: str = "evaluation_results"):
        self.rag_system = rag_system
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Evaluation data storage
        self.evaluation_results = {
            'embedding_quality': [],
            'retrieval_analysis': [],
            'response_consistency': [],
            'user_simulation': [],
            'performance_metrics': [],
            'cross_modal_quality': []
        }
        
        # Generate realistic test queries from actual product data
        self.test_queries = self._generate_realistic_queries()
        
    def _generate_realistic_queries(self) -> List[Dict]:
        """Generate realistic queries based on actual product data"""
        if not self.rag_system.products_df is not None:
            return []
            
        queries = []
        
        # Extract common product terms and categories
        product_names = self.rag_system.products_df['Product Name'].dropna().tolist()
        
        # Generate different types of realistic queries
        query_templates = [
            # Feature-based queries
            "wireless {category}",
            "bluetooth {category}",
            "waterproof {category}",
            "portable {category}",
            "rechargeable {category}",
            
            # Price-based queries
            "cheap {category}",
            "affordable {category}",
            "budget {category}",
            "premium {category}",
            
            # Use case queries
            "{category} for gaming",
            "{category} for work",
            "{category} for travel",
            "{category} for home",
            
            # Comparison queries
            "best {category}",
            "top rated {category}",
            "compare {category}",
            
            # Specific feature queries
            "{category} with long battery",
            "{category} with fast charging",
            "{category} with good sound quality"
        ]
        
        # Extract categories from product names
        categories = self._extract_categories_from_products(product_names)
        
        # Generate queries
        for template in query_templates[:15]:  # Limit to avoid too many
            for category in categories[:5]:  # Top 5 categories
                query = template.format(category=category)
                queries.append({
                    'query': query,
                    'type': 'generated',
                    'category': category,
                    'template': template
                })
        
        # Add some direct product name queries (partial matches)
        for product_name in random.sample(product_names, min(10, len(product_names))):
            # Extract key terms from product name
            words = re.findall(r'\b\w+\b', product_name.lower())
            if len(words) >= 2:
                # Create partial queries
                partial_query = ' '.join(words[:2])
                queries.append({
                    'query': partial_query,
                    'type': 'partial_product',
                    'original_product': product_name
                })
        
        return queries[:25]  # Return reasonable number of test queries
    
    def _extract_categories_from_products(self, product_names: List[str]) -> List[str]:
        """Extract common categories from product names"""
        # Common product categories
        category_keywords = [
            'headphones', 'earbuds', 'speaker', 'phone', 'laptop', 'tablet',
            'watch', 'camera', 'charger', 'case', 'keyboard', 'mouse',
            'monitor', 'tv', 'router', 'adapter', 'cable', 'battery',
            'bag', 'backpack', 'wallet', 'shoes', 'shirt', 'dress',
            'jacket', 'pants', 'book', 'toy', 'game', 'tool'
        ]
        
        found_categories = []
        for category in category_keywords:
            # Check if category appears in product names
            count = sum(1 for name in product_names if category in name.lower())
            if count > 0:
                found_categories.append((category, count))
        
        # Sort by frequency and return top categories
        found_categories.sort(key=lambda x: x[1], reverse=True)
        return [cat[0] for cat in found_categories[:10]]
    
    async def evaluate_embedding_quality(self) -> Dict:
        """Evaluate the quality of embeddings without ground truth"""
        logger.info("Evaluating embedding quality...")
        
        if self.rag_system.multimodal_embeddings_matrix is None:
            return {}
        
        embeddings = self.rag_system.multimodal_embeddings_matrix
        
        # 1. Intra-cluster coherence (how similar are nearby embeddings)
        intra_coherence_scores = []
        for i in range(min(100, len(embeddings))):  # Sample for efficiency
            # Find 5 nearest neighbors
            similarities = cosine_similarity([embeddings[i]], embeddings)[0]
            top_5_indices = np.argsort(similarities)[-6:-1]  # Exclude self
            top_5_similarities = similarities[top_5_indices]
            intra_coherence_scores.append(np.mean(top_5_similarities))
        
        avg_intra_coherence = np.mean(intra_coherence_scores)
        
        # 2. Inter-cluster separation (using k-means clustering)
        try:
            # Cluster embeddings
            n_clusters = min(20, len(embeddings) // 10)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette-like score manually
                inter_cluster_distances = []
                for i in range(len(embeddings)):
                    same_cluster = embeddings[cluster_labels == cluster_labels[i]]
                    other_clusters = embeddings[cluster_labels != cluster_labels[i]]
                    
                    if len(same_cluster) > 1 and len(other_clusters) > 0:
                        avg_same = np.mean(cosine_similarity([embeddings[i]], same_cluster)[0])
                        avg_other = np.mean(cosine_similarity([embeddings[i]], other_clusters)[0])
                        inter_cluster_distances.append(avg_same - avg_other)
                
                avg_inter_separation = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
            else:
                avg_inter_separation = 0
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            avg_inter_separation = 0
        
        # 3. Embedding distribution analysis
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        norm_std = np.std(embedding_norms)
        norm_mean = np.mean(embedding_norms)
        
        # 4. Dimensionality utilization (PCA analysis)
        try:
            pca = PCA()
            pca.fit(embeddings[:1000])  # Sample for efficiency
            explained_variance_ratio = pca.explained_variance_ratio_
            # How many dimensions explain 90% of variance
            cumsum = np.cumsum(explained_variance_ratio)
            dims_90_percent = np.argmax(cumsum >= 0.9) + 1
            effective_dimensionality = dims_90_percent / len(explained_variance_ratio)
        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            effective_dimensionality = 0.5
        
        embedding_quality = {
            'intra_cluster_coherence': float(avg_intra_coherence),
            'inter_cluster_separation': float(avg_inter_separation),
            'embedding_norm_consistency': float(1.0 - (norm_std / norm_mean)) if norm_mean > 0 else 0,
            'effective_dimensionality': float(effective_dimensionality),
            'overall_embedding_quality': float((avg_intra_coherence + max(0, avg_inter_separation) + effective_dimensionality) / 3)
        }
        
        self.evaluation_results['embedding_quality'].append({
            'metrics': embedding_quality,
            'timestamp': datetime.now().isoformat()
        })
        
        return embedding_quality
    
    async def evaluate_retrieval_diversity_and_coverage(self) -> Dict:
        """Evaluate retrieval diversity and coverage without ground truth"""
        logger.info("Evaluating retrieval diversity and coverage...")
        
        diversity_scores = []
        coverage_metrics = []
        result_consistency = []
        
        for query_data in self.test_queries:
            try:
                # Get retrieval results
                result = await self.rag_system.process_query(
                    text_query=query_data['query'],
                    top_k=10
                )
                
                retrieved_products = result.get('retrieved_products', [])
                if len(retrieved_products) < 2:
                    continue
                
                # 1. Diversity Analysis - how different are the retrieved products
                product_embeddings = []
                for product in retrieved_products:
                    product_id = product.get('product_id')
                    if product_id in self.rag_system.product_ids:
                        idx = self.rag_system.product_ids.index(product_id)
                        if idx < len(self.rag_system.multimodal_embeddings_matrix):
                            product_embeddings.append(self.rag_system.multimodal_embeddings_matrix[idx])
                
                if len(product_embeddings) >= 2:
                    # Calculate pairwise similarities
                    similarities = cosine_similarity(product_embeddings)
                    # Get upper triangle (excluding diagonal)
                    upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
                    avg_similarity = np.mean(upper_triangle)
                    diversity_score = 1.0 - avg_similarity  # Higher diversity = lower similarity
                    diversity_scores.append(diversity_score)
                
                # 2. Coverage Analysis - how well do results cover different aspects
                product_names = [p.get('product_name', '') for p in retrieved_products]
                unique_words = set()
                for name in product_names:
                    words = re.findall(r'\b\w+\b', name.lower())
                    unique_words.update(words)
                
                # Coverage score based on vocabulary diversity
                total_words = sum(len(re.findall(r'\b\w+\b', name.lower())) for name in product_names)
                coverage_score = len(unique_words) / total_words if total_words > 0 else 0
                coverage_metrics.append(coverage_score)
                
                # 3. Result Consistency - run same query multiple times
                if random.random() < 0.3:  # Test 30% of queries for consistency
                    result2 = await self.rag_system.process_query(
                        text_query=query_data['query'],
                        top_k=10
                    )
                    
                    retrieved_products2 = result2.get('retrieved_products', [])
                    
                    # Calculate overlap in top 5 results
                    top5_ids_1 = [p.get('product_id') for p in retrieved_products[:5]]
                    top5_ids_2 = [p.get('product_id') for p in retrieved_products2[:5]]
                    
                    overlap = len(set(top5_ids_1) & set(top5_ids_2))
                    consistency_score = overlap / 5.0
                    result_consistency.append(consistency_score)
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query_data['query']}': {e}")
        
        retrieval_metrics = {
            'avg_diversity_score': np.mean(diversity_scores) if diversity_scores else 0,
            'avg_coverage_score': np.mean(coverage_metrics) if coverage_metrics else 0,
            'result_consistency': np.mean(result_consistency) if result_consistency else 0,
            'diversity_std': np.std(diversity_scores) if diversity_scores else 0,
            'queries_evaluated': len(diversity_scores)
        }
        
        self.evaluation_results['retrieval_analysis'].append({
            'metrics': retrieval_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return retrieval_metrics
    
    async def evaluate_response_consistency(self) -> Dict:
        """Evaluate response consistency and quality without ground truth"""
        logger.info("Evaluating response consistency...")
        
        response_lengths = []
        response_coherence_scores = []
        product_mention_consistency = []
        
        for query_data in self.test_queries[:10]:  # Limit for API cost
            try:
                # Get response
                result = await self.rag_system.process_query(
                    text_query=query_data['query'],
                    top_k=5
                )
                
                response_text = result.get('response', '')
                retrieved_products = result.get('retrieved_products', [])
                
                if not response_text:
                    continue
                
                # 1. Response length analysis
                word_count = len(response_text.split())
                response_lengths.append(word_count)
                
                # 2. Product mention consistency
                product_names = [p.get('product_name', '') for p in retrieved_products]
                mentioned_products = 0
                for product_name in product_names:
                    # Check if product is mentioned in response
                    product_words = product_name.lower().split()[:3]  # First 3 words
                    if any(word in response_text.lower() for word in product_words if len(word) > 3):
                        mentioned_products += 1
                
                mention_ratio = mentioned_products / len(product_names) if product_names else 0
                product_mention_consistency.append(mention_ratio)
                
                # 3. Response coherence (simple heuristics)
                sentences = response_text.split('.')
                avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
                
                # Coherence based on sentence length consistency and structure
                coherence_score = min(1.0, avg_sentence_length / 20.0)  # Normalize
                response_coherence_scores.append(coherence_score)
                
            except Exception as e:
                logger.error(f"Error evaluating response for '{query_data['query']}': {e}")
        
        consistency_metrics = {
            'avg_response_length': np.mean(response_lengths) if response_lengths else 0,
            'response_length_consistency': 1.0 - (np.std(response_lengths) / np.mean(response_lengths)) if response_lengths and np.mean(response_lengths) > 0 else 0,
            'product_mention_rate': np.mean(product_mention_consistency) if product_mention_consistency else 0,
            'avg_coherence_score': np.mean(response_coherence_scores) if response_coherence_scores else 0,
            'responses_evaluated': len(response_lengths)
        }
        
        self.evaluation_results['response_consistency'].append({
            'metrics': consistency_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return consistency_metrics
    
    async def simulate_user_behavior(self) -> Dict:
        """Simulate realistic user behavior patterns"""
        logger.info("Simulating user behavior...")
        
        click_predictions = []
        engagement_scores = []
        query_satisfaction_estimates = []
        
        for query_data in self.test_queries:
            try:
                result = await self.rag_system.process_query(
                    text_query=query_data['query'],
                    top_k=5
                )
                
                retrieved_products = result.get('retrieved_products', [])
                if not retrieved_products:
                    continue
                
                # 1. Click-through prediction based on similarity scores and position
                ctr_scores = []
                for i, product in enumerate(retrieved_products):
                    similarity = product.get('similarity_score', 0)
                    position_bias = 1.0 / (i + 1)  # Position bias (higher positions get more clicks)
                    
                    # Simple CTR model: similarity * position_bias * random factor
                    ctr_score = similarity * position_bias * (0.8 + 0.4 * random.random())
                    ctr_scores.append(ctr_score)
                
                # Predict if user would click on top result
                top_ctr = ctr_scores[0] if ctr_scores else 0
                click_predictions.append(top_ctr)
                
                # 2. Engagement score based on result quality
                avg_similarity = np.mean([p.get('similarity_score', 0) for p in retrieved_products])
                result_count = len(retrieved_products)
                
                # Engagement increases with quality and sufficient options
                engagement = avg_similarity * min(1.0, result_count / 5.0)
                engagement_scores.append(engagement)
                
                # 3. Query satisfaction estimate
                # Based on: top result quality, diversity, and result count
                top_similarity = retrieved_products[0].get('similarity_score', 0) if retrieved_products else 0
                
                # Calculate diversity of top 3 results
                if len(retrieved_products) >= 3:
                    top3_similarities = [p.get('similarity_score', 0) for p in retrieved_products[:3]]
                    diversity = 1.0 - np.std(top3_similarities)
                else:
                    diversity = 0.5
                
                satisfaction = (top_similarity * 0.5 + diversity * 0.3 + min(1.0, result_count / 5.0) * 0.2)
                query_satisfaction_estimates.append(satisfaction)
                
            except Exception as e:
                logger.error(f"Error simulating user behavior for '{query_data['query']}': {e}")
        
        user_behavior_metrics = {
            'predicted_ctr': np.mean(click_predictions) if click_predictions else 0,
            'avg_engagement_score': np.mean(engagement_scores) if engagement_scores else 0,
            'estimated_satisfaction': np.mean(query_satisfaction_estimates) if query_satisfaction_estimates else 0,
            'ctr_variance': np.var(click_predictions) if click_predictions else 0,
            'queries_simulated': len(click_predictions)
        }
        
        self.evaluation_results['user_simulation'].append({
            'metrics': user_behavior_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return user_behavior_metrics
    
    async def evaluate_cross_modal_alignment(self) -> Dict:
        """Evaluate cross-modal alignment quality without labels"""
        logger.info("Evaluating cross-modal alignment...")
        
        if (self.rag_system.image_embeddings_matrix is None or 
            self.rag_system.text_embeddings_matrix is None):
            return {}
        
        # Sample products for analysis
        sample_size = min(200, len(self.rag_system.product_ids))
        sample_indices = np.random.choice(len(self.rag_system.product_ids), sample_size, replace=False)
        
        alignment_scores = []
        cross_modal_consistency = []
        
        for idx in sample_indices:
            try:
                # Get embeddings for same product
                img_emb = self.rag_system.image_embeddings_matrix[idx]
                txt_emb = self.rag_system.text_embeddings_matrix[idx]
                
                # 1. Direct alignment - similarity between image and text of same product
                direct_similarity = cosine_similarity([img_emb], [txt_emb])[0][0]
                alignment_scores.append(direct_similarity)
                
                # 2. Cross-modal consistency - check if image and text retrieve similar products
                # Find top 10 similar products using image embedding
                img_similarities = cosine_similarity([img_emb], self.rag_system.image_embeddings_matrix)[0]
                img_top10 = set(np.argsort(img_similarities)[-11:-1])  # Exclude self
                
                # Find top 10 similar products using text embedding
                txt_similarities = cosine_similarity([txt_emb], self.rag_system.text_embeddings_matrix)[0]
                txt_top10 = set(np.argsort(txt_similarities)[-11:-1])  # Exclude self
                
                # Calculate overlap
                overlap = len(img_top10 & txt_top10)
                consistency_score = overlap / 10.0
                cross_modal_consistency.append(consistency_score)
                
            except Exception as e:
                logger.error(f"Error in cross-modal evaluation for index {idx}: {e}")
        
        cross_modal_metrics = {
            'avg_alignment_score': np.mean(alignment_scores) if alignment_scores else 0,
            'alignment_consistency': 1.0 - np.std(alignment_scores) if alignment_scores else 0,
            'cross_modal_consistency': np.mean(cross_modal_consistency) if cross_modal_consistency else 0,
            'products_analyzed': len(alignment_scores)
        }
        
        self.evaluation_results['cross_modal_quality'].append({
            'metrics': cross_modal_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return cross_modal_metrics
    
    async def evaluate_system_performance(self) -> Dict:
        """Evaluate system performance metrics"""
        logger.info("Evaluating system performance...")
        
        response_times = []
        embedding_times = []
        memory_efficiency_scores = []
        
        for query_data in self.test_queries[:10]:  # Sample for performance testing
            try:
                # Measure response time
                start_time = time.time()
                
                result = await self.rag_system.process_query(
                    text_query=query_data['query'],
                    top_k=5
                )
                
                total_time = time.time() - start_time
                response_times.append(total_time)
                
                # Measure embedding time
                embed_start = time.time()
                query_embedding = self.rag_system.get_clip_query_embedding(query_data['query'])
                embed_time = time.time() - embed_start
                embedding_times.append(embed_time)
                
            except Exception as e:
                logger.error(f"Error measuring performance for '{query_data['query']}': {e}")
        
        # Memory efficiency (based on embedding matrix size vs number of products)
        if self.rag_system.multimodal_embeddings_matrix is not None:
            matrix_size_mb = self.rag_system.multimodal_embeddings_matrix.nbytes / (1024 * 1024)
            num_products = len(self.rag_system.product_ids)
            memory_per_product = matrix_size_mb / num_products if num_products > 0 else 0
            memory_efficiency = 1.0 / (1.0 + memory_per_product)  # Lower memory per product = higher efficiency
        else:
            memory_efficiency = 0
        
        performance_metrics = {
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'response_time_p95': np.percentile(response_times, 95) if response_times else 0,
            'avg_embedding_time': np.mean(embedding_times) if embedding_times else 0,
            'throughput_qps': 1.0 / np.mean(response_times) if response_times else 0,
            'memory_efficiency': memory_efficiency,
            'response_time_consistency': 1.0 - (np.std(response_times) / np.mean(response_times)) if response_times and np.mean(response_times) > 0 else 0
        }
        
        self.evaluation_results['performance_metrics'].append({
            'metrics': performance_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return performance_metrics
    
    async def run_comprehensive_evaluation(self) -> Dict:
        """Run all evaluation metrics"""
        logger.info("Starting comprehensive real-world evaluation...")
        
        evaluation_results = {}
        
        try:
            # 1. Embedding Quality
            evaluation_results['embedding_quality'] = await self.evaluate_embedding_quality()
            
            # 2. Retrieval Analysis
            evaluation_results['retrieval_analysis'] = await self.evaluate_retrieval_diversity_and_coverage()
            
            # 3. Response Consistency
            evaluation_results['response_consistency'] = await self.evaluate_response_consistency()
            
            # 4. User Behavior Simulation
            evaluation_results['user_behavior'] = await self.simulate_user_behavior()
            
            # 5. Cross-Modal Quality
            evaluation_results['cross_modal_quality'] = await self.evaluate_cross_modal_alignment()
            
            # 6. System Performance
            evaluation_results['system_performance'] = await self.evaluate_system_performance()
            
            # 7. Overall Score
            evaluation_results['overall_score'] = self._calculate_overall_score(evaluation_results)
            
            # Save results
            self._save_evaluation_results(evaluation_results)
            
            logger.info("Comprehensive evaluation completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            return {}
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate weighted overall score"""
        weights = {
            'embedding_quality': 0.2,
            'retrieval_analysis': 0.25,
            'response_consistency': 0.15,
            'user_behavior': 0.2,
            'cross_modal_quality': 0.1,
            'system_performance': 0.1
        }
        
        scores = []
        
        # Embedding quality score
        if 'embedding_quality' in results:
            eq_score = results['embedding_quality'].get('overall_embedding_quality', 0) * 10
            scores.append(eq_score * weights['embedding_quality'])
        
        # Retrieval analysis score
        if 'retrieval_analysis' in results:
            ra = results['retrieval_analysis']
            ra_score = (ra.get('avg_diversity_score', 0) + ra.get('avg_coverage_score', 0) + ra.get('result_consistency', 0)) / 3 * 10
            scores.append(ra_score * weights['retrieval_analysis'])
        
        # Response consistency score
        if 'response_consistency' in results:
            rc = results['response_consistency']
            rc_score = (rc.get('response_length_consistency', 0) + rc.get('product_mention_rate', 0) + rc.get('avg_coherence_score', 0)) / 3 * 10
            scores.append(rc_score * weights['response_consistency'])
        
        # User behavior score
        if 'user_behavior' in results:
            ub = results['user_behavior']
            ub_score = (ub.get('predicted_ctr', 0) + ub.get('avg_engagement_score', 0) + ub.get('estimated_satisfaction', 0)) / 3 * 10
            scores.append(ub_score * weights['user_behavior'])
        
        # Cross-modal quality score
        if 'cross_modal_quality' in results:
            cmq = results['cross_modal_quality']
            cmq_score = (cmq.get('avg_alignment_score', 0) + cmq.get('cross_modal_consistency', 0)) / 2 * 10
            scores.append(cmq_score * weights['cross_modal_quality'])
        
        # System performance score
        if 'system_performance' in results:
            sp = results['system_performance']
            # Performance score based on speed and efficiency
            response_time = sp.get('avg_response_time', 1.0)
            perf_score = min(10.0, 10.0 / max(response_time, 0.1))
            scores.append(perf_score * weights['system_performance'])
        
        return sum(scores) if scores else 0.0
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary results
        summary_file = self.results_dir / f"realworld_evaluation_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed results
        detailed_file = self.results_dir / f"realworld_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Real-world evaluation results saved to {summary_file}")
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate human-readable evaluation report"""
        report = f"""
# Real-World Multimodal RAG System Evaluation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Performance Score: {results.get('overall_score', 0):.2f}/10

## 1. Embedding Quality Assessment
- Overall Embedding Quality: {results.get('embedding_quality', {}).get('overall_embedding_quality', 0):.3f}
- Intra-cluster Coherence: {results.get('embedding_quality', {}).get('intra_cluster_coherence', 0):.3f}
- Inter-cluster Separation: {results.get('embedding_quality', {}).get('inter_cluster_separation', 0):.3f}
- Effective Dimensionality: {results.get('embedding_quality', {}).get('effective_dimensionality', 0):.3f}

## 2. Retrieval Analysis
- Average Diversity Score: {results.get('retrieval_analysis', {}).get('avg_diversity_score', 0):.3f}
- Average Coverage Score: {results.get('retrieval_analysis', {}).get('avg_coverage_score', 0):.3f}
- Result Consistency: {results.get('retrieval_analysis', {}).get('result_consistency', 0):.3f}
- Queries Evaluated: {results.get('retrieval_analysis', {}).get('queries_evaluated', 0)}

## 3. Response Consistency
- Response Length Consistency: {results.get('response_consistency', {}).get('response_length_consistency', 0):.3f}
- Product Mention Rate: {results.get('response_consistency', {}).get('product_mention_rate', 0):.3f}
- Average Coherence Score: {results.get('response_consistency', {}).get('avg_coherence_score', 0):.3f}

## 4. User Behavior Simulation
- Predicted Click-Through Rate: {results.get('user_behavior', {}).get('predicted_ctr', 0):.3f}
- Average Engagement Score: {results.get('user_behavior', {}).get('avg_engagement_score', 0):.3f}
- Estimated Satisfaction: {results.get('user_behavior', {}).get('estimated_satisfaction', 0):.3f}

## 5. Cross-Modal Quality
- Average Alignment Score: {results.get('cross_modal_quality', {}).get('avg_alignment_score', 0):.3f}
- Cross-Modal Consistency: {results.get('cross_modal_quality', {}).get('cross_modal_consistency', 0):.3f}

## 6. System Performance
- Average Response Time: {results.get('system_performance', {}).get('avg_response_time', 0):.3f}s
- 95th Percentile Response Time: {results.get('system_performance', {}).get('response_time_p95', 0):.3f}s
- Throughput: {results.get('system_performance', {}).get('throughput_qps', 0):.2f} queries/second
- Memory Efficiency: {results.get('system_performance', {}).get('memory_efficiency', 0):.3f}

## Key Insights and Recommendations
"""
        
        # Add insights based on results
        overall_score = results.get('overall_score', 0)
        if overall_score >= 7.5:
            report += "- System shows strong performance across all metrics\n"
        elif overall_score >= 6.0:
            report += "- System performance is good with some areas for improvement\n"
        else:
            report += "- System needs significant improvements in multiple areas\n"
        
        # Specific recommendations
        embedding_quality = results.get('embedding_quality', {}).get('overall_embedding_quality', 0)
        if embedding_quality < 0.6:
            report += "- Consider improving embedding quality through better training or preprocessing\n"
        
        diversity = results.get('retrieval_analysis', {}).get('avg_diversity_score', 0)
        if diversity < 0.5:
            report += "- Improve retrieval diversity to provide more varied results\n"
        
        response_time = results.get('system_performance', {}).get('avg_response_time', 0)
        if response_time > 2.0:
            report += "- Optimize system performance to reduce response times\n"
        
        return report

# Backward compatibility - alias for the old class name
MultimodalRAGEvaluator = RealWorldRAGEvaluator

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    from multimodal_rag import MultimodalEcommerceRAG
    
    load_dotenv()
    
    async def run_evaluation():
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        # Initialize RAG system
        rag_system = MultimodalEcommerceRAG(openai_api_key)
        
        # Initialize evaluator
        evaluator = RealWorldRAGEvaluator(rag_system, openai_api_key)
        
        # Run comprehensive evaluation
        results = await evaluator.run_comprehensive_evaluation()
        
        # Generate and print report
        report = evaluator.generate_evaluation_report(results)
        print(report)
        
        return results
    
    asyncio.run(run_evaluation()) 