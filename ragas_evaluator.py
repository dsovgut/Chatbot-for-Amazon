import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance
)
from datasets import Dataset

logger = logging.getLogger(__name__)

class RAGASEvaluator:
    """Real-time RAGAS evaluation system for e-commerce RAG"""
    
    def __init__(self, openai_api_key: str, log_file: str = "evaluation_log.json", use_synthetic_ground_truth: bool = False):
        self.openai_api_key = openai_api_key
        self.log_file = Path(log_file)
        self.use_synthetic_ground_truth = use_synthetic_ground_truth
        
        if use_synthetic_ground_truth:
            # Use all metrics when we have ground truth
            from ragas.metrics import ContextPrecision, ContextRecall
            self.metrics = [
                Faithfulness(),
                AnswerRelevancy(), 
                ContextPrecision(),
                ContextRecall(),
                ContextRelevance()
            ]
        else:
            # Only use metrics that don't require ground truth/reference data
            self.metrics = [
                Faithfulness(),
                AnswerRelevancy(), 
                ContextRelevance()
            ]
        
        # Initialize log file if it doesn't exist
        if not self.log_file.exists():
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create empty log file with proper structure"""
        initial_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_evaluations": 0
            },
            "evaluations": []
        }
        with open(self.log_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
        logger.info(f"Initialized evaluation log: {self.log_file}")
    
    def _load_log(self) -> Dict[str, Any]:
        """Load existing evaluation log"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading log: {e}")
            self._initialize_log_file()
            return self._load_log()
    
    def _save_log(self, data: Dict[str, Any]):
        """Save evaluation log"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving log: {e}")
    
    def _format_contexts(self, products: List[Dict]) -> List[str]:
        """Format product data for RAGAS context"""
        contexts = []
        for product in products:
            # Extract key product information
            name = product.get('Product Name', product.get('product_name', 'Unknown'))
            brand = product.get('Brand Name', product.get('brand_name', ''))
            category = product.get('Category', product.get('category', ''))
            description = product.get('About Product', product.get('about_product', ''))
            
            # Create formatted context
            context_parts = [f"Product: {name}"]
            if brand and str(brand).lower() not in ['nan', 'none', '']:
                context_parts.append(f"Brand: {brand}")
            if category and str(category).lower() not in ['nan', 'none', '']:
                context_parts.append(f"Category: {category}")
            if description and str(description).lower() not in ['nan', 'none', '']:
                context_parts.append(f"Description: {description}")
            
            contexts.append(" | ".join(context_parts))
        
        return contexts
    
    async def evaluate_query(self, query: str, retrieved_products: List[Dict], 
                           ai_response: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single query using RAGAS metrics
        
        Args:
            query: User's search query
            retrieved_products: List of products returned by RAG system
            ai_response: AI's response to the user
            
        Returns:
            Dictionary with evaluation scores or None if evaluation fails
        """
        try:
            # Format data for RAGAS
            contexts = self._format_contexts(retrieved_products)
            
            if not contexts:
                logger.warning("No contexts available for evaluation")
                return None
            
            # Create dataset for RAGAS
            data = {
                "question": [query],
                "contexts": [contexts],
                "answer": [ai_response]
            }
            
            # Add ground truth if using synthetic approach
            if self.use_synthetic_ground_truth:
                ground_truth = await self._generate_synthetic_ground_truth(query, contexts)
                if ground_truth:
                    data["ground_truth"] = [ground_truth]
                else:
                    logger.warning("Failed to generate ground truth, falling back to reference-free evaluation")
                    # Temporarily switch to reference-free metrics
                    original_metrics = self.metrics
                    self.metrics = [Faithfulness(), AnswerRelevancy(), ContextRelevance()]
            
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            logger.info(f"Running RAGAS evaluation for query: '{query[:50]}...'")
            result = evaluate(dataset, metrics=self.metrics)
            
            # Extract scores based on available metrics
            scores = {}
            if "faithfulness" in result:
                scores["faithfulness"] = float(result["faithfulness"])
            if "answer_relevancy" in result:
                scores["answer_relevancy"] = float(result["answer_relevancy"])
            if "context_relevance" in result:
                scores["context_relevance"] = float(result["context_relevance"])
            
            # Add ground truth metrics if available
            if self.use_synthetic_ground_truth and "ground_truth" in data:
                if "context_precision" in result:
                    scores["context_precision"] = float(result["context_precision"])
                if "context_recall" in result:
                    scores["context_recall"] = float(result["context_recall"])
            
            # Restore original metrics if we temporarily changed them
            if self.use_synthetic_ground_truth and 'original_metrics' in locals():
                self.metrics = original_metrics
            
            logger.info(f"RAGAS evaluation completed. Scores: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return None
    
    def log_evaluation(self, query: str, retrieved_products: List[Dict], 
                      ai_response: str, scores: Dict[str, float]):
        """Log evaluation results to persistent storage"""
        try:
            # Load existing log
            log_data = self._load_log()
            
            # Create evaluation entry
            evaluation_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "num_products_retrieved": len(retrieved_products),
                "ai_response_length": len(ai_response),
                "scores": scores,
                "product_names": [
                    p.get('Product Name', p.get('product_name', 'Unknown'))[:100] 
                    for p in retrieved_products[:3]  # Store first 3 product names
                ]
            }
            
            # Add to log
            log_data["evaluations"].append(evaluation_entry)
            log_data["metadata"]["total_evaluations"] += 1
            log_data["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Save updated log
            self._save_log(log_data)
            
            logger.info(f"Logged evaluation for query: '{query[:50]}...'")
            
        except Exception as e:
            logger.error(f"Failed to log evaluation: {e}")
    
    async def evaluate_and_log(self, query: str, retrieved_products: List[Dict], 
                             ai_response: str):
        """
        Complete evaluation pipeline: evaluate and log results
        This is the main method to call from the RAG system
        """
        try:
            # Run evaluation
            scores = await self.evaluate_query(query, retrieved_products, ai_response)
            
            if scores:
                # Log results
                self.log_evaluation(query, retrieved_products, ai_response, scores)
            else:
                logger.warning(f"Skipping log for failed evaluation: '{query[:50]}...'")
                
        except Exception as e:
            logger.error(f"Complete evaluation pipeline failed: {e}")
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate performance metrics across all evaluations"""
        try:
            log_data = self._load_log()
            evaluations = log_data.get("evaluations", [])
            
            if not evaluations:
                return {
                    "total_evaluations": 0,
                    "average_scores": {},
                    "recent_performance": "No data available"
                }
            
            # Calculate averages
            score_sums = {}
            score_counts = {}
            
            for eval_entry in evaluations:
                scores = eval_entry.get("scores", {})
                for metric, value in scores.items():
                    if metric not in score_sums:
                        score_sums[metric] = 0
                        score_counts[metric] = 0
                    score_sums[metric] += value
                    score_counts[metric] += 1
            
            # Calculate averages
            average_scores = {}
            for metric in score_sums:
                if score_counts[metric] > 0:
                    average_scores[metric] = round(score_sums[metric] / score_counts[metric], 3)
            
            # Recent performance (last 10 evaluations)
            recent_evals = evaluations[-10:] if len(evaluations) >= 10 else evaluations
            recent_avg = {}
            if recent_evals:
                for metric in average_scores:
                    recent_scores = [e.get("scores", {}).get(metric, 0) for e in recent_evals]
                    recent_avg[metric] = round(sum(recent_scores) / len(recent_scores), 3)
            
            return {
                "total_evaluations": len(evaluations),
                "average_scores": average_scores,
                "recent_scores": recent_avg,
                "last_updated": log_data.get("metadata", {}).get("last_updated", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate aggregate metrics: {e}")
            return {"error": str(e)}
    
    def get_recent_evaluations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent evaluation results for display"""
        try:
            log_data = self._load_log()
            evaluations = log_data.get("evaluations", [])
            
            # Return most recent evaluations
            recent = evaluations[-limit:] if len(evaluations) > limit else evaluations
            
            # Reverse to show newest first
            return list(reversed(recent))
            
        except Exception as e:
            logger.error(f"Failed to get recent evaluations: {e}")
            return []
    
    async def _generate_synthetic_ground_truth(self, query: str, contexts: List[str]) -> str:
        """Generate a synthetic reference answer for evaluation purposes"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            context_text = "\n".join(contexts)
            
            response = await client.chat.completions.acreate(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert e-commerce assistant. Generate a comprehensive, accurate reference answer for the given query based on the provided product context. 

This reference answer will be used for evaluation purposes, so it should be:
1. Factually accurate based on the product data
2. Comprehensive and helpful
3. Well-structured and clear
4. Focused on the user's specific needs

Only use information from the provided context."""
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nProduct Context:\n{context_text}\n\nGenerate a reference answer:"
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic ground truth: {e}")
            return ""