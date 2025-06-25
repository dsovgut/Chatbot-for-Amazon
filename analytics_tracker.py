"""
Simple Analytics Tracker for E-commerce RAG System
Tracks performance metrics without requiring ground truth data
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleAnalyticsTracker:
    """Track analytics for e-commerce RAG system without ground truth requirements"""
    
    def __init__(self, log_file: str = "analytics_log.json"):
        self.log_file = Path(log_file)
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the analytics log file"""
        if not self.log_file.exists():
            initial_data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0",
                    "description": "E-commerce RAG Analytics"
                },
                "queries": [],
                "summary": {
                    "total_queries": 0,
                    "total_response_time": 0,
                    "avg_response_time": 0,
                    "search_modes": {"text": 0, "image": 0, "multimodal": 0},
                    "avg_similarity_score": 0,
                    "avg_products_retrieved": 0
                }
            }
            with open(self.log_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def track_query(self, query_data: Dict) -> bool:
        """
        Track a single query with its metrics
        
        Args:
            query_data: Dictionary containing:
                - query: User's search query
                - response_time: Total response time in seconds
                - embedding_time: Time to generate embeddings
                - retrieval_time: Time for vector search
                - search_mode: Type of search (text/image/multimodal)
                - num_products: Number of products retrieved
                - similarity_scores: List of similarity scores
                - avg_similarity: Average similarity score
                - timestamp: Query timestamp
        """
        try:
            # Load existing data
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            # Add new query
            data["queries"].append(query_data)
            
            # Update summary statistics
            summary = data["summary"]
            summary["total_queries"] += 1
            summary["total_response_time"] += query_data.get("response_time", 0)
            summary["avg_response_time"] = summary["total_response_time"] / summary["total_queries"]
            
            # Update search mode counts
            search_mode = query_data.get("search_mode", "text")
            if search_mode in summary["search_modes"]:
                summary["search_modes"][search_mode] += 1
            
            # Update similarity and product counts
            num_products = query_data.get("num_products", 0)
            avg_similarity = query_data.get("avg_similarity", 0)
            
            # Calculate running averages
            total_queries = summary["total_queries"]
            prev_avg_similarity = summary["avg_similarity_score"]
            prev_avg_products = summary["avg_products_retrieved"]
            
            summary["avg_similarity_score"] = (
                (prev_avg_similarity * (total_queries - 1) + avg_similarity) / total_queries
            )
            summary["avg_products_retrieved"] = (
                (prev_avg_products * (total_queries - 1) + num_products) / total_queries
            )
            
            # Save updated data
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Tracked query analytics: {query_data.get('query', 'Unknown')[:50]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track query analytics: {e}")
            return False
    
    def get_analytics_summary(self) -> Dict:
        """Get summary analytics"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            return data["summary"]
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent queries"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            return data["queries"][-limit:]
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            return []
    
    def get_performance_trends(self) -> Dict:
        """Get performance trends over time"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            queries = data["queries"]
            if not queries:
                return {}
            
            # Calculate trends
            recent_queries = queries[-10:] if len(queries) >= 10 else queries
            
            recent_avg_response_time = sum(q.get("response_time", 0) for q in recent_queries) / len(recent_queries)
            recent_avg_similarity = sum(q.get("avg_similarity", 0) for q in recent_queries) / len(recent_queries)
            
            return {
                "recent_avg_response_time": recent_avg_response_time,
                "recent_avg_similarity": recent_avg_similarity,
                "total_queries": len(queries),
                "queries_last_hour": len([q for q in queries if self._is_recent(q.get("timestamp", ""), hours=1)]),
                "queries_today": len([q for q in queries if self._is_recent(q.get("timestamp", ""), hours=24)])
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return {}
    
    def _is_recent(self, timestamp_str: str, hours: int = 1) -> bool:
        """Check if timestamp is within the last N hours"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now()
            diff = now - timestamp.replace(tzinfo=None)
            return diff.total_seconds() < (hours * 3600)
        except:
            return False 