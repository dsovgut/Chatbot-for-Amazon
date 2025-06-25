import chainlit as cl
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ragas_evaluator import RAGASEvaluator
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize evaluator
evaluator = None

def initialize_evaluator():
    """Initialize the RAGAS evaluator"""
    global evaluator
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            evaluator = RAGASEvaluator(openai_api_key)
            return True
        return False
    except Exception as e:
        print(f"Failed to initialize evaluator: {e}")
        return False

@cl.on_chat_start
async def start():
    """Initialize the evaluation dashboard"""
    
    # Initialize evaluator
    if not initialize_evaluator():
        await cl.Message(
            content="❌ **Error**: Could not initialize RAGAS evaluator. Please check your OpenAI API key."
        ).send()
        return
    
    # Display dashboard
    await display_evaluation_dashboard()

async def display_evaluation_dashboard():
    """Display the main evaluation dashboard"""
    
    # Get aggregate metrics
    aggregate_metrics = evaluator.get_aggregate_metrics()
    
    # Create dashboard content
    dashboard_content = create_dashboard_content(aggregate_metrics)
    
    # Send dashboard
    await cl.Message(content=dashboard_content).send()
    
    # Get recent evaluations
    recent_evaluations = evaluator.get_recent_evaluations(limit=10)
    
    if recent_evaluations:
        # Display recent evaluations
        recent_content = create_recent_evaluations_content(recent_evaluations)
        await cl.Message(content=recent_content).send()
    else:
        await cl.Message(
            content="📝 **No evaluation data yet**\n\nStart using the main chat to generate evaluation data!"
        ).send()

def create_dashboard_content(metrics: dict) -> str:
    """Create the main dashboard content"""
    
    if "error" in metrics:
        return f"❌ **Error loading metrics**: {metrics['error']}"
    
    total_evals = metrics.get("total_evaluations", 0)
    
    if total_evals == 0:
        return """
# 📊 RAG Evaluation Dashboard

## 🎯 System Status
**No evaluations yet** - Start using the main chat to generate data!

## 📈 How It Works
- Every user query is automatically evaluated using RAGAS
- Metrics are calculated in real-time and stored persistently
- This dashboard shows aggregate performance across all queries

## 🔍 Metrics Explained
- **Faithfulness**: Are responses grounded in retrieved products?
- **Answer Relevancy**: How well do responses address user queries?
- **Context Precision**: How relevant are retrieved products?
- **Context Recall**: Are we retrieving enough relevant products?
- **Context Relevancy**: How focused are our retrievals?

*Start chatting to see your evaluation data!*
"""
    
    avg_scores = metrics.get("average_scores", {})
    recent_scores = metrics.get("recent_scores", {})
    last_updated = metrics.get("last_updated", "Unknown")
    
    # Format scores
    def format_score(score):
        if isinstance(score, (int, float)):
            return f"{score:.3f}"
        return "N/A"
    
    def get_emoji(score):
        if isinstance(score, (int, float)):
            if score >= 0.8:
                return "🟢"
            elif score >= 0.6:
                return "🟡"
            else:
                return "🔴"
        return "⚪"
    
    content = f"""
# 📊 RAG Evaluation Dashboard

## 🎯 Overall Performance
**Total Evaluations**: {total_evals}  
**Last Updated**: {last_updated[:19] if last_updated != "Unknown" else "Unknown"}

## 📈 Average Scores (All Time)
"""
    
    for metric, score in avg_scores.items():
        emoji = get_emoji(score)
        formatted_score = format_score(score)
        metric_name = metric.replace("_", " ").title()
        content += f"**{metric_name}**: {emoji} {formatted_score}\n"
    
    if recent_scores:
        content += "\n## 🔄 Recent Performance (Last 10 Queries)\n"
        for metric, score in recent_scores.items():
            emoji = get_emoji(score)
            formatted_score = format_score(score)
            metric_name = metric.replace("_", " ").title()
            content += f"**{metric_name}**: {emoji} {formatted_score}\n"
    
    content += """

## 📊 Score Interpretation
- 🟢 **0.8+**: Excellent performance
- 🟡 **0.6-0.8**: Good performance  
- 🔴 **<0.6**: Needs improvement
"""
    
    return content

def create_recent_evaluations_content(evaluations: list) -> str:
    """Create content showing recent evaluations"""
    
    content = "# 📝 Recent Query Evaluations\n\n"
    
    for i, eval_data in enumerate(evaluations[:10], 1):
        query = eval_data.get("query", "Unknown query")
        timestamp = eval_data.get("timestamp", "Unknown time")
        scores = eval_data.get("scores", {})
        num_products = eval_data.get("num_products_retrieved", 0)
        product_names = eval_data.get("product_names", [])
        
        # Format timestamp
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = timestamp[:19] if len(timestamp) > 19 else timestamp
        
        content += f"## {i}. Query: \"{query[:60]}{'...' if len(query) > 60 else ''}\"\n"
        content += f"**Time**: {time_str} | **Products Retrieved**: {num_products}\n\n"
        
        # Show scores
        if scores:
            content += "**Scores**: "
            score_parts = []
            for metric, score in scores.items():
                if isinstance(score, (int, float)):
                    emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                    score_parts.append(f"{metric.replace('_', ' ').title()}: {emoji} {score:.2f}")
            content += " | ".join(score_parts)
        
        # Show top products
        if product_names:
            content += f"\n**Top Products**: {', '.join(product_names[:2])}"
            if len(product_names) > 2:
                content += f" (+{len(product_names)-2} more)"
        
        content += "\n\n---\n\n"
    
    return content

@cl.on_message
async def handle_message(message: cl.Message):
    """Handle messages in evaluation tab"""
    
    user_input = message.content.lower().strip()
    
    if user_input in ["refresh", "update", "reload"]:
        await cl.Message(content="🔄 Refreshing dashboard...").send()
        await display_evaluation_dashboard()
    
    elif user_input in ["help", "commands"]:
        help_content = """
# 🆘 Evaluation Dashboard Help

## Available Commands
- **refresh** / **update** / **reload**: Refresh the dashboard with latest data
- **help** / **commands**: Show this help message

## About This Dashboard
This page shows real-time evaluation metrics for the RAG system using RAGAS (RAG Assessment). 

Every query in the main chat is automatically evaluated and the results are stored here permanently.

## Metrics Explained
- **Faithfulness**: Measures if AI responses are grounded in retrieved product data
- **Answer Relevancy**: How well responses address the user's specific query  
- **Context Precision**: Quality and relevance of retrieved products
- **Context Recall**: Whether we retrieved enough relevant products
- **Context Relevancy**: How focused our product retrievals are

## Tips
- Use the main chat tab to generate evaluation data
- Check back here periodically to monitor system performance
- Look for patterns in low-scoring queries to identify improvement areas
"""
        await cl.Message(content=help_content).send()
    
    else:
        await cl.Message(
            content="💡 **Tip**: Type 'refresh' to update the dashboard or 'help' for available commands."
        ).send() 