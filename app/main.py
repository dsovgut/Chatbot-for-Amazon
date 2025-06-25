# app/main.py

# --- Imports ---
import chainlit as cl
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio
import logging
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Optional
import base64
from PIL import Image
import io
import json
import time
from datetime import datetime
import pandas as pd
import re

# Import our custom RAG system and evaluation
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from multimodal_rag import MultimodalEcommerceRAG
from evaluation_metrics import RealWorldRAGEvaluator
from analytics_tracker import SimpleAnalyticsTracker

# --- Load Environment Variables ---
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize OpenAI Client ---
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# TEMPORARILY DISABLED - This might be causing RAGAS evaluation prompts to show in UI
# cl.instrument_openai()

# --- Initialize RAG System and Evaluators ---
rag_system = None
evaluator = None
analytics_tracker = None

def initialize_rag_system():
    """Initialize the RAG system and analytics tracker"""
    global rag_system, evaluator, analytics_tracker
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
            
        # Fix the data directory path - go up one level from app directory
        data_dir = Path(__file__).parent.parent / "data"
        rag_system = MultimodalEcommerceRAG(openai_api_key, data_dir=str(data_dir))
        evaluator = RealWorldRAGEvaluator(rag_system, openai_api_key)
        analytics_tracker = SimpleAnalyticsTracker("analytics_log.json")
        logger.info("RAG system and analytics tracker initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return False

# --- Chat Profiles Configuration ---
@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Shopping Assistant",
            markdown_description="**AI Shopping Assistant** - Search for products using text or images, get recommendations, and ask follow-up questions.",
            icon="🛍️",
        ),
        cl.ChatProfile(
            name="Evaluation Dashboard", 
            markdown_description="**Analytics Dashboard** - View real-time analytics for the e-commerce RAG system.",
            icon="📊",
        ),
    ]

# --- System Prompt for E-commerce Assistant ---
SYSTEM_PROMPT = """You are an expert AI assistant for a multimodal e-commerce platform.

Your capabilities include:
- Analyzing product images and descriptions using advanced computer vision
- Providing detailed product information and recommendations
- Comparing products and features
- Answering questions about specifications, usage, and compatibility
- Helping customers find products based on text descriptions or uploaded images
- Providing shopping advice and guidance

You have access to a comprehensive product database with images and detailed descriptions.
Always be helpful, accurate, and customer-focused in your responses."""

# --- Helper Functions ---

def save_uploaded_file(file_element) -> Optional[str]:
    """Save uploaded file to temporary location and return path"""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        import uuid
        file_id = str(uuid.uuid4())
        file_extension = Path(file_element.name).suffix.lower()
        temp_path = temp_dir / f"{file_id}{file_extension}"
        
        # Get file content - try the most common Chainlit patterns
        file_content = None
        
        # Method 1: Direct content attribute (most common)
        if hasattr(file_element, 'content') and file_element.content:
            file_content = file_element.content
            logger.info(f"Using .content, size: {len(file_content)}")
        
        # Method 2: Path-based access
        elif hasattr(file_element, 'path') and file_element.path:
            with open(file_element.path, 'rb') as f:
                file_content = f.read()
            logger.info(f"Using .path, size: {len(file_content)}")
        
        # Method 3: File object
        elif hasattr(file_element, 'file') and file_element.file:
            file_content = file_element.file.read()
            logger.info(f"Using .file, size: {len(file_content)}")
        
        if not file_content:
            logger.error(f"No file content found for {file_element.name}")
            return None
        
        # Save file
        with open(temp_path, "wb") as f:
            f.write(file_content)
            
        # Verify file was saved
        if temp_path.exists() and temp_path.stat().st_size > 0:
            logger.info(f"✅ File saved: {temp_path.name}, {temp_path.stat().st_size} bytes")
            return str(temp_path)
        else:
            logger.error(f"❌ File save failed: {temp_path}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Error saving file: {e}")
        return None

def is_image_file(file_path: str) -> bool:
    """Check if file is an image"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    return Path(file_path).suffix.lower() in image_extensions

def cleanup_temp_files():
    """Clean up temporary uploaded files"""
    try:
        temp_dir = Path("temp_uploads")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")

async def display_product_results(products: List[Dict], message_content: str = ""):
    """Display clean AI response with organized sections"""
    if not products:
        await cl.Message(content="No products found matching your criteria.").send()
        return
    
    # Send only the AI response - clean and uncluttered
    if message_content:
        await cl.Message(content=message_content).send()
    
    # Create citations content
    citations_content = "## 📚 Sources & Citations\n\n"
    for i, product in enumerate(products[:8], 1):  # Show all 8 products
        try:
            # Debug: Log the raw product data for the first product
            if i == 1:
                logger.info(f"Raw product data for first result: {dict(list(product.items())[:10])}")
            
            # Safely extract values with proper NaN handling
            def safe_get(value, default=''):
                """Safely get value handling numpy NaN and None"""
                if value is None:
                    return default
                if pd.isna(value):
                    return default
                if isinstance(value, (int, float)) and pd.isna(value):
                    return default
                str_val = str(value).strip()
                if str_val.lower() in ['nan', 'none', 'null', '']:
                    return default
                return str_val
            
            # Map the correct column names from the CSV with safe extraction
            product_name = safe_get(product.get('Product Name', product.get('product_name', '')), 'Unknown Product')
            brand = safe_get(product.get('Brand Name', product.get('brand_name', '')), 'Brand not specified')
            category = safe_get(product.get('Category', product.get('category', '')), 'Category not specified')
            about_product = safe_get(product.get('About Product', product.get('about_product', '')), '')
            image_urls = safe_get(product.get('Image', product.get('image', '')), '')
            similarity = product.get('similarity_score', 0)
            list_price = safe_get(product.get('List Price', ''), '')
            selling_price = safe_get(product.get('Selling Price', ''), '')
            
            # Debug: Log the mapped values for the first product
            if i == 1:
                logger.info(f"Mapped values - Name: '{product_name}', Brand: '{brand}', Category: '{category}', About: '{about_product[:50] if about_product else 'None'}'")
            
            # Clean up product name
            if pd.isna(product_name) or str(product_name).lower() in ['nan', '', 'none', 'null']:
                product_name = 'Unknown Product'
            else:
                product_name = str(product_name).strip()
            
            # Clean up brand (handle NaN and empty values)
            if pd.isna(brand) or str(brand).lower() in ['nan', '', 'none', 'null']:
                brand = 'Brand not specified'
            else:
                brand = str(brand).strip()
                # If brand is still empty after stripping
                if not brand:
                    brand = 'Brand not specified'
            
            # Clean up category (handle NaN and empty values)
            if pd.isna(category) or str(category).lower() in ['nan', '', 'none', 'null']:
                category = 'Category not specified'
            else:
                category = str(category).strip()
                # If category is still empty after stripping
                if not category:
                    category = 'Category not specified'
                else:
                    # Simplify long category paths
                    if ' | ' in category:
                        category_parts = category.split(' | ')
                        if len(category_parts) > 2:
                            category = ' > '.join(category_parts[:2]) + '...'
            
            # Clean up description (handle NaN and messy formatting)
            if pd.isna(about_product) or str(about_product).lower() in ['nan', '', 'none', 'null']:
                description = 'No description available'
            else:
                description = str(about_product).strip()
                
                # Remove common Amazon prefixes
                if description.startswith('Make sure this fits'):
                    parts = description.split(' | ', 1)
                    if len(parts) > 1:
                        description = parts[1]
                
                # Clean up messy formatting (dashes, pipes, etc.)
                description = description.replace(' | ', ' • ')
                description = re.sub(r'\s*-\s*-\s*', ' • ', description)
                description = re.sub(r'\s*\|\s*', ' • ', description)
                description = re.sub(r'\s+', ' ', description)
                
                # Remove if it's just dashes or meaningless
                if re.match(r'^[\s\-\|\•\.]*$', description) or len(description.strip()) < 10:
                    description = 'No description available'
                
                # Additional check for repetitive patterns
                if description.count('•') > 5 and len(description.replace('•', '').replace(' ', '').replace('-', '')) < 10:
                    description = 'No description available'
                
                # Truncate if too long
                if len(description) > 150:
                    description = description[:150] + "..."
            
            # Debug: Log the final cleaned values for the first product
            if i == 1:
                logger.info(f"Final cleaned values - Name: '{product_name}', Brand: '{brand}', Category: '{category}', Description: '{description[:50]}'")
            
            # Get Amazon URL (extract from image URLs)
            amazon_url = 'Not available'
            if not pd.isna(image_urls) and str(image_urls) != 'nan':
                urls = str(image_urls).split('|')
                if urls and urls[0].startswith('https://'):
                    # Convert image URL to product URL (approximate)
                    amazon_url = urls[0]
            
            # Format pricing
            price_info = ""
            if not pd.isna(selling_price) and str(selling_price) not in ['nan', '', 'none', 'null']:
                price_info = f" • **Price:** {selling_price}"
                if not pd.isna(list_price) and str(list_price) not in ['nan', '', 'none', 'null'] and str(list_price) != str(selling_price):
                    price_info += f" (was {list_price})"
            
            # Create citation entry
            citations_content += f"""### [{i}] {product_name}
- **Brand:** {brand}
- **Category:** {category}
- **Relevance:** {similarity:.1%}{price_info}
- **Description:** {description}
- **🔗 Amazon:** [View Product]({amazon_url})

---

"""
            
        except Exception as e:
            logger.error(f"Error processing citation {i}: {e}")
            citations_content += f"### [{i}] Error processing product citation\n\n---\n\n"
    
    # Send citations as a separate message
    await cl.Message(content=citations_content).send()
    
    # Collect product images
    image_elements = []
    for i, product in enumerate(products[:8], 1):  # Show all 8 images
        try:
            image_path = product.get('image_path', '')
            
            # Fix the image path - make it relative to the app directory
            if image_path and not os.path.isabs(image_path):
                # Convert relative path to be relative to app directory
                image_path = os.path.join('..', image_path)
            
            logger.info(f"Processing image {i}: path='{image_path}', exists={os.path.exists(image_path) if image_path else False}")
            
            if image_path and os.path.exists(image_path):
                product_name = product.get('Product Name', product.get('product_name', f'Product {i}'))
                image_element = cl.Image(
                    name=f"product_{i}",
                    display="inline", 
                    path=image_path
                )
                image_elements.append(image_element)
                logger.info(f"Added image element for product {i}: {product_name}")
            else:
                logger.warning(f"Image not found for product {i}: {image_path}")
        except Exception as e:
            logger.error(f"Error processing image for product {i}: {e}")
    
    # Send images as a separate message if available
    if image_elements:
        logger.info(f"Sending {len(image_elements)} product images")
        await cl.Message(
            content="## 📷 Product Images",
            elements=image_elements
        ).send()
    else:
        logger.warning("No image elements to display")

async def display_evaluation_metrics(query_metrics: Dict):
    """Display performance metrics in a separate message"""
    if not query_metrics:
        return
    
    metrics_content = f"""## ⚡ Performance Metrics

### 🎯 Search Quality
- **Products Found:** {query_metrics.get('num_results', 0)}
- **Search Mode:** {query_metrics.get('search_mode', 'N/A')}
- **Average Relevance:** {query_metrics.get('avg_relevance', 0):.1%}

### ⏱️ Response Times
- **Total Response:** {query_metrics.get('response_time', 0):.2f}s
- **Embedding Generation:** {query_metrics.get('embedding_time', 0):.3f}s
- **Vector Search:** {query_metrics.get('retrieval_time', 0):.3f}s

### 🔍 Similarity Scores"""
    
    # Add top similarity scores
    similarity_scores = query_metrics.get('similarity_scores', [])
    for i, score in enumerate(similarity_scores[:8], 1):  # Show all 8 similarity scores
        metrics_content += f"\n- **Product {i}:** {score:.3f}"
    
    await cl.Message(content=metrics_content).send()

@cl.action_callback("show_citations")
async def show_citations_callback(action):
    """Show/hide citations"""
    products = cl.user_session.get("current_products", [])
    
    if not products:
        await cl.Message(content="No citation data available.").send()
        return
    
    citations_content = "## 📚 Sources & Citations\n\n"
    for i, product in enumerate(products[:3], 1):
        try:
            # Map the correct column names from the CSV
            product_name = product.get('Product Name', product.get('product_name', 'Unknown Product'))
            brand = product.get('Brand Name', product.get('brand_name', ''))
            category = product.get('Category', product.get('category', ''))
            about_product = product.get('About Product', product.get('about_product', ''))
            image_urls = product.get('Image', product.get('image', ''))
            similarity = product.get('similarity_score', 0)
            list_price = product.get('List Price', '')
            selling_price = product.get('Selling Price', '')
            
            # Clean up brand (handle NaN and empty values)
            if pd.isna(brand) or str(brand).lower() in ['nan', '', 'none', 'null']:
                brand = 'Brand not specified'
            else:
                brand = str(brand).strip()
            
            # Clean up category (handle NaN and empty values)
            if pd.isna(category) or str(category).lower() in ['nan', '', 'none', 'null']:
                category = 'Category not specified'
            else:
                category = str(category).strip()
                # Simplify long category paths
                if ' | ' in category:
                    category_parts = category.split(' | ')
                    if len(category_parts) > 2:
                        category = ' > '.join(category_parts[:2]) + '...'
            
            # Clean up description (handle NaN and messy formatting)
            if pd.isna(about_product) or str(about_product).lower() in ['nan', '', 'none', 'null']:
                description = 'No description available'
            else:
                description = str(about_product).strip()
                
                # Remove common Amazon prefixes
                if description.startswith('Make sure this fits'):
                    parts = description.split(' | ', 1)
                    if len(parts) > 1:
                        description = parts[1]
                
                # Clean up messy formatting (dashes, pipes, etc.)
                description = description.replace(' | ', ' • ')
                description = re.sub(r'\s*-\s*-\s*', ' • ', description)
                description = re.sub(r'\s*\|\s*', ' • ', description)
                description = re.sub(r'\s+', ' ', description)
                
                # Remove if it's just dashes or meaningless
                if re.match(r'^[\s\-\|\•\.]*$', description) or len(description.strip()) < 10:
                    description = 'No description available'
                
                # Truncate if too long
                if len(description) > 150:
                    description = description[:150] + "..."
            
            # Get Amazon URL (extract from image URLs)
            amazon_url = 'Not available'
            if not pd.isna(image_urls) and str(image_urls) != 'nan':
                urls = str(image_urls).split('|')
                if urls and urls[0].startswith('https://'):
                    # Convert image URL to product URL (approximate)
                    amazon_url = urls[0]
            
            # Format pricing
            price_info = ""
            if not pd.isna(selling_price) and str(selling_price) not in ['nan', '', 'none', 'null']:
                price_info = f" • **Price:** {selling_price}"
                if not pd.isna(list_price) and str(list_price) not in ['nan', '', 'none', 'null'] and str(list_price) != str(selling_price):
                    price_info += f" (was {list_price})"
            
            # Create citation entry
            citations_content += f"""### [{i}] {product_name}
- **Brand:** {brand}
- **Category:** {category}
- **Relevance:** {similarity:.1%}{price_info}
- **Description:** {description}
- **🔗 Amazon:** [View Product]({amazon_url})

---

"""
            
        except Exception as e:
            logger.error(f"Error processing citation {i}: {e}")
            citations_content += f"### [{i}] Error processing product citation\n\n---\n\n"
    
    await cl.Message(content=citations_content).send()

@cl.action_callback("show_images")
async def show_images_callback(action):
    """Show product images"""
    products = cl.user_session.get("current_products", [])
    
    if not products:
        await cl.Message(content="No image data available.").send()
        return
    
    # Collect product images
    image_elements = []
    for i, product in enumerate(products[:8], 1):  # Show all 8 images
        try:
            image_path = product.get('image_path', '')
            
            # Fix the image path - make it relative to the app directory
            if image_path and not os.path.isabs(image_path):
                # Convert relative path to be relative to app directory
                image_path = os.path.join('..', image_path)
            
            logger.info(f"Processing image {i}: path='{image_path}', exists={os.path.exists(image_path) if image_path else False}")
            
            if image_path and os.path.exists(image_path):
                product_name = product.get('Product Name', product.get('product_name', f'Product {i}'))
                image_element = cl.Image(
                    name=f"product_{i}",
                    display="inline", 
                    path=image_path
                )
                image_elements.append(image_element)
                logger.info(f"Added image element for product {i}: {product_name}")
            else:
                logger.warning(f"Image not found for product {i}: {image_path}")
        except Exception as e:
            logger.error(f"Error processing image for product {i}: {e}")
    
    if image_elements:
        logger.info(f"Sending {len(image_elements)} product images")
        await cl.Message(
            content="## 📷 Product Images",
            elements=image_elements
        ).send()
    else:
        logger.warning("No image elements to display")

@cl.action_callback("show_metrics")
async def show_metrics_callback(action):
    """Show performance metrics"""
    query_metrics = cl.user_session.get("current_metrics", {})
    
    if not query_metrics:
        await cl.Message(content="No metrics data available.").send()
        return
    
    metrics_content = f"""## ⚡ Performance Metrics

### 🎯 Search Quality
- **Products Found:** {query_metrics.get('num_results', 0)}
- **Search Mode:** {query_metrics.get('search_mode', 'N/A')}
- **Average Relevance:** {query_metrics.get('avg_relevance', 0):.1%}

### ⏱️ Response Times
- **Total Response:** {query_metrics.get('response_time', 0):.2f}s
- **Embedding Generation:** {query_metrics.get('embedding_time', 0):.3f}s
- **Vector Search:** {query_metrics.get('retrieval_time', 0):.3f}s

### 🔍 Similarity Scores"""
    
    # Add top similarity scores
    similarity_scores = query_metrics.get('similarity_scores', [])
    for i, score in enumerate(similarity_scores[:8], 1):  # Show all 8 similarity scores
        metrics_content += f"\n- **Product {i}:** {score:.3f}"
    
    await cl.Message(content=metrics_content).send()

async def calculate_query_metrics(result: Dict, response_time: float, embedding_time: float, retrieval_time: float) -> Dict:
    """Calculate metrics for the current query"""
    retrieved_products = result.get('retrieved_products', [])
    
    metrics = {
        'num_results': len(retrieved_products),
        'search_mode': result.get('search_mode', 'unknown'),
        'response_time': response_time,
        'embedding_time': embedding_time,
        'retrieval_time': retrieval_time,
        'similarity_scores': [p.get('similarity_score', 0) for p in retrieved_products],
        'avg_relevance': sum(p.get('similarity_score', 0) for p in retrieved_products) / len(retrieved_products) if retrieved_products else 0
    }
    
    return metrics

# --- RAG System Integration ---

async def get_rag_response(user_query: str, chat_history: list, uploaded_image_path: str = "", top_k=8) -> Dict:
    """
    Get response from the multimodal RAG system with performance tracking
    """
    if not rag_system:
        return {
            'response': "I apologize, but the product search system is not available right now. Please try again later.",
            'retrieved_products': [],
            'error': "RAG system not initialized",
            'metrics': {}
        }
    
    try:
        # Track performance metrics
        start_time = time.time()
        
        # Measure embedding time
        embed_start = time.time()
        if uploaded_image_path:
            query_embedding = rag_system.get_clip_query_embedding(user_query, uploaded_image_path)
        else:
            query_embedding = rag_system.get_clip_query_embedding(user_query)
        embedding_time = time.time() - embed_start
        
        # Process query with RAG system
        result = await rag_system.process_query(
            text_query=user_query,
            image_path=uploaded_image_path,
            chat_history=chat_history,
            top_k=top_k
        )
        
        total_time = time.time() - start_time
        
        # Calculate retrieval time (approximate)
        retrieval_time = total_time - embedding_time - 1.0  # Subtract estimated LLM time
        retrieval_time = max(0, retrieval_time)
        
        # Calculate query metrics
        metrics = await calculate_query_metrics(result, total_time, embedding_time, retrieval_time)
        result['metrics'] = metrics
        
        return result
        
    except Exception as e:
        logger.error(f"Error in RAG system call: {e}")
        return {
            'response': f"I encountered an error while searching for products: {str(e)}",
            'retrieved_products': [],
            'error': str(e),
            'metrics': {}
        }

# --- Chainlit Event Handlers ---

@cl.on_chat_start
async def start_chat():
    """Initialize chat session"""
    # Get the current chat profile
    chat_profile = cl.user_session.get("chat_profile")
    
    # Initialize RAG system if not already done
    if not rag_system:
        success = initialize_rag_system()
        if not success:
            await cl.Message(
                content="⚠️ **System Notice**: The product database is currently unavailable. Some features may be limited."
            ).send()
    
    # Initialize session history and metrics
    cl.user_session.set("history", [{"role": "system", "content": SYSTEM_PROMPT}])
    cl.user_session.set("session_metrics", {
        'queries_count': 0,
        'total_response_time': 0,
        'avg_response_time': 0,
        'search_modes_used': [],
        'session_start': datetime.now().isoformat()
    })
    
    # Clean up any existing temp files
    cleanup_temp_files()
    
    # Handle different chat profiles
    if chat_profile == "Evaluation Dashboard":
        # Disable chat input for evaluation dashboard
        await cl.ChatSettings([]).send()
        await show_evaluation_dashboard()
    else:
        # Default to Shopping Assistant
        await show_shopping_assistant_welcome()

async def show_shopping_assistant_welcome():
    """Show welcome message for shopping assistant"""
    welcome_message = """# 🛍️ Welcome to Your AI Shopping Assistant!

I'm here to help you find the perfect products using advanced multimodal AI! Here's what I can do:

**🔍 Product Search:**
- Search by text description: *"wireless noise-cancelling headphones"*
- Upload product images for visual search
- Get detailed product information and specifications

**💡 Smart Recommendations:**
- Compare similar products
- Find alternatives and accessories
- Get personalized suggestions

**📊 Performance Tracking:**
- Real-time evaluation metrics
- Response quality assessment
- Multimodal search performance

**📱 How to use:**
- Type your product questions or requirements
- Upload images of products you're interested in
- Ask for comparisons, features, or recommendations

What can I help you find today?"""
    
    await cl.Message(content=welcome_message).send()
    
    # Display system status
    if rag_system and rag_system.products_df is not None:
        status_msg = f"""
**🔧 System Status:**
- Products in database: {len(rag_system.products_df):,}
- CLIP embeddings loaded: ✅
- Evaluation metrics: ✅
- Ready for multimodal search! 🚀
"""
        await cl.Message(content=status_msg).send()
    
    logger.info("Shopping Assistant session started successfully")

def _classify_query_intent(user_query: str) -> bool:
    """
    Use CLIP semantic similarity to intelligently determine if a query is a follow-up question
    rather than a new product search. This uses actual AI instead of brittle keyword matching.
    """
    if not rag_system or not hasattr(rag_system, 'clip_model'):
        # Fallback to simple logic if CLIP not available
        return len(user_query.split()) <= 3 and any(word in user_query.lower() for word in ['which', 'what', 'cheapest', 'expensive'])
    
    try:
        # Check for explicit follow-up indicators first
        query_lower = user_query.lower()
        explicit_followup_phrases = [
            "out of all of them", "of all of them", "among these", "between these", 
            "which one", "what about", "how about", "tell me more", "more details",
            "cheapest one", "most expensive", "best one", "which is better"
        ]
        
        # If query contains explicit follow-up phrases, it's definitely a follow-up
        if any(phrase in query_lower for phrase in explicit_followup_phrases):
            logger.info(f"Query: '{user_query}' | Explicit follow-up phrase detected")
            return True
        
        # Define example queries for each category
        followup_examples = [
            "which one is cheaper",
            "what about this one", 
            "which should I buy",
            "how much does it cost",
            "which is better",
            "what do you recommend",
            "compare these",
            "tell me more",
            "which one",
            "what about",
            "cheapest",
            "most expensive",
            "cheapest scooter out of all of them",
            "what's the cheapest of all of them",
            "which is the best one"
        ]
        
        product_search_examples = [
            "find me jeans",
            "show me dresses", 
            "I need headphones",
            "looking for shoes",
            "wireless mouse",
            "bluetooth speaker",
            "gaming laptop",
            "coffee maker",
            "running shoes",
            "winter jacket",
            "smartphone case",
            "office chair"
        ]
        
        # Get CLIP text embeddings for the user query
        query_embedding = rag_system.get_clip_query_embedding(user_query, "")
        
        if len(query_embedding) == 0:
            # Fallback if embedding fails
            return False
        
        # Get embeddings for example queries
        followup_embeddings = []
        for example in followup_examples:
            emb = rag_system.get_clip_query_embedding(example, "")
            if len(emb) > 0:
                followup_embeddings.append(emb)
        
        product_search_embeddings = []
        for example in product_search_examples:
            emb = rag_system.get_clip_query_embedding(example, "")
            if len(emb) > 0:
                product_search_embeddings.append(emb)
        
        if not followup_embeddings or not product_search_embeddings:
            return False
        
        # Calculate similarity to each category
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Average similarity to follow-up examples
        followup_similarities = []
        for emb in followup_embeddings:
            sim = cosine_similarity([query_embedding], [emb])[0][0]
            followup_similarities.append(sim)
        avg_followup_sim = np.mean(followup_similarities)
        
        # Average similarity to product search examples
        search_similarities = []
        for emb in product_search_embeddings:
            sim = cosine_similarity([query_embedding], [emb])[0][0]
            search_similarities.append(sim)
        avg_search_sim = np.mean(search_similarities)
        
        # Classify based on which category has higher similarity
        # Add a bias toward product search (require followup to be significantly more similar)
        followup_threshold = 0.1  # Follow-up must be at least 0.1 more similar than search
        
        is_followup = (avg_followup_sim - avg_search_sim) > followup_threshold
        
        logger.info(f"Query: '{user_query}' | Follow-up sim: {avg_followup_sim:.3f} | Search sim: {avg_search_sim:.3f} | Classification: {'Follow-up' if is_followup else 'New search'}")
        
        return is_followup
        
    except Exception as e:
        logger.error(f"Error in semantic query classification: {e}")
        # Conservative fallback
        return False

@cl.on_message
async def handle_message(msg: cl.Message):
    """Handle incoming messages with text and/or images"""
    # Get the current chat profile
    chat_profile = cl.user_session.get("chat_profile")
    
    # Route to appropriate handler based on profile
    if chat_profile == "Evaluation Dashboard":
        await handle_evaluation_message(msg)
    else:
        # Default to Shopping Assistant
        await handle_shopping_message(msg)

async def handle_evaluation_message(msg: cl.Message):
    """Handle messages in evaluation dashboard"""
    user_input = msg.content.lower().strip()
    
    if user_input in ["refresh", "update", "reload"]:
        await cl.Message(content="🔄 Refreshing dashboard...").send()
        await show_evaluation_dashboard()
    
    elif user_input in ["help", "commands"]:
        help_content = """
# 🆘 Analytics Dashboard Help

## Available Commands
- **refresh** / **update** / **reload**: Refresh the dashboard with latest data
- **help** / **commands**: Show this help message

## About This Dashboard
This page shows real-time analytics for the e-commerce RAG system.

Every query in the Shopping Assistant is automatically tracked and the performance metrics are stored here.

## Metrics Explained
- **Response Time**: How fast the system responds to queries
- **Similarity Scores**: How well retrieved products match user queries
- **Search Success Rate**: Percentage of queries that return relevant products
- **Query Distribution**: Types of searches (text, image, multimodal)
- **User Engagement**: Session patterns and follow-up behavior

## Tips
- Switch to the Shopping Assistant tab to generate analytics data
- Check back here periodically to monitor system performance
- Look for patterns in response times and similarity scores
"""
        await cl.Message(content=help_content).send()
    
    else:
        await cl.Message(
            content="📊 **Analytics Dashboard - Read Only**\n\nThis dashboard is for viewing system analytics only. To generate new data, please switch to the **Shopping Assistant** tab.\n\n💡 **Available commands**: `refresh`, `help`"
        ).send()

async def handle_shopping_message(msg: cl.Message):
    """Handle messages in shopping assistant"""
    history = cl.user_session.get("history", [])
    session_metrics = cl.user_session.get("session_metrics", {})
    user_query = msg.content or ""
    uploaded_image_path = ""
    
    # Add user message to history
    if user_query:
        history.append({"role": "user", "content": user_query})

    # Process uploaded files (images)
    if msg.elements:
        for element in msg.elements:
            if hasattr(element, 'content') and hasattr(element, 'name'):
                # Save uploaded file
                temp_path = save_uploaded_file(element)
                if temp_path and is_image_file(temp_path):
                    uploaded_image_path = temp_path
                    logger.info(f"Image uploaded: {element.name}")
                    break
    
    # Check if we have any input
    if not user_query and not uploaded_image_path:
        await cl.Message(
            content="Please provide a text query or upload an image to search for products."
        ).send()
        return
    
    # Determine if this is a follow-up question or new search
    is_followup = is_followup_question(user_query, history) and not uploaded_image_path
    
    if is_followup:
        # Handle as conversational follow-up without new search
        async with cl.Step(name="💭 Thinking about your question...") as step:
            step.output = "Analyzing your follow-up question..."
            
            response_text = await handle_followup_question(user_query, history)
            step.output = "Providing personalized advice"
        
        # Send response without product search
        await cl.Message(content=response_text).send()
        
        # Add to history
        history.append({"role": "assistant", "content": response_text})
        cl.user_session.set("history", history)
        
    else:
        # Handle as new product search (original logic)
        async with cl.Step(name="🔍 Searching products...") as step:
            step.output = "Analyzing your request and searching our product database..."
            
            # Get response from RAG system with metrics
            rag_result = await get_rag_response(user_query, history, uploaded_image_path, top_k=8)
            
            response_text = rag_result.get('response', '')
            retrieved_products = rag_result.get('retrieved_products', [])
            query_analysis = rag_result.get('query_analysis', '')
            query_metrics = rag_result.get('metrics', {})
            
            step.output = f"Found {len(retrieved_products)} relevant products"
        
        # Update session metrics
        session_metrics['queries_count'] += 1
        if 'response_time' in query_metrics:
            session_metrics['total_response_time'] += query_metrics['response_time']
            session_metrics['avg_response_time'] = session_metrics['total_response_time'] / session_metrics['queries_count']
        
        if 'search_mode' in query_metrics:
            session_metrics['search_modes_used'].append(query_metrics['search_mode'])
        
        cl.user_session.set("session_metrics", session_metrics)
        
        # Track analytics
        if analytics_tracker and query_metrics:
            analytics_data = {
                "query": user_query[:100],  # Truncate for privacy
                "response_time": query_metrics.get("response_time", 0),
                "embedding_time": query_metrics.get("embedding_time", 0),
                "retrieval_time": query_metrics.get("retrieval_time", 0),
                "search_mode": query_metrics.get("search_mode", "text"),
                "num_products": query_metrics.get("num_results", 0),
                "similarity_scores": query_metrics.get("similarity_scores", []),
                "avg_similarity": query_metrics.get("avg_relevance", 0),
                "timestamp": datetime.now().isoformat()
            }
            analytics_tracker.track_query(analytics_data)
        
        # Store retrieved products in session for follow-up questions
        if retrieved_products:
            cl.user_session.set("last_retrieved_products", retrieved_products)
        
        # Display query analysis if image was uploaded
        if query_analysis and uploaded_image_path:
            analysis_msg = f"**🔍 Image Analysis:**\n{query_analysis}\n\n"
            await cl.Message(content=analysis_msg).send()
        
        # Display main response with product results
        if retrieved_products:
            # Check if LLM response indicates no relevant products found
            no_products_indicators = [
                "couldn't find any products matching",
                "no products matching your request",
                "no relevant products found",
                "couldn't find products that match"
            ]
            
            response_lower = response_text.lower()
            llm_says_no_products = any(indicator in response_lower for indicator in no_products_indicators)
            
            if llm_says_no_products:
                # LLM determined products aren't relevant - don't show citations
                await cl.Message(content=response_text).send()
                if query_metrics:
                    await display_evaluation_metrics(query_metrics)
            else:
                # Display integrated response with products
                await display_product_results(retrieved_products, response_text)
                
                # Display evaluation metrics
                await display_evaluation_metrics(query_metrics)
            
        else:
            # No products found
            await cl.Message(content=response_text).send()
            if query_metrics:
                await display_evaluation_metrics(query_metrics)
        
        # Add assistant response to history WITH product context for follow-ups (but clean for display)
        if response_text:
            # Create enriched response that includes product data for follow-up context
            enriched_response = response_text
            
            # Add product details to the assistant message for LLM context (but not displayed)
            if retrieved_products:
                product_context = "\n\n[PRODUCT CONTEXT FOR FOLLOW-UP QUESTIONS]:\n"
                for i, product in enumerate(retrieved_products[:8], 1):  # Include all 8 products
                    try:
                        product_name = product.get('Product Name', product.get('product_name', 'Unknown Product'))
                        brand = product.get('Brand Name', product.get('brand_name', ''))
                        category = product.get('Category', product.get('category', ''))
                        about_product = product.get('About Product', product.get('about_product', ''))
                        list_price = product.get('List Price', '')
                        selling_price = product.get('Selling Price', '')
                        specifications = product.get('Product Specification', product.get('product_specification', ''))
                        
                        # Clean up brand and category
                        if pd.isna(brand) or str(brand).lower() in ['nan', '', 'none', 'null']:
                            brand = 'Brand not specified'
                        if pd.isna(category) or str(category).lower() in ['nan', '', 'none', 'null']:
                            category = 'Category not specified'
                        
                        # Format pricing info
                        price_info = "Price not available"
                        if not pd.isna(selling_price) and str(selling_price) not in ['nan', '', 'none', 'null']:
                            price_info = f"Selling Price: {selling_price}"
                            if not pd.isna(list_price) and str(list_price) not in ['nan', '', 'none', 'null'] and str(list_price) != str(selling_price):
                                price_info += f", List Price: {list_price}"
                        
                        # Add product details
                        product_context += f"\nProduct {i}: {product_name}\n"
                        product_context += f"- Brand: {brand}\n"
                        product_context += f"- Category: {category}\n"
                        product_context += f"- {price_info}\n"
                        
                        if not pd.isna(about_product) and str(about_product) not in ['nan', '', 'none', 'null']:
                            description = str(about_product).strip()
                            if len(description) > 200:
                                description = description[:200] + "..."
                            product_context += f"- Description: {description}\n"
                        
                        if not pd.isna(specifications) and str(specifications) not in ['nan', '', 'none', 'null']:
                            specs = str(specifications).strip()
                            if len(specs) > 200:
                                specs = specs[:200] + "..."
                            product_context += f"- Specifications: {specs}\n"
                            
                    except Exception as e:
                        logger.error(f"Error adding product {i} to context: {e}")
                
                enriched_response += product_context
            
            # Store the enriched response for LLM context but display only the clean response
            history.append({"role": "assistant", "content": enriched_response})
            
            # Store product context separately for follow-up questions
            if retrieved_products:
                cl.user_session.set("current_products", retrieved_products)
                cl.user_session.set("current_metrics", query_metrics)
        
        # Update session history
        cl.user_session.set("history", history)
    
    # Clean up uploaded file
    if uploaded_image_path and os.path.exists(uploaded_image_path):
        try:
            os.remove(uploaded_image_path)
        except Exception as e:
            logger.error(f"Error removing temp file: {e}")

@cl.on_chat_end
async def end_chat():
    """Clean up when chat ends and display session summary"""
    session_metrics = cl.user_session.get("session_metrics", {})
    
    if session_metrics.get('queries_count', 0) > 0:
        # Display session summary
        summary = f"""
**📊 Session Summary:**
- Total queries: {session_metrics.get('queries_count', 0)}
- Average response time: {session_metrics.get('avg_response_time', 0):.2f}s
- Search modes used: {', '.join(set(session_metrics.get('search_modes_used', [])))}
- Session duration: {datetime.now().isoformat()}

Thank you for using our AI shopping assistant! 🛍️
"""
        await cl.Message(content=summary).send()
    
    cleanup_temp_files()
    logger.info("Chat session ended")

# --- Evaluation Dashboard Functions ---

async def show_evaluation_dashboard():
    """Display a proper analytics dashboard for e-commerce RAG system"""
    
    if not analytics_tracker:
        await cl.Message(
            content="❌ **Error**: Analytics tracker not initialized."
        ).send()
        return
    
    # Get analytics data
    summary = analytics_tracker.get_analytics_summary()
    trends = analytics_tracker.get_performance_trends()
    recent_queries = analytics_tracker.get_recent_queries(5)
    
    # Calculate search mode percentages
    search_modes = summary.get("search_modes", {"text": 0, "image": 0, "multimodal": 0})
    total_searches = sum(search_modes.values())
    
    if total_searches > 0:
        text_pct = (search_modes.get("text", 0) / total_searches) * 100
        image_pct = (search_modes.get("image", 0) / total_searches) * 100
        multimodal_pct = (search_modes.get("multimodal", 0) / total_searches) * 100
    else:
        text_pct = image_pct = multimodal_pct = 0
    
    # Create dashboard content with real data
    dashboard_content = f"""
# 📊 E-commerce RAG Analytics Dashboard

## 🎯 System Performance Overview

### Real-Time Metrics
- **Total Queries Processed**: {summary.get('total_queries', 0):,}
- **Average Response Time**: {summary.get('avg_response_time', 0):.2f}s
- **Products Database**: 9,955 items
- **Search Success Rate**: {(summary.get('avg_similarity_score', 0) * 100):.1f}%

### Search Analytics
- **Average Products Retrieved**: {summary.get('avg_products_retrieved', 0):.1f}
- **Query Types Distribution**: 
  - Text Queries: {text_pct:.1f}%
  - Image Queries: {image_pct:.1f}%
  - Multimodal Queries: {multimodal_pct:.1f}%

### Performance Metrics
- **Embedding Generation**: {trends.get('recent_avg_response_time', 0):.3f}s avg
- **Vector Search**: N/A
- **LLM Response**: N/A
- **Total Pipeline**: {summary.get('avg_response_time', 0):.2f}s avg

### Quality Indicators
- **Average Similarity Scores**: {summary.get('avg_similarity_score', 0):.3f}
- **Recent Performance**: {trends.get('recent_avg_similarity', 0):.3f}
- **User Engagement**: 
  - Queries Today: {trends.get('queries_today', 0)}
  - Queries Last Hour: {trends.get('queries_last_hour', 0)}

## 📈 Trends & Insights

### Search Patterns
- **Total Sessions**: {summary.get('total_queries', 0)}
- **Recent Avg Response Time**: {trends.get('recent_avg_response_time', 0):.2f}s
- **Recent Avg Similarity**: {trends.get('recent_avg_similarity', 0):.3f}

### System Health
- **Error Rate**: 0.0% (No errors tracked)
- **Cache Hit Rate**: Not implemented
- **Memory Usage**: Normal
- **API Response Times**: Normal

---

## 💡 How to Use This Dashboard

1. **Switch to Shopping Assistant** to generate analytics data
2. **Ask product questions** to populate metrics
3. **Return here** to view updated analytics
4. **Monitor trends** over time for system optimization

**Note**: This dashboard updates automatically as users interact with the shopping assistant.
"""
    
    await cl.Message(content=dashboard_content).send()
    
    # Show recent queries if available
    if recent_queries:
        recent_content = "## 📝 Recent Queries\n\n"
        for i, query in enumerate(recent_queries, 1):
            query_text = query.get("query", "Unknown")[:50]
            response_time = query.get("response_time", 0)
            similarity = query.get("avg_similarity", 0)
            num_products = query.get("num_products", 0)
            search_mode = query.get("search_mode", "text")
            
            recent_content += f"**{i}.** \"{query_text}{'...' if len(query.get('query', '')) > 50 else ''}\"\n"
            recent_content += f"- Response: {response_time:.2f}s | Similarity: {similarity:.3f} | Products: {num_products} | Mode: {search_mode}\n\n"
        
        await cl.Message(content=recent_content).send()
    
    # Add a note about the read-only nature
    note_content = """
## ⚠️ Dashboard Information

This is a **read-only analytics dashboard**. The chat input below is disabled for this view.

To generate analytics data:
1. Switch to the **Shopping Assistant** tab
2. Ask product questions and search queries
3. Return here to see updated metrics

**Available Commands**: Type `refresh` to update the dashboard.
"""
    await cl.Message(content=note_content).send()

# --- Additional Features ---

@cl.action_callback("get_recommendations")
async def get_product_recommendations(action):
    """Get recommendations for a specific product"""
    if not rag_system:
        await cl.Message(content="Product recommendation system is not available.").send()
        return
        
    try:
        product_id = int(action.value)
        recommendations = rag_system.get_product_recommendations(product_id, top_k=3)
        
        if recommendations:
            rec_text = "**🎯 You might also like:**\n\n"
            for i, rec in enumerate(recommendations, 1):
                rec_text += f"**{i}. {rec['product_name']}**\n"
                if rec.get('product_specification'):
                    spec = rec['product_specification']
                    if len(spec) > 150:
                        spec = spec[:150] + "..."
                    rec_text += f"*{spec}*\n"
                rec_text += f"*Similarity:* {rec['similarity_score']:.1%}\n\n"
                
            await display_product_results(recommendations, rec_text)
        else:
            await cl.Message(content="No similar products found.").send()

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        await cl.Message(content="Error getting product recommendations.").send()

@cl.action_callback("run_evaluation")
async def run_system_evaluation(action):
    """Run comprehensive system evaluation"""
    if not evaluator:
        await cl.Message(content="Evaluation system is not available.").send()
        return
    
    try:
        await cl.Message(content="🔄 Running comprehensive system evaluation... This may take a few minutes.").send()
        
        # Run evaluation
        results = await evaluator.run_comprehensive_evaluation()
        
        if results:
            # Generate and display report
            report = evaluator.generate_evaluation_report(results)
            await cl.Message(content=f"```\n{report}\n```").send()
            
            # Display key metrics
            metrics_summary = f"""
**🎯 Key Performance Indicators:**

**Overall Score:** {results.get('overall_score', 0):.1f}/10

**Retrieval Performance:**
- Precision@5: {results.get('retrieval', {}).get('precision_at_k', {}).get(5, 0):.1%}
- NDCG@5: {results.get('retrieval', {}).get('ndcg_at_k', {}).get(5, 0):.3f}

**Response Quality:**
- Average Relevance: {results.get('response_quality', {}).get('relevance_scores', 0):.1f}/10
- Average Helpfulness: {results.get('response_quality', {}).get('helpfulness_scores', 0):.1f}/10

**Performance:**
- Average Response Time: {results.get('performance', {}).get('avg_response_time', 0):.2f}s
- Throughput: {results.get('performance', {}).get('throughput_qps', 0):.1f} queries/sec
"""
            await cl.Message(content=metrics_summary).send()
        else:
            await cl.Message(content="❌ Evaluation failed. Please check the logs.").send()

    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        await cl.Message(content=f"Error running evaluation: {str(e)}").send()

# --- Error Handling ---

@cl.on_stop
async def on_stop():
    """Handle stop events"""
    cleanup_temp_files()

def is_followup_question(user_query: str, chat_history: list) -> bool:
    """
    Determine if the user query is a follow-up question about previously retrieved products.
    Uses both conversation context and semantic similarity.
    """
    # First check if there's conversation context
    if not chat_history or len(chat_history) < 2:
        return False
    
    # Check if the last assistant message mentioned products
    last_assistant_msg = None
    for msg in reversed(chat_history):
        if msg.get('role') == 'assistant':
            last_assistant_msg = msg.get('content', '')
            break
    
    # If no previous assistant message or it doesn't seem to contain product info, not a follow-up
    if not last_assistant_msg or not any(keyword in last_assistant_msg.lower() for keyword in 
                                        ['product', 'recommend', 'option', 'choice', 'consider']):
        return False
    
    # Use semantic similarity to classify the query type
    return _classify_query_intent(user_query)

async def handle_followup_question(user_query: str, chat_history: list) -> str:
    """
    Handle follow-up questions using OpenAI with conversation context,
    without triggering a new product search.
    """
    try:
        # Get recent conversation context (includes product context from previous responses)
        recent_context = chat_history[-6:] if len(chat_history) > 6 else chat_history
        
        # Create a focused prompt for follow-up questions
        followup_prompt = """You are an expert e-commerce assistant helping a customer make decisions about products they've already seen.

The customer is asking a follow-up question about products that were previously shown to them. 
Your job is to:
1. Answer their question based on the conversation context and product information
2. Look for [PRODUCT CONTEXT FOR FOLLOW-UP QUESTIONS] sections in the conversation history - these contain detailed product information including prices, specifications, descriptions, and brands
3. Use this product data to answer specific questions about prices, features, specifications, comparisons, etc.
4. Provide helpful comparisons and recommendations
5. Be conversational and helpful
6. DO NOT search for new products - work with what's already been discussed

CRITICAL INSTRUCTION:
- DO NOT include or repeat the [PRODUCT CONTEXT FOR FOLLOW-UP QUESTIONS] section in your response
- Only provide a clean, conversational answer to the customer's question
- Reference the product information naturally in your response without showing the raw data structure

The conversation history includes detailed product information that you should reference to answer questions about prices, specifications, features, and other product details.
Be concise but helpful in your response."""

        messages = [
            {"role": "system", "content": followup_prompt}
        ] + recent_context + [
            {"role": "user", "content": user_query}
        ]
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error handling follow-up question: {e}")
        return "I'd be happy to help you decide! Could you be more specific about what you'd like to know about the products I showed you?"

if __name__ == "__main__":
    # Initialize RAG system on startup
    initialize_rag_system()