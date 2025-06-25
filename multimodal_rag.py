import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import base64
import logging
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

class MultimodalEcommerceRAG:
    def __init__(self, openai_api_key: str, data_dir: str = "data"):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.data_dir = Path(data_dir)
        self.embeddings_dir = self.data_dir / "embeddings"
        
        # Initialize CLIP model for query processing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        logger.info("Loading CLIP model for query processing...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP model loaded successfully")
        
        # Load processed data
        self.products_df = None
        self.image_embeddings_matrix = None
        self.text_embeddings_matrix = None
        self.multimodal_embeddings_matrix = None
        self.product_ids = None
        
        self.load_data()
        
    def load_data(self):
        """Load processed products and CLIP embeddings"""
        try:
            # Load products dataframe
            products_path = self.data_dir / "processed_products.csv"
            if products_path.exists():
                self.products_df = pd.read_csv(products_path)
                logger.info(f"Loaded {len(self.products_df)} products")
            else:
                logger.warning("Processed products file not found. Please run data processing first.")
                return
                
            # Try to load numpy embeddings first (faster)
            try:
                logger.info("Attempting to load numpy embeddings...")
                self.image_embeddings_matrix = np.load(self.embeddings_dir / "image_embeddings.npy")
                self.text_embeddings_matrix = np.load(self.embeddings_dir / "text_embeddings.npy")
                self.multimodal_embeddings_matrix = np.load(self.embeddings_dir / "multimodal_embeddings.npy")
                self.product_ids = self.products_df['product_id'].tolist()
                
                logger.info(f"Successfully loaded numpy embeddings:")
                logger.info(f"  - Image embeddings shape: {self.image_embeddings_matrix.shape}")
                logger.info(f"  - Text embeddings shape: {self.text_embeddings_matrix.shape}")
                logger.info(f"  - Multimodal embeddings shape: {self.multimodal_embeddings_matrix.shape}")
                logger.info(f"  - Product IDs count: {len(self.product_ids)}")
                
            except FileNotFoundError as e:
                logger.warning(f"Numpy embeddings not found: {e}")
                # Fallback to JSON embeddings
                embeddings_path = self.embeddings_dir / "clip_embeddings.json"
                if embeddings_path.exists():
                    logger.info("Loading JSON embeddings as fallback...")
                    with open(embeddings_path, 'r') as f:
                        embeddings_data = json.load(f)
                        
                    self.product_ids = embeddings_data['product_ids']
                    self.image_embeddings_matrix = np.array(embeddings_data['image_embeddings'])
                    self.text_embeddings_matrix = np.array(embeddings_data['text_embeddings'])
                    self.multimodal_embeddings_matrix = np.array(embeddings_data['multimodal_embeddings'])
                    logger.info(f"Loaded CLIP embeddings for {len(self.product_ids)} products from JSON")
                else:
                    logger.warning("CLIP embeddings file not found. Please run data processing first.")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def get_clip_query_embedding(self, text_query: str = "", image_path: str = "") -> np.ndarray:
        """Get CLIP embedding for query (text and/or image) with improved e-commerce text processing"""
        try:
            # Prepare inputs
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
            else:
                # Create a dummy white image if no image provided
                image = Image.new("RGB", (224, 224), color="white")
                
            # Enhanced text preprocessing for e-commerce search
            if text_query:
                # Clean and enhance the query for better e-commerce matching
                processed_query = self._preprocess_search_query(text_query)
            else:
                processed_query = ""
                
            # Process with CLIP
            inputs = self.clip_processor(
                text=[processed_query], 
                images=image, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77  # CLIP's max sequence length
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                
                if processed_query and image_path:
                    # Multimodal query: combine text and image embeddings
                    text_emb = outputs.text_embeds.cpu().numpy()[0]
                    image_emb = outputs.image_embeds.cpu().numpy()[0]
                    combined_emb = (text_emb + image_emb) / 2
                    return combined_emb
                elif processed_query:
                    # Text-only query
                    return outputs.text_embeds.cpu().numpy()[0]
                elif image_path:
                    # Image-only query
                    return outputs.image_embeds.cpu().numpy()[0]
                else:
                    return np.array([])
                    
        except Exception as e:
            logger.error(f"Error getting CLIP query embedding: {e}")
            return np.array([])
    
    def _preprocess_search_query(self, query: str) -> str:
        """
        Preprocess search queries to work better with CLIP embeddings for e-commerce.
        Uses smart automatic enhancement based on the actual product database.
        """
        if not query:
            return ""
        
        # Convert to lowercase for processing
        processed = query.lower().strip()
        
        # Smart query enhancement using actual product data
        enhanced_query = self._enhance_query_with_product_context(processed)
        
        # Clean up and limit length for CLIP
        enhanced_query = ' '.join(enhanced_query.split())  # Remove extra spaces
        
        # Limit to CLIP's token limit (approximately 77 tokens)
        if len(enhanced_query.split()) > 60:
            enhanced_query = ' '.join(enhanced_query.split()[:60])
        
        logger.debug(f"Query preprocessing: '{query}' -> '{enhanced_query}'")
        return enhanced_query
    
    def _enhance_query_with_product_context(self, query: str) -> str:
        """
        Enhance queries by analyzing actual product data to find related terms.
        This is dynamic and learns from the actual database content.
        """
        if self.products_df is None or len(query.split()) > 5:
            # Don't enhance long queries or if no product data
            return query
        
        try:
            # Extract key terms from the query
            query_words = set(query.lower().split())
            
            # First try exact matches in product names
            exact_matches = []
            for _, product in self.products_df.iterrows():
                product_name = str(product.get('Product Name', '')).lower()
                if any(word in product_name for word in query_words):
                    exact_matches.append(product_name)
            
            # If we have exact matches, use those for enhancement
            if exact_matches:
                all_words = []
                for product_name in exact_matches[:10]:  # Limit for performance
                    words = product_name.split()
                    all_words.extend(words)
                
                # Find frequently occurring words that aren't in the original query
                from collections import Counter
                word_counts = Counter(all_words)
                
                # Get relevant enhancement terms
                enhancement_terms = []
                for word, count in word_counts.most_common(5):
                    if (len(word) > 2 and 
                        word not in query_words and 
                        word not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'make', 'sure', 'fits'] and
                        count >= 2):  # Must appear in at least 2 products
                        enhancement_terms.append(word)
                
                # Add the most relevant enhancement terms
                if enhancement_terms:
                    enhanced = query + " " + " ".join(enhancement_terms[:2])
                    return enhanced
            
            # If no exact matches, try broader category matching
            matching_products = []
            for _, product in self.products_df.iterrows():
                product_text = ""
                
                # Combine relevant product fields for analysis
                for field in ['Product Name', 'Category', 'About Product']:
                    field_value = product.get(field, '')
                    if pd.notna(field_value) and str(field_value) != 'nan':
                        product_text += " " + str(field_value).lower()
                
                # Check if any query words appear in this product
                if any(word in product_text for word in query_words):
                    matching_products.append(product_text)
            
            if not matching_products:
                return query
            
            # Extract common terms from matching products to enhance the query
            all_words = []
            for product_text in matching_products[:15]:  # Limit for performance
                words = product_text.split()
                all_words.extend(words)
            
            # Find frequently occurring words that aren't in the original query
            from collections import Counter
            word_counts = Counter(all_words)
            
            # Get relevant enhancement terms
            enhancement_terms = []
            for word, count in word_counts.most_common(8):
                if (len(word) > 2 and 
                    word not in query_words and 
                    word not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'make', 'sure', 'fits', 'entering', 'model', 'number'] and
                    count >= 2):  # Must appear in at least 2 products
                    enhancement_terms.append(word)
            
            # Add the most relevant enhancement terms
            if enhancement_terms:
                enhanced = query + " " + " ".join(enhancement_terms[:2])
                return enhanced
            
        except Exception as e:
            logger.debug(f"Error in query enhancement: {e}")
        
        return query
            
    def analyze_uploaded_image(self, image_path: str) -> str:
        """Analyze uploaded image and extract product information using OpenAI Vision"""
        try:
            base64_image = self.encode_image_base64(image_path)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this product image and provide:
1. Product type/category
2. Key features and characteristics
3. Colors and design elements
4. Any visible brand or text
5. Suggested search terms for finding similar products

Format your response as a detailed description that could be used for product search."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return ""
            
    def retrieve_similar_products(self, query_embedding: np.ndarray, search_mode: str = "multimodal", top_k: int = 3) -> List[Dict]:
        """Retrieve most similar products based on query embedding"""
        if len(query_embedding) == 0:
            return []
            
        try:
            # Debug logging
            logger.info(f"Retrieve called with search_mode: {search_mode}")
            logger.info(f"Available matrices: image={self.image_embeddings_matrix is not None}, text={self.text_embeddings_matrix is not None}, multimodal={self.multimodal_embeddings_matrix is not None}")
            
            # Choose embedding matrix based on search mode
            if search_mode == "image" and self.image_embeddings_matrix is not None:
                embeddings_matrix = self.image_embeddings_matrix
            elif search_mode == "text" and self.text_embeddings_matrix is not None:
                embeddings_matrix = self.text_embeddings_matrix
            elif search_mode == "multimodal" and self.multimodal_embeddings_matrix is not None:
                embeddings_matrix = self.multimodal_embeddings_matrix
            else:
                logger.warning(f"Embeddings matrix not available for mode: {search_mode}")
                return []
            
            # Calculate cosine similarity
            similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
            
            # Handle potential NaN or infinite values
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Very low threshold for 10k product database - we want to find something!
            min_similarity_threshold = 0.01  # Even lower threshold
            
            # Get more candidates initially (10x the requested amount for better filtering)
            initial_candidates = min(top_k * 10, 100)  # Get up to 100 candidates
            top_indices = np.argsort(similarities)[::-1]
            
            results = []
            candidates_checked = 0
            
            for idx in top_indices:
                if len(results) >= top_k:
                    break
                    
                candidates_checked += 1
                if candidates_checked > initial_candidates * 3:  # Safety limit
                    break
                    
                similarity_score = similarities[idx]
                
                # Only filter out extremely low similarity scores
                if similarity_score < min_similarity_threshold:
                    continue
                
                product_id = self.product_ids[idx]
                
                # Get product details and add to results
                product_row = self.products_df[self.products_df['product_id'] == product_id].iloc[0]
                
                # More aggressive filtering for poor data quality
                product_name = product_row.get('Product Name', '')
                category = product_row.get('Category', '')
                about_product = product_row.get('About Product', '')
                
                # Skip products with very poor names
                if (pd.isna(product_name) or 
                    str(product_name).strip() in ['-', '', 'nan', 'None'] or
                    len(str(product_name).strip()) < 3):
                    logger.debug(f"Skipping product with poor name: '{product_name}'")
                    continue
                
                # Skip obviously irrelevant products based on category mismatch
                product_name_lower = str(product_name).lower()
                category_lower = str(category).lower() if pd.notna(category) else ""
                about_lower = str(about_product).lower() if pd.notna(about_product) else ""
                
                # Check for obvious mismatches (teacher workstation for toys, etc.)
                if any(bad_term in product_name_lower for bad_term in ['teacher', 'workstation', 'curriculum', 'educational software']):
                    # Only allow if it's actually relevant to the query
                    query_lower = str(query_embedding).lower() if hasattr(query_embedding, 'lower') else ""
                    if not any(term in category_lower + about_lower for term in ['toy', 'game', 'play']):
                        logger.debug(f"Skipping irrelevant product: '{product_name}'")
                        continue
                
                result = {
                    'product_id': product_id,
                    'similarity_score': float(similarity_score),
                    'product_name': product_name,
                    'product_specification': product_row.get('Product Specification', ''),
                    'image_path': product_row.get('image_path', ''),
                    'image_description': product_row.get('image_description', ''),
                    'combined_text': product_row.get('combined_text', ''),
                    'search_mode': search_mode,
                    # Add all original CSV columns with proper defaults
                    'Product Name': product_row.get('Product Name', 'Unknown Product'),
                    'Brand Name': product_row.get('Brand Name', ''),
                    'Category': product_row.get('Category', ''),
                    'About Product': product_row.get('About Product', ''),
                    'Product Description': product_row.get('Product Description', ''),
                    'Image': product_row.get('Image', ''),
                    'List Price': product_row.get('List Price', ''),
                    'Selling Price': product_row.get('Selling Price', ''),
                    'Uniq Id': product_row.get('Uniq Id', '')
                }
                results.append(result)
            
            # Log search statistics
            if results:
                avg_similarity = np.mean([r['similarity_score'] for r in results])
                logger.info(f"Retrieved {len(results)} products with avg similarity {avg_similarity:.3f} (threshold: {min_similarity_threshold})")
            else:
                max_similarity = np.max(similarities) if len(similarities) > 0 else 0
                logger.warning(f"No products found above threshold {min_similarity_threshold}. Max similarity was {max_similarity:.3f}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving similar products: {e}")
            return []
            
    def format_product_context(self, products: List[Dict]) -> str:
        """Format retrieved products for LLM context"""
        if not products:
            return "No relevant products found."
            
        context = "Here are the most relevant products from our catalog:\n\n"
        
        for i, product in enumerate(products, 1):
            context += f"Product {i}:\n"
            context += f"Name: {product['product_name']}\n"
            
            if product['product_specification']:
                context += f"Specifications: {product['product_specification']}\n"
                
            if product['image_description']:
                context += f"Visual Description: {product['image_description']}\n"
                
            context += f"Relevance Score: {product['similarity_score']:.3f}\n"
            context += f"Search Mode: {product['search_mode']}\n"
            context += "-" * 50 + "\n\n"
            
        return context
        
    def generate_response(self, user_query: str, retrieved_products: List[Dict], 
                         chat_history: List[Dict] = None) -> str:
        """Generate response using OpenAI LLM with retrieved context"""
        try:
            # Format context from retrieved products
            product_context = self.format_product_context(retrieved_products)
            
            # Create system prompt
            system_prompt = """You are an expert AI shopping assistant for a multimodal e-commerce platform.

Your role is to help customers find products by providing natural, conversational recommendations based on the product database context provided to you.

CRITICAL RULES - NEVER VIOLATE THESE:
1. ONLY recommend products that are explicitly listed in the Product Context below
2. NEVER mention products that are not in the provided context
3. NEVER use your general knowledge about products - only use the provided data
4. NEVER make up product names, brands, or specifications
5. NEVER mention products from your training data or general knowledge
6. If ANY products are provided in the context, you MUST work with them and provide recommendations
7. NEVER say "couldn't find any products" if the Product Context contains products

RESPONSE GUIDELINES:
1. Write naturally and conversationally - like a knowledgeable friend helping with shopping
2. Synthesize information from the product context to create helpful recommendations
3. DO NOT repeat raw specifications or create bullet-point lists of product details
4. DO NOT mention similarity scores, search modes, or technical details
5. Focus on what makes products appealing and suitable for the customer's needs
6. Compare products naturally when multiple options are relevant
7. Suggest specific products by name ONLY if they appear in the Product Context
8. Keep responses concise but informative (2-3 paragraphs maximum)
9. End with a helpful question or suggestion for next steps

MANDATORY APPROACH:
- If you have ANY products in the context (even 1), you MUST provide helpful recommendations
- Even if products aren't perfect matches, work with what's available: "While these might not be exactly what you're looking for, here are some interesting options I found..."
- Focus on the most relevant products from what's available
- Always be helpful and constructive rather than dismissive
- Look for any connection between the query and available products

ONLY say "I couldn't find any products matching your request" if the Product Context is completely empty (contains zero products).

Product Context:
{context}"""

            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt.format(context=product_context)}
            ]
            
            # Add chat history if provided
            if chat_history:
                # Add last few messages for context (limit to avoid token limits)
                recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
                for msg in recent_history:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            
            # Add current user query
            messages.append({"role": "user", "content": user_query})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
            
    def get_product_image_for_display(self, product_id: int) -> Optional[str]:
        """Get product image path for display in UI"""
        try:
            product_row = self.products_df[self.products_df['product_id'] == product_id]
            if not product_row.empty:
                image_path = product_row.iloc[0].get('image_path', '')
                if image_path and os.path.exists(image_path):
                    return image_path
        except Exception as e:
            logger.error(f"Error getting product image: {e}")
        return None
        
    async def process_query(self, text_query: str = "", image_path: str = "", 
                          chat_history: List[Dict] = None, top_k: int = 5) -> Dict[str, Any]:
        """Main method to process user query and return response with context"""
        try:
            # Determine search mode and create query embedding
            if text_query and image_path:
                search_mode = "multimodal"
                query_embedding = self.get_clip_query_embedding(text_query, image_path)
            elif image_path:
                search_mode = "image"
                query_embedding = self.get_clip_query_embedding("", image_path)
            elif text_query:
                search_mode = "multimodal"  # Use multimodal for text queries too
                query_embedding = self.get_clip_query_embedding(text_query, "")
            else:
                return {
                    'response': "I need either a text query or an image to help you find products.",
                    'retrieved_products': [],
                    'query_analysis': "",
                    'search_mode': "none"
                }
            
            if len(query_embedding) == 0:
                return {
                    'response': "I couldn't process your query. Please try again with a different text or image.",
                    'retrieved_products': [],
                    'query_analysis': "",
                    'search_mode': search_mode
                }
            
            # Retrieve similar products
            retrieved_products = self.retrieve_similar_products(query_embedding, search_mode, top_k)
            
            # Generate response
            full_query = text_query
            image_analysis = ""
            if image_path and os.path.exists(image_path):
                image_analysis = self.analyze_uploaded_image(image_path)
                full_query += f"\n[User uploaded an image: {image_analysis}]"
                
            response = self.generate_response(full_query, retrieved_products, chat_history)
            
            return {
                'response': response,
                'retrieved_products': retrieved_products,
                'query_analysis': image_analysis,
                'search_mode': search_mode,
                'query_embedding_created': True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your request. Please try again.",
                'retrieved_products': [],
                'query_analysis': "",
                'search_mode': "error",
                'error': str(e)
            }
            
    def get_product_recommendations(self, product_id: int, top_k: int = 3) -> List[Dict]:
        """Get recommendations based on a specific product using multimodal similarity"""
        try:
            # Find the product's embedding
            if product_id not in self.product_ids:
                return []
                
            product_idx = self.product_ids.index(product_id)
            
            # Use multimodal embeddings for recommendations
            if self.multimodal_embeddings_matrix is None:
                return []
                
            product_embedding = self.multimodal_embeddings_matrix[product_idx]
            
            # Find similar products (excluding the original)
            similarities = cosine_similarity([product_embedding], self.multimodal_embeddings_matrix)[0]
            
            # Get top-k most similar products (excluding self)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip first (self)
            
            recommendations = []
            for idx in top_indices:
                rec_product_id = self.product_ids[idx]
                similarity_score = similarities[idx]
                
                product_row = self.products_df[self.products_df['product_id'] == rec_product_id].iloc[0]
                
                recommendation = {
                    'product_id': rec_product_id,
                    'similarity_score': float(similarity_score),
                    'product_name': product_row.get('Product Name', ''),
                    'product_specification': product_row.get('Product Specification', ''),
                    'image_path': product_row.get('image_path', ''),
                    'image_description': product_row.get('image_description', ''),
                    'search_mode': 'multimodal'
                }
                recommendations.append(recommendation)
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
        
    # Initialize RAG system
    rag = MultimodalEcommerceRAG(openai_api_key)
    
    # Test text query
    async def test_rag():
        result = await rag.process_query(
            text_query="I'm looking for wireless headphones with noise cancellation",
            top_k=3
        )
        
        print("Response:", result['response'])
        print(f"\nSearch Mode: {result['search_mode']}")
        print("\nRetrieved Products:")
        for product in result['retrieved_products']:
            print(f"- {product['product_name']} (Score: {product['similarity_score']:.3f})")
            
    asyncio.run(test_rag()) 