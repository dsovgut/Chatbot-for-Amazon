import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import json
import time
from typing import List, Dict, Optional, Tuple
import asyncio
import aiohttp
import aiofiles
from openai import OpenAI
import base64
from tqdm import tqdm
import logging
from pathlib import Path
import hashlib
import torch
from transformers import CLIPProcessor, CLIPModel
import re
import ssl
import certifi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonDataProcessor:
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.dataset_path = Path("data/amazon_products.csv")
        self.images_dir = Path("data/images")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create embeddings directory
        self.embeddings_dir = Path("data/embeddings")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup SSL context for macOS
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
    def download_dataset(self):
        """Download the Amazon Product Dataset 2020 from Kaggle"""
        try:
            import kagglehub
            logger.info("Downloading Amazon Product Dataset 2020...")
            path = kagglehub.dataset_download("promptcloud/amazon-product-dataset-2020")
            
            # Find the CSV file in the downloaded path
            csv_files = list(Path(path).glob("*.csv"))
            if csv_files:
                source_file = csv_files[0]
                # Copy to our data directory
                import shutil
                shutil.copy2(source_file, self.dataset_path)
                logger.info(f"Dataset downloaded and saved to {self.dataset_path}")
            else:
                logger.error("No CSV file found in downloaded dataset")
                
        except ImportError:
            logger.error("kagglehub not installed. Please install with: pip install kagglehub")
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            
    def load_and_clean_dataset(self, max_products: int = None) -> pd.DataFrame:
        """Load and clean the Amazon dataset"""
        if not self.dataset_path.exists():
            self.download_dataset()
            
        if not self.dataset_path.exists():
            logger.error("Dataset file not found. Please download manually.")
            return pd.DataFrame()
            
        logger.info("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        
        # Select all relevant columns for intelligent processing
        desired_columns = [
            'Uniq Id', 'Product Name', 'Brand Name', 'Category', 
            'About Product', 'Product Specification', 'Product Description',
            'Image', 'List Price', 'Selling Price'
        ]
        
        # Use only columns that exist in the dataset
        available_columns = [col for col in desired_columns if col in df.columns]
        
        if 'Product Name' not in available_columns or 'Image' not in available_columns:
            logger.error("Essential columns (Product Name, Image) not found in dataset")
            return pd.DataFrame()
            
        df = df[available_columns].copy()
        
        # Clean data
        df = df.dropna(subset=['Product Name'])
        df = df.dropna(subset=['Image'])
        
        # Remove duplicates based on Uniq Id only (since that should be unique)
        # Don't remove based on Product Name or Image as many legitimate products may share these
        df = df.drop_duplicates(subset=['Uniq Id'])
        
        # Limit dataset size for processing if specified
        if max_products:
            df = df.head(max_products)
            logger.info(f"Limited to {max_products} products for processing")
        
        # Reset index and ensure we have Uniq Id
        df = df.reset_index(drop=True)
        if 'Uniq Id' not in df.columns:
            df['Uniq Id'] = [f"product_{i}" for i in range(len(df))]
        
        logger.info(f"Loaded {len(df)} products with {len(available_columns)} columns")
        logger.info(f"Available columns: {available_columns}")
        return df
        
    async def validate_and_download_image(self, session: aiohttp.ClientSession, 
                                        url: str, product_id: str) -> Optional[str]:
        """Validate and download a single image with improved error handling"""
        try:
            # Create filename based on product_id and URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            # Clean product_id for filename (remove special characters)
            clean_id = re.sub(r'[^\w\-_]', '_', str(product_id))
            filename = f"{clean_id}_{url_hash}.jpg"
            filepath = self.images_dir / filename
            
            # Skip if already downloaded
            if filepath.exists():
                return str(filepath)
            
            # Clean URL - handle URL encoding issues
            clean_url = url.replace('%2B', '+').replace('%2b', '+')
            
            # Set headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
                
            async with session.get(clean_url, headers=headers, timeout=20) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    # Validate image
                    try:
                        img = Image.open(BytesIO(content))
                        img = img.convert("RGB")
                        
                        # Check if image is too small (likely placeholder)
                        if img.size[0] < 50 or img.size[1] < 50:
                            logger.warning(f"Image too small for {url}: {img.size}")
                            return None
                        
                        # Resize if too large (CLIP works best with 224x224)
                        if img.size[0] > 512 or img.size[1] > 512:
                            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                            
                        # Save image
                        img.save(filepath, "JPEG", quality=90)
                        return str(filepath)
                        
                    except Exception as e:
                        logger.warning(f"Invalid image format for {url}: {e}")
                        return None
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading {url}")
            return None
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None
    
    def download_image_sync(self, url: str, product_id: str) -> Optional[str]:
        """Synchronous fallback for image download using requests"""
        try:
            # Create filename based on product_id and URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            clean_id = re.sub(r'[^\w\-_]', '_', str(product_id))
            filename = f"{clean_id}_{url_hash}.jpg"
            filepath = self.images_dir / filename
            
            # Skip if already downloaded
            if filepath.exists():
                return str(filepath)
            
            # Clean URL
            clean_url = url.replace('%2B', '+').replace('%2b', '+')
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            }
            
            response = requests.get(clean_url, headers=headers, timeout=15, verify=False)
            
            if response.status_code == 200:
                try:
                    img = Image.open(BytesIO(response.content))
                    img = img.convert("RGB")
                    
                    # Check if image is too small
                    if img.size[0] < 50 or img.size[1] < 50:
                        return None
                    
                    # Resize if too large
                    if img.size[0] > 512 or img.size[1] > 512:
                        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        
                    # Save image
                    img.save(filepath, "JPEG", quality=90)
                    return str(filepath)
                    
                except Exception as e:
                    logger.warning(f"Invalid image format for {url}: {e}")
                    return None
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Sync download failed for {url}: {e}")
            return None
            
    async def download_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Download all valid images with improved SSL handling"""
        logger.info("Checking existing images and downloading missing ones...")
        
        # First pass: check which images already exist
        existing_images = 0
        missing_images = []
        image_paths = []
        
        for idx, row in df.iterrows():
            # Handle multiple image URLs separated by |
            image_urls = str(row['Image']).split('|')
            primary_image_url = image_urls[0].strip()  # Use first image
            
            # Generate expected filename
            url_hash = hashlib.md5(primary_image_url.encode()).hexdigest()[:8]
            clean_id = re.sub(r'[^\w\-_]', '_', str(row['Uniq Id']))
            filename = f"{clean_id}_{url_hash}.jpg"
            filepath = self.images_dir / filename
            
            if filepath.exists():
                existing_images += 1
                image_paths.append(str(filepath))
            else:
                missing_images.append((idx, primary_image_url, row['Uniq Id']))
                image_paths.append(None)  # Placeholder
        
        logger.info(f"Found {existing_images} existing images, need to download {len(missing_images)} missing images")
        
        # Only download missing images if there are any
        if missing_images:
            # Create SSL context that works on macOS
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(
                limit=10,  # Reduced concurrent connections
                ssl=ssl_context,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            try:
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    tasks = []
                    for idx, url, product_id in missing_images:
                        task = self.validate_and_download_image(session, url, product_id)
                        tasks.append((task, idx, url, product_id))
                        
                    # Process in smaller batches to avoid overwhelming
                    batch_size = 20
                    total_processed = 0
                    
                    # Create progress bar for missing images only
                    progress_bar = tqdm(total=len(tasks), desc=f"Downloading missing images")
                    
                    for i in range(0, len(tasks), batch_size):
                        batch = tasks[i:i+batch_size]
                        batch_tasks = [task[0] for task in batch]
                        
                        try:
                            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                            
                            for j, result in enumerate(batch_results):
                                original_idx = batch[j][1]
                                if isinstance(result, Exception):
                                    logger.warning(f"Async download failed for {batch[j][2]}: {result}")
                                    # Try synchronous fallback
                                    sync_result = self.download_image_sync(batch[j][2], batch[j][3])
                                    image_paths[original_idx] = sync_result
                                else:
                                    image_paths[original_idx] = result
                                
                                total_processed += 1
                                progress_bar.update(1)
                                progress_bar.set_description(f"Downloaded {total_processed}/{len(tasks)} missing images")
                                    
                        except Exception as e:
                            logger.warning(f"Batch download failed: {e}")
                            # Fallback to sync downloads for this batch
                            for task_info in batch:
                                idx, url, product_id = task_info[1], task_info[2], task_info[3]
                                sync_result = self.download_image_sync(url, product_id)
                                image_paths[idx] = sync_result
                                total_processed += 1
                                progress_bar.update(1)
                                progress_bar.set_description(f"Downloaded {total_processed}/{len(tasks)} missing images")
                        
                        # Small delay between batches
                        await asyncio.sleep(1)
                    
                    progress_bar.close()
                    
            except Exception as e:
                logger.error(f"Session creation failed: {e}")
                # Fallback to all sync downloads for missing images
                logger.info("Falling back to synchronous downloads...")
                for idx, url, product_id in tqdm(missing_images, desc="Sync downloading missing"):
                    result = self.download_image_sync(url, product_id)
                    image_paths[idx] = result
        
        # Add image paths to dataframe
        df['image_path'] = image_paths
        
        # Filter out failed downloads
        successful_downloads = df['image_path'].notna() & (df['image_path'] != '') & (df['image_path'] != None)
        df = df[successful_downloads].copy()
        
        total_existing = existing_images
        total_downloaded = len([p for p in image_paths if p is not None]) - existing_images
        logger.info(f"Using {total_existing} existing images and downloaded {total_downloaded} new images")
        logger.info(f"Total valid images: {len(df)}")
        return df
        
    def get_image_description_openai(self, image_path: str) -> str:
        """Get image description using OpenAI Vision API for enhanced text"""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this product image in detail, focusing on key features, colors, design, materials, brand elements, and any visible text. Be concise but comprehensive for e-commerce search."
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
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting image description: {e}")
            return ""
            
    def get_clip_embeddings(self, image_path: str, text: str, model, processor, device) -> Tuple[np.ndarray, np.ndarray]:
        """Get CLIP embeddings for image and text using provided model"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Process inputs
            inputs = processor(
                text=[text], 
                images=image, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77  # CLIP's max sequence length
            ).to(device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                image_embedding = outputs.image_embeds.cpu().numpy()[0]
                text_embedding = outputs.text_embeds.cpu().numpy()[0]
            
            return image_embedding, text_embedding
            
        except Exception as e:
            logger.error(f"Error getting CLIP embeddings for {image_path}: {e}")
            return np.array([]), np.array([])
    
    def get_text_embedding_only(self, text: str, model, processor, device) -> np.ndarray:
        """Get CLIP text embedding only"""
        try:
            # Process text input
            inputs = processor(
                text=[text], 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            
            # Get text embedding
            with torch.no_grad():
                text_embedding = model.get_text_features(**inputs).cpu().numpy()[0]
            
            return text_embedding
            
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            return np.array([])
            
    def _create_text_for_embedding(self, row) -> str:
        """
        Create text representation using actual product attributes intelligently.
        No artificial text creation - just use the real product information.
        """
        text_parts = []
        
        # 1. Product Name - always most important
        product_name = str(row.get('Product Name', '')).strip()
        if product_name and product_name != 'nan':
            text_parts.append(product_name)
        
        # 2. Brand Name - important for identification
        brand_name = str(row.get('Brand Name', '')).strip()
        if brand_name and brand_name != 'nan' and len(brand_name) > 1:
            text_parts.append(brand_name)
        
        # 3. Category - provides context (clean up the hierarchy)
        category = str(row.get('Category', '')).strip()
        if category and category != 'nan':
            # Clean category but keep the natural hierarchy
            category_clean = category.replace(' | ', ' ').replace('|', ' ')
            text_parts.append(category_clean)
        
        # 4. About Product - natural user-facing description
        about_product = str(row.get('About Product', '')).strip()
        if about_product and about_product != 'nan':
            # Clean but keep natural language
            about_clean = self._clean_product_text(about_product)
            if len(about_clean) > 20:  # Only if substantial
                text_parts.append(about_clean)
        
        # 5. Product Description - additional natural description
        product_desc = str(row.get('Product Description', '')).strip()
        if product_desc and product_desc != 'nan':
            desc_clean = self._clean_product_text(product_desc)
            # Only add if different from About Product and substantial
            if len(desc_clean) > 20 and desc_clean != about_product:
                text_parts.append(desc_clean)
        
        # 6. Color - important attribute for many products
        color = str(row.get('Color', '')).strip()
        if color and color != 'nan' and len(color) > 1:
            text_parts.append(color)
        
        # 7. Size - important for clothing, etc.
        size = str(row.get('Size Quantity Variant', '')).strip()
        if size and size != 'nan' and len(size) > 1:
            text_parts.append(size)
        
        # 8. Product Specification - technical details (limit length)
        product_spec = str(row.get('Product Specification', '')).strip()
        if product_spec and product_spec != 'nan':
            spec_clean = self._clean_product_text(product_spec)
            if len(spec_clean) > 20:
                # Limit to prevent overwhelming
                if len(spec_clean) > 200:
                    spec_clean = spec_clean[:200] + "..."
                text_parts.append(spec_clean)
        
        # Join with spaces - let CLIP understand the natural text
        final_text = ' '.join(text_parts)
        
        # Basic cleanup only
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        
        # Reasonable length limit for embeddings
        if len(final_text) > 800:
            final_text = final_text[:800] + "..."
        
        # Fallback
        if not final_text or len(final_text) < 5:
            final_text = product_name if product_name else "Unknown Product"
        
        return final_text
    
    def _clean_product_text(self, text: str) -> str:
        """
        Clean product text using e-commerce best practices.
        Removes common formatting issues, excessive punctuation, and noise.
        """
        if not text or text == 'nan':
            return ""
        
        # Remove HTML-like tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z0-9]+;', ' ', text)
        
        # Clean up common e-commerce formatting issues
        text = re.sub(r'\|+', ' | ', text)  # Normalize pipe separators
        text = re.sub(r'Make sure this fits by entering your model number\.?\s*\|?\s*', '', text)
        text = re.sub(r'show up to \d+ reviews by default', '', text)
        text = re.sub(r'View shipping rates and policies', '', text)
        text = re.sub(r'Learn More', '', text)
        text = re.sub(r'ASIN:\s*[A-Z0-9]+', '', text)
        text = re.sub(r'Item model number:\s*[^\|]+', '', text)
        text = re.sub(r'Manufacturer recommended age:\s*[^\|]+', '', text)
        text = re.sub(r'#\d+\s+in\s+[^\|]+', '', text)  # Remove ranking info
        
        # Clean up dimensions and weights (keep but normalize)
        text = re.sub(r'Product Dimensions:\s*', 'Dimensions: ', text)
        text = re.sub(r'Shipping Weight:\s*', 'Weight: ', text)
        text = re.sub(r'Item Weight:\s*', 'Weight: ', text)
        
        # Remove excessive punctuation and whitespace
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'\s*\|\s*\|\s*', ' | ', text)  # Clean up double pipes
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing pipes and whitespace
        text = re.sub(r'^\s*\|\s*|\s*\|\s*$', '', text)
        text = text.strip()
        
        return text
    
    def create_multimodal_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create CLIP embeddings for images and text with intelligent text processing"""
        logger.info("Creating CLIP embeddings for products...")
        
        # Initialize CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            return df
        
        embeddings_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating CLIP embeddings"):
            try:
                product_id = row['Uniq Id']
                
                # Use intelligent text processing
                text_for_embedding = self._create_text_for_embedding(row)
                
                # Get image path
                image_path = row.get('image_path', '')
                
                # Create embeddings
                if image_path and os.path.exists(image_path):
                    # Both image and text embeddings
                    img_embedding, text_embedding = self.get_clip_embeddings(
                        image_path, text_for_embedding, model, processor, device
                    )
                    
                    if len(img_embedding) > 0 and len(text_embedding) > 0:
                        # Create multimodal embedding (average of image and text)
                        multimodal_embedding = (img_embedding + text_embedding) / 2
                        
                        embeddings_data.append({
                            'product_id': product_id,
                            'text_for_embedding': text_for_embedding,
                            'image_embedding': img_embedding.tolist(),
                            'text_embedding': text_embedding.tolist(),
                            'multimodal_embedding': multimodal_embedding.tolist()
                        })
                    else:
                        logger.warning(f"Failed to create embeddings for product {product_id}")
                else:
                    # Text-only embedding
                    text_embedding = self.get_text_embedding_only(text_for_embedding, model, processor, device)
                    
                    if len(text_embedding) > 0:
                        # Use text embedding as multimodal embedding
                        embeddings_data.append({
                            'product_id': product_id,
                            'text_for_embedding': text_for_embedding,
                            'image_embedding': None,
                            'text_embedding': text_embedding.tolist(),
                            'multimodal_embedding': text_embedding.tolist()
                        })
                    else:
                        logger.warning(f"Failed to create text embedding for product {product_id}")
                        
            except Exception as e:
                logger.error(f"Error processing product {idx}: {e}")
                continue
        
        logger.info(f"Created embeddings for {len(embeddings_data)} products")
        
        # Save embeddings
        self.save_embeddings(embeddings_data)
        
        return df
        
    def save_embeddings(self, embeddings_data: List[Dict]):
        """Save embeddings data to files in the format expected by multimodal RAG"""
        # Convert list of dicts to the expected format
        product_ids = []
        image_embeddings = []
        text_embeddings = []
        multimodal_embeddings = []
        
        for item in embeddings_data:
            product_ids.append(item['product_id'])
            
            # Handle None image embeddings
            if item['image_embedding'] is not None:
                image_embeddings.append(item['image_embedding'])
            else:
                # Use text embedding as fallback for image embedding
                image_embeddings.append(item['text_embedding'])
                
            text_embeddings.append(item['text_embedding'])
            multimodal_embeddings.append(item['multimodal_embedding'])
        
        # Save in the expected JSON format
        embeddings_dict = {
            'product_ids': product_ids,
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'multimodal_embeddings': multimodal_embeddings
        }
        
        embeddings_path = self.embeddings_dir / "clip_embeddings.json"
        with open(embeddings_path, 'w') as f:
            json.dump(embeddings_dict, f)
        
        # Also save as numpy arrays for faster loading
        np.save(self.embeddings_dir / "image_embeddings.npy", np.array(image_embeddings))
        np.save(self.embeddings_dir / "text_embeddings.npy", np.array(text_embeddings))
        np.save(self.embeddings_dir / "multimodal_embeddings.npy", np.array(multimodal_embeddings))
        
        logger.info(f"Saved {len(embeddings_data)} embeddings to {embeddings_path}")
        logger.info(f"Saved numpy arrays to {self.embeddings_dir}")
        
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data and embeddings"""
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save main dataframe (without embeddings for CSV)
        processed_path = data_dir / "processed_products.csv"
        # Only drop embedding columns if they exist
        columns_to_drop = [col for col in ['image_embedding', 'text_embedding', 'multimodal_embedding'] if col in df.columns]
        if columns_to_drop:
            df_to_save = df.drop(columns_to_drop, axis=1)
        else:
            df_to_save = df.copy()
        
        # Ensure we have a product_id column for the RAG system
        if 'product_id' not in df_to_save.columns and 'Uniq Id' in df_to_save.columns:
            df_to_save['product_id'] = df_to_save['Uniq Id']
        
        df_to_save.to_csv(processed_path, index=False)
        
        logger.info(f"Processed data saved to {processed_path}")
        logger.info(f"Final dataset contains {len(df)} products with valid images and embeddings")
        
    async def process_dataset(self):
        """Main processing pipeline"""
        logger.info("Starting full dataset processing...")
        
        # Load and clean dataset
        df = self.load_and_clean_dataset()
        if df.empty:
            logger.error("No data to process")
            return
            
        logger.info(f"Processing {len(df)} products...")
        
        # Download images
        df = await self.download_images(df)
        if df.empty:
            logger.error("No valid images downloaded")
            return
            
        # Create CLIP embeddings
        df = self.create_multimodal_embeddings(df)
        
        # Save processed data
        self.save_processed_data(df)
        
        logger.info("Dataset processing completed!")
        logger.info(f"Final dataset contains {len(df)} products with valid images and embeddings")
        return df

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    processor = AmazonDataProcessor(openai_api_key)
    
    # Process the full dataset
    asyncio.run(processor.process_dataset()) 