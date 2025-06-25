Multimodal Conversational AI for E-Commerce
This project is a multimodal conversational AI system designed to enhance the e-commerce customer experience. It leverages a Retrieval-Augmented Generation (RAG) architecture with Vision-Language models to answer product-related questions using both text and images.

Project Context
In the e-commerce domain, traditional customer support systems often rely solely on text, limiting their ability to provide comprehensive answers, especially when queries involve visual elements. This project bridges that gap by integrating both language and vision capabilities, allowing the system to interpret and respond to customer queries that include text and images, thereby providing more accurate, context-aware responses.

System Architecture
The system follows a RAG pipeline to process user queries and generate responses.

Data Flow:

Data Ingestion: Raw product data from a CSV file is cleaned and processed.

Image Processing: Product images are downloaded, validated, and standardized.

Embedding Generation: The CLIP model (openai/clip-vit-base-patch32) generates 512-dimension embeddings for product text and images. These are stored as .npy files.

User Query: A user submits a query, which can be text, an image, or a combination of both.

Query Encoding: The user's query is encoded into an embedding using the same CLIP model.

Vector Search: The system performs a cosine similarity search against the stored product embeddings to find the most relevant items.

Response Generation: The retrieved product information is passed as context to a Large Language Model (GPT-4o) which generates a natural language response for the user.

Core Modules
The system is built with a modular architecture:

app/main.py: The main user interface, built with Chainlit, which handles user interactions.

multimodal_rag.py: Contains the core MultimodalEcommerceRAG class that manages the retrieval and generation logic.

data_processor.py: A pipeline for processing the raw dataset, including text cleaning, image downloading, and embedding generation.

analytics_tracker.py: A simple module for tracking system performance metrics like response time and retrieval time.

File Structure
data/
├── amazon_products.csv
└── processed_products.csv
images/
└── ... (downloaded product images)
embeddings/
├── image_embeddings.npy
├── text_embeddings.npy
└── multimodal_embeddings.npy
app/
└── main.py
multimodal_rag.py
data_processor.py
analytics_tracker.py

Key Features
Multimodal Search: Supports three search modes:

Text Search: Uses the CLIP text encoder.

Image Search: Uses the CLIP image encoder.

Combined Search: Averages text and image embeddings for mixed queries.

Dynamic Response Generation: Integrates with GPT-4o and GPT-4 Vision to provide context-aware, conversational answers.

Data Processing Pipeline: Includes asynchronous image downloading, validation, and standardization for efficient data handling.

Technical Stack
ML / AI: PyTorch, Hugging Face Transformers, OpenAI API, Scikit-learn

Web Framework: Chainlit

Data Handling: Pandas, NumPy

Utilities: aiohttp

Dataset
The project utilizes the Amazon Product Dataset 2020 from Kaggle, which contains over 10,000 products. The data includes product names, descriptions, specifications, categories, pricing, and image URLs.

For processing, product text fields (title, brand, features, etc.) are concatenated into a unified description. Images are validated to be at least 50x50 pixels and standardized to a 512x512 JPEG format.

Performance Evaluation
The system was evaluated using an LLM-as-Judge methodology (GPT-4o-mini) on a set of 15 realistic user queries. A relevance score of ≥3 (out of 5) was considered a success.

Retrieval & Relevance Metrics:

Success Rate: 100%

Precision@5: 0.587 (58.7%)

Mean Reciprocal Rank (MRR): 0.802

Normalized Discounted Cumulative Gain (NDCG): 0.837

Average Relevance Score: 3.25 / 5.0

System Performance:

Average Response Time: 2.1 seconds

Throughput: 0.48 queries/second

Limitations
Dataset Quality: The performance is dependent on the quality and completeness of the product descriptions and images in the dataset.

API Dependencies: The system relies on the availability and latency of the OpenAI API.

Image Availability: The retrieval process requires valid and accessible image URLs in the dataset.

Contributors
Alex Tsourmas (alextsourmas@uchicago.edu)

Jane Lee (gylee@uchicago.edu)

Yeochan Youn (yyoun5@uchicago.edu)

Kaylie Nguyen (adqnguyen@uchicago.edu)

Danylo Sovgut (sovgut@uchicago.edu
