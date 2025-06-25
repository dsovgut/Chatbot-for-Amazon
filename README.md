# Multimodal Conversational AI for E-commerce

A sophisticated multimodal conversational AI system for e-commerce that combines vision and language capabilities using **CLIP (Contrastive Language-Image Pre-training)** to provide intelligent product recommendations and customer support. **Now includes comprehensive evaluation metrics and performance tracking!**

## 🌟 Features

### 🔍 **True Multimodal Search with CLIP**
- **Shared embedding space**: CLIP creates unified representations for both text and images
- **Text-based queries**: Search using natural language descriptions
- **Image-based search**: Upload product images for visual similarity search
- **Combined search**: Seamlessly blend text and image queries for enhanced accuracy

### 🤖 **Intelligent Conversational AI**
- **Context-aware responses**: Maintains conversation history for personalized interactions
- **Visual understanding**: Analyzes uploaded images using GPT-4 Vision + CLIP embeddings
- **Product recommendations**: AI-powered suggestions based on multimodal similarity
- **Detailed product information**: Comprehensive specifications and features
- **Comparison capabilities**: Side-by-side product comparisons

### 📊 **Comprehensive Real-World Evaluation Metrics**
- **Real-time performance tracking**: See metrics for every query
- **Practical evaluation**: No artificial ground truth needed - uses real product data
- **Embedding quality assessment**: Intra-cluster coherence, inter-cluster separation, dimensionality analysis
- **Retrieval diversity & coverage**: Measures result variety and comprehensiveness
- **Response consistency**: Evaluates coherence and product mention accuracy
- **User behavior simulation**: Predicts click-through rates, engagement, and satisfaction
- **Cross-modal alignment**: Assesses image-text consistency without labels
- **System performance**: Response times, throughput, and memory efficiency
- **Automated reporting**: Generate detailed insights and actionable recommendations

### 🎯 **Advanced CLIP-based RAG System**
- **CLIP embeddings**: Uses OpenAI's CLIP model for true multimodal understanding
- **Multiple search modes**: Image-only, text-only, and multimodal search
- **Semantic similarity**: Finds products based on visual and textual meaning
- **Real-time processing**: Fast response times with efficient vector search
- **Local vector database**: Stores embeddings locally for fast retrieval

### 📱 **Modern User Interface**
- **Chainlit-powered UI**: Beautiful, responsive chat interface
- **Image display**: Shows product images inline with responses
- **File upload support**: Drag-and-drop image uploads
- **Real-time streaming**: Live response generation
- **Chat history**: Maintains conversation context
- **Performance dashboard**: Live metrics display for each query

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- OpenAI API key
- 8GB+ RAM (16GB+ recommended for full dataset)
- GPU recommended (CUDA-compatible) for faster CLIP processing

### 1. Clone and Setup Environment

        ```bash
git clone <repository-url>
cd genai_final_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Configure API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your_openai_api_key_here
```

### 3. Choose Your Setup

#### Option A: Test Setup (Recommended for first time)
```bash
python setup.py --test-setup
```
This processes 500 products (~30 minutes, ~$10-20 in API costs)

#### Option B: Full Dataset Setup
```bash
python setup.py --full-setup
```
This processes the entire dataset (~4-8 hours, ~$50-100 in API costs)

### 4. Launch the Application

    ```bash
    chainlit run app/main.py -w --port 8005
    ```

Open your browser to `http://localhost:8005`

## 📊 Real-World Evaluation Metrics System

### 🔬 **Practical Evaluation Framework**

Our system implements comprehensive evaluation metrics that work with real data without requiring artificial ground truth labels:

#### **1. Embedding Quality Assessment**
- **Intra-cluster coherence**: How similar are semantically related products
- **Inter-cluster separation**: How well are different product categories separated
- **Effective dimensionality**: How efficiently the embedding space is utilized
- **Norm consistency**: Stability of embedding magnitudes across products

#### **2. Retrieval Diversity & Coverage Analysis**
- **Result diversity**: Variety in retrieved product types and features
- **Coverage analysis**: How comprehensively results address query aspects
- **Consistency testing**: Stability of results across repeated queries
- **Vocabulary diversity**: Range of product features represented

#### **3. Response Consistency & Quality**
- **Length consistency**: Stability of response lengths across similar queries
- **Product mention accuracy**: How well responses reference retrieved products
- **Coherence scoring**: Logical flow and structure of responses
- **Content relevance**: Alignment between responses and retrieved products

#### **4. User Behavior Simulation**
- **Click-through prediction**: Likelihood of user engagement with results
- **Engagement scoring**: Predicted user interaction quality
- **Satisfaction estimation**: Overall user experience prediction
- **Position bias modeling**: Impact of result ranking on user behavior

#### **5. Cross-Modal Alignment Quality**
- **Image-text consistency**: How well product images and descriptions align
- **Cross-modal retrieval**: Consistency between visual and textual search
- **Multimodal coherence**: Quality of combined image-text understanding
- **Alignment distribution**: Consistency across different product types

#### **6. System Performance Metrics**
- **Response time analysis**: End-to-end query processing speed
- **Memory efficiency**: Resource utilization per product
- **Throughput measurement**: Queries processed per second
- **Scalability assessment**: Performance under different loads

### 🎯 **Real-time Evaluation in UI**

Every query in the chat interface displays:
- **Products found**: Number of relevant results
- **Search mode**: Text, image, or multimodal
- **Average relevance**: Mean similarity score
- **Response time**: Total processing time
- **CLIP similarity scores**: Top product matches

### 📈 **Standalone Evaluation Tools**

#### Quick Real-World Evaluation
```bash
python run_evaluation.py --quick
```
Runs evaluation with 5 realistic queries (~2 minutes)

#### Standard Real-World Evaluation  
```bash
python run_evaluation.py --save-report
```
Comprehensive evaluation with detailed report (~5-10 minutes)

#### Detailed Real-World Evaluation
```bash
python run_evaluation.py --detailed --save-report
```
Full evaluation with all metrics (~15-30 minutes)

### 📋 **Evaluation Reports**

Generated reports include:
- **Overall performance score** (0-10 scale)
- **Detailed metric breakdowns** by category
- **Performance grades** (Excellent, Very Good, Good, Fair, Needs Improvement)
- **Actionable recommendations** for system improvements
- **Realistic insights** based on actual product data

Example real-world evaluation output:
```
🎯 OVERALL PERFORMANCE SCORE: 7.8/10
Grade: 🥇 VERY GOOD

🧠 Embedding Quality Assessment:
  • Overall Quality: 0.742
  • Intra-cluster Coherence: 0.681
  • Inter-cluster Separation: 0.589

🔍 Retrieval Diversity & Coverage:
  • Average Diversity Score: 0.634
  • Average Coverage Score: 0.721
  • Result Consistency: 0.856

👤 User Behavior Simulation:
  • Predicted Click-Through Rate: 0.423
  • Estimated User Satisfaction: 0.678

⚡ System Performance:
  • Average Response Time: 2.1s
  • Throughput: 0.48 queries/second
```

## 📊 Dataset

The system uses the **complete Amazon Product Dataset 2020** from Kaggle, which includes:
- **~10,000+ products** with names, descriptions, and specifications
- **High-quality product images** (validated and downloaded)
- **Comprehensive product metadata**
- **Multiple product categories** (electronics, home, fashion, etc.)

The dataset is automatically downloaded and processed with CLIP embeddings during setup.

## 🛠️ System Architecture

### CLIP-based Multimodal Processing
1. **Dataset Download**: Automatic download from Kaggle
2. **Image Validation**: Filters and validates product images (removes broken links)
3. **CLIP Embedding Generation**: Creates unified embeddings for images and text
4. **OpenAI Vision Enhancement**: Uses GPT-4 Vision for detailed image descriptions
5. **Vector Storage**: Stores embeddings locally for fast similarity search

### RAG System Components
- **CLIP Query Processing**: Handles text, image, and multimodal queries
- **Semantic Retrieval**: Finds relevant products using cosine similarity in CLIP space
- **Context Generation**: Formats retrieved products for LLM consumption
- **Response Generation**: Uses GPT-4 to generate conversational responses

### Evaluation Pipeline
- **Ground Truth Dataset**: Curated test queries with expected results
- **Automated Metrics**: Real-time calculation of performance indicators
- **LLM-based Assessment**: Quality evaluation using GPT-4
- **Report Generation**: Automated insights and recommendations

### Frontend Architecture
- **Chainlit Framework**: Modern chat interface with real-time streaming
- **File Upload Handling**: Secure temporary file processing for images
- **Session Management**: Maintains conversation history and context
- **Metrics Dashboard**: Live performance tracking and display
- **Error Handling**: Graceful error recovery and user feedback

## 💬 Usage Examples

### Text-Based Queries
```
"I'm looking for wireless noise-cancelling headphones under $200"
"Show me gaming laptops with RTX graphics cards"
"What are the best fitness trackers for running?"
"Find me a red dress for a wedding"
```

### Image-Based Queries
- Upload a product image and ask: "Find similar products to this"
- "What is this product and how do I use it?"
- "Show me alternatives to this item in different colors"
- Upload a room photo: "What furniture would match this style?"

### Multimodal Queries
- Upload an image + text: "Find products similar to this but in blue color"
- "Show me headphones like this one but with better battery life"
- Upload product image: "Compare this with similar products under $100"

### Comparison Requests
```
"Compare the iPhone 15 with Samsung Galaxy S24"
"What's the difference between these two laptops?"
"Which fitness tracker is better for swimming?"
```

## 🔧 Configuration Options

### Dataset Processing
```bash
# Test setup with limited data
python setup.py --test-setup

# Process custom number of products
python setup.py --process-data --max-products 1000

# Process full dataset
python setup.py --full-setup

# Check system status
python setup.py --check
```

### Evaluation Options
```bash
# Quick evaluation (3 queries)
python run_evaluation.py --quick

# Standard evaluation with report
python run_evaluation.py --save-report

# Detailed evaluation (all queries)
python run_evaluation.py --detailed --save-report

# Custom output directory
python run_evaluation.py --output-dir my_results --save-report
```

### Application Settings
- **Port configuration**: Modify port in launch command
- **CLIP model**: Currently uses `openai/clip-vit-base-patch32`
- **Embedding dimensions**: 512 (CLIP standard)
- **Search modes**: image, text, multimodal

## 📁 Project Structure

```
genai_final_project/
├── app/
│   └── main.py                 # Main Chainlit application with metrics
├── data/                       # Processed dataset and embeddings
│   ├── images/                 # Downloaded product images
│   ├── embeddings/             # CLIP vector embeddings
│   │   ├── clip_embeddings.json
│   │   ├── image_embeddings.npy
│   │   ├── text_embeddings.npy
│   │   └── multimodal_embeddings.npy
│   └── processed_products.csv  # Cleaned product data
├── evaluation_results/         # Evaluation metrics and reports
│   ├── evaluation_summary_*.json
│   ├── evaluation_detailed_*.json
│   └── evaluation_report_*.md
├── temp_uploads/               # Temporary file storage
├── data_processor.py           # CLIP-based dataset processing
├── multimodal_rag.py          # CLIP-based RAG system
├── evaluation_metrics.py      # Comprehensive evaluation framework
├── run_evaluation.py          # Standalone evaluation script
├── setup.py                   # Setup and initialization script
├── test_system.py             # System testing with evaluation
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── README.md                  # This file
```

## 🎯 Key Technologies

- **CLIP (OpenAI)**: Multimodal vision-language understanding
- **OpenAI GPT-4o-mini**: Vision analysis and conversational responses
- **PyTorch**: Deep learning framework for CLIP
- **Transformers (Hugging Face)**: CLIP model implementation
- **Chainlit**: Modern chat interface framework
- **scikit-learn**: Vector similarity calculations and evaluation metrics
- **Pandas**: Data processing and manipulation
- **Pillow**: Image processing and validation
- **aiohttp**: Async HTTP client for image downloads

## 🔍 Real-World Evaluation Metrics Implementation

### Practical Approach
Our evaluation framework uses realistic metrics that don't require artificial labels:
- **Data-driven queries**: Automatically generated from actual product names and categories
- **Embedding analysis**: Mathematical assessment of CLIP embedding quality
- **Behavioral modeling**: Simulates realistic user interactions and preferences
- **Performance profiling**: Real-time system performance tracking

### Automated Assessment
- **Realistic test queries**: Generated from your actual product catalog
- **Similarity-based scoring**: Uses CLIP embeddings for relevance assessment
- **Consistency testing**: Measures system stability and reliability
- **Performance monitoring**: Tracks response times and resource usage

### Continuous Monitoring
- **Per-Query Metrics**: Every user interaction generates performance data
- **Session Analytics**: Track user behavior and system performance over time
- **Trend Analysis**: Monitor system performance improvements or degradation
- **A/B Testing Ready**: Framework supports comparative evaluation

## 🚨 Troubleshooting

### Common Issues

**"RAG system not initialized"**
- Ensure OpenAI API key is set correctly
- Run `python setup.py --check` to verify setup

**"No processed data found"**
- Run `python setup.py --test-setup` for quick setup
- Run `python setup.py --full-setup` for complete dataset

**"CLIP model loading failed"**
- Ensure PyTorch and transformers are installed correctly
- Check GPU drivers if using CUDA

**"Evaluation failed"**
- Check OpenAI API key and credits
- Ensure processed data exists
- Try `python run_evaluation.py --quick` for limited evaluation

**"Image upload failed"**
- Ensure uploaded file is a valid image format (JPG, PNG, etc.)
- Check file size (max 10MB recommended)

**"API rate limit exceeded"**
- Reduce processing batch size in `data_processor.py`
- Add delays between API calls
- Consider processing in smaller chunks

### Performance Optimization

**For faster processing:**
- Use GPU for CLIP processing (10x faster than CPU)
- Use `--test-setup` for initial testing
- Increase batch sizes if you have more RAM

**For better accuracy:**
- Process the full dataset with `--full-setup`
- Use higher resolution images (up to 512x512)
- Combine text and image queries for multimodal search

**Memory optimization:**
- Use numpy embeddings (.npy files) for faster loading
- Process dataset in chunks if memory is limited
- Close unused applications during processing

## 💰 Cost Estimation

### API Costs (OpenAI)
- **Test setup (500 products)**: ~$10-20
- **Full dataset (~10,000 products)**: ~$50-100
- **Runtime queries**: ~$0.01-0.05 per query
- **Evaluation runs**: ~$2-5 per comprehensive evaluation

### Hardware Requirements
- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: CUDA-compatible GPU recommended (RTX 3060+ or better)
- **Storage**: 5GB+ for full dataset and embeddings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run evaluation to ensure performance
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for CLIP and GPT-4 Vision models
- **Amazon Product Dataset 2020** from Kaggle
- **Hugging Face** for transformers library
- **Chainlit** team for the excellent chat framework
- **PyTorch** community for deep learning framework
- **scikit-learn** for evaluation metrics implementation

---

**Built with ❤️ for the future of multimodal e-commerce AI**

## 🚀 Getting Started Now

Ready to try it? Run these commands:

```bash
# Quick setup for testing
python setup.py --test-setup

# Test the system (including evaluation)
python test_system.py

# Run evaluation
python run_evaluation.py --quick --save-report

# Launch the app
chainlit run app/main.py -w --port 8005
```

Then open `http://localhost:8005` and start chatting with your AI shopping assistant!

### 📊 See Your System Performance

Every query shows real-time metrics:
- Response time and throughput
- CLIP similarity scores
- Search mode effectiveness
- Product relevance scores

Run comprehensive evaluation anytime:
```bash
python run_evaluation.py --detailed --save-report
```

**Experience the power of practical, real-world multimodal AI evaluation!** 🎯
