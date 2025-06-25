# Installation Guide - CLIP-based Multimodal E-commerce AI

## Quick Setup (Recommended)

### For Testing (500 products, ~30 minutes)
1. **Clone the repository and navigate to the project directory**
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Create a `.env` file with your OpenAI API key:**
   ```
   OPENAI_API_KEY=sk-your_openai_api_key_here
   ```

4. **Run the test setup:**
   ```bash
   python setup.py --test-setup
   ```

5. **Launch the application:**
   ```bash
   chainlit run app/main.py -w --port 8005
   ```

6. **Open your browser to:** `http://localhost:8005`

### For Full Dataset (~10,000 products, 4-8 hours)
Follow steps 1-3 above, then:

4. **Run the full setup:**
   ```bash
   python setup.py --full-setup
   ```
   ⚠️ **Warning**: This will cost ~$50-100 in OpenAI API calls

## Manual Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies (includes PyTorch, CLIP, transformers)
pip install -r requirements.txt
```

### 2. API Key Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your_openai_api_key_here
```

### 3. Dataset Processing Options

#### Option A: Test Dataset (Recommended first)
```bash
python setup.py --process-data --max-products 500
```

#### Option B: Custom Size
```bash
python setup.py --process-data --max-products 1000
```

#### Option C: Full Dataset
```bash
python setup.py --process-data
```

### 4. Test the System

```bash
python test_system.py
```

### 5. Run the Application

```bash
chainlit run app/main.py -w --port 8005
```

## System Requirements

### Minimum Requirements
- **Python:** 3.9 or higher
- **RAM:** 8GB (for test setup)
- **Storage:** 2GB for test setup
- **Internet:** Required for dataset download and API calls

### Recommended Requirements
- **Python:** 3.10 or higher
- **RAM:** 16GB+ (for full dataset)
- **GPU:** CUDA-compatible GPU (RTX 3060+ or better)
- **Storage:** 5GB+ for full dataset
- **Internet:** Stable connection for large dataset download

### GPU Support
The system automatically detects and uses GPU if available:
- **With GPU**: 10x faster CLIP processing
- **Without GPU**: Still works, but slower processing

Check GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Installation Issues

**"No module named 'torch'"**
```bash
# Install PyTorch manually first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**"No module named 'transformers'"**
```bash
pip install transformers>=4.30.0
```

**"No module named 'kagglehub'"**
```bash
pip install kagglehub
```

### Setup Issues

**"OPENAI_API_KEY not found"**
- Ensure your `.env` file is in the project root
- Check that your API key is valid and has sufficient credits
- Verify the file format: `OPENAI_API_KEY=sk-...` (no quotes)

**"No processed data found"**
```bash
python setup.py --test-setup
```

**"CLIP model loading failed"**
- Check internet connection (models download from Hugging Face)
- Ensure sufficient disk space (~2GB for CLIP model)
- Try clearing cache: `rm -rf ~/.cache/huggingface/`

**"Permission denied" errors**
- Ensure you have write permissions in the project directory
- Try running with elevated permissions if necessary

**"CUDA out of memory"**
- Reduce batch size in `data_processor.py`
- Use CPU instead: set `device = "cpu"` in the code
- Close other GPU-intensive applications

### Performance Issues

**Slow processing**
- Use GPU if available (10x speedup)
- Reduce dataset size for testing
- Increase RAM if possible

**High memory usage**
- Process dataset in smaller chunks
- Use `--max-products` to limit dataset size
- Close unnecessary applications

**API rate limits**
- Add delays between API calls in `data_processor.py`
- Reduce batch sizes
- Check your OpenAI API plan limits

## Cost Management

### API Cost Estimation
- **Test setup (500 products)**: ~$10-20
- **Medium setup (1000 products)**: ~$20-40  
- **Full dataset (~10,000 products)**: ~$50-100

### Cost Optimization Tips
1. Start with test setup to verify everything works
2. Use `--max-products` to control costs
3. Monitor your OpenAI API usage dashboard
4. Process in chunks if budget is limited

## Advanced Configuration

### Custom CLIP Model
To use a different CLIP model, modify `data_processor.py` and `multimodal_rag.py`:
```python
# Change this line in both files:
self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
```

### Batch Size Tuning
For better performance, adjust batch sizes in `data_processor.py`:
```python
# For more RAM/GPU memory:
batch_size = 100  # Increase from 50

# For less RAM/GPU memory:
batch_size = 20   # Decrease from 50
```

### Storage Optimization
The system saves embeddings in multiple formats:
- **JSON**: Human-readable, slower loading
- **NumPy**: Binary format, faster loading (recommended)

To use only NumPy format, modify the save functions in `data_processor.py`.

## Development Setup

For development and customization:

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_system.py

# Run with auto-reload
chainlit run app/main.py -w --port 8005

# Check system status
python setup.py --check
```

## Verification Steps

After installation, verify everything works:

1. **Check environment:**
   ```bash
   python setup.py --check
   ```

2. **Test CLIP loading:**
   ```bash
   python -c "from transformers import CLIPModel; print('CLIP OK')"
   ```

3. **Test system:**
   ```bash
   python test_system.py
   ```

4. **Launch app:**
   ```bash
   chainlit run app/main.py -w --port 8005
   ```

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Run `python test_system.py` for diagnostics
3. Check the logs for error messages
4. Ensure all requirements are met
5. Try the test setup first before full setup

## Next Steps

Once installed:
1. Upload product images to test visual search
2. Try text queries like "wireless headphones under $100"
3. Experiment with multimodal queries (text + image)
4. Explore the conversation history features 