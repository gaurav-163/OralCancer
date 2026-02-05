# Oral Cancer Detection AI

An AI-powered oral cancer screening and risk assessment application built with Streamlit, TensorFlow, and LangGraph.

## Features

- **Image Classification**: Uses deep learning model to classify oral lesions
- **Risk Assessment**: Evaluates overall risk based on prediction and patient factors
- **AI Analysis**: Generates detailed clinical interpretation using Cohere LLM
- **Patient Information**: Considers patient history for personalized recommendations
- **Professional UI**: Clean, modern interface designed for healthcare professionals

## Model Information

- **Model**: `gauravvv7/Oralcancer` from Hugging Face
- **Input Size**: 300x300 RGB images
- **Output**: Binary classification (0 = Oral Cancer, 1 = Normal)
- **Classes**: Normal, Oral Cancer

## Quick Start

### Prerequisites

- Python 3.10+
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd oral_cancer_ai/app
   ```

2. **Install dependencies using UV**
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the `app` directory:
   ```bash
   COHERE_API_KEY=your_cohere_api_key_here
   ```
   
   Get a free API key from: https://dashboard.cohere.com/

4. **Download the model** (automatic on first run)
   
   The model will be automatically downloaded from Hugging Face on first run.
   
   Or manually download:
   ```bash
   huggingface-cli download gauravvv7/Oralcancer oral-cancer-model.h5 --local-dir models/
   ```

### Running the Application

```bash
# Using UV (recommended)
uv run streamlit run main.py

# Or using standard Python
streamlit run main.py
```

The app will be available at: **http://localhost:8501**

## Usage

1. **Upload an Image**: Click "Browse files" or drag and drop an image of the oral cavity
2. **Fill Patient Info** (optional): Add patient details in the sidebar for personalized analysis
3. **Click "Analyze Image"**: The AI will process the image and generate results
4. **Review Results**:
   - Risk Assessment (High/Moderate/Low)
   - Classification Results with probabilities
   - AI-generated analysis
   - Personalized recommendations

## Project Structure

```
app/
├── main.py              # Main Streamlit application
├── agents/              # LangGraph workflow agents
│   ├── graph.py         # Workflow graph definition
│   ├── nodes.py         # Processing nodes
│   └── state.py         # State management
├── components/          # UI components
│   ├── sidebar.py       # Settings sidebar
│   ├── image_upload.py  # Image upload component
│   └── results_display.py # Results visualization
├── model/               # ML model wrapper
│   └── predictor.py     # Model loading and prediction
├── models/              # Downloaded model files
├── utils/               # Utility functions
│   ├── image_processing.py
│   └── prompts.py       # LLM prompts
└── .env                 # Environment variables (create this)
```

## Configuration

Edit `../config/config.yaml` to customize:

```yaml
# Model settings
model:
  huggingface_repo_id: "gauravvv7/Oralcancer"
  classes:
    - "Normal"
    - "Oral Cancer"
  confidence_threshold: 0.7

# LLM settings
llm:
  provider: "cohere"
  model: "command-r-08-2024"
```

## Troubleshooting

### Model not loading
- Ensure you have internet connection for first-time download
- Check if `models/oral-cancer-model.h5` exists
- Verify TensorFlow is installed correctly

### LLM Analysis not working
- Verify your `COHERE_API_KEY` is set correctly in `.env`
- Check if the API key has remaining credits
- The app will still show classification results even if LLM fails

### Memory issues
- The model requires ~270MB of memory
- Close other applications if running on limited RAM

## Medical Disclaimer

This tool is designed to assist healthcare professionals and should **NOT** be used as a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider for proper diagnosis and treatment.

## License

This project is for educational and research purposes only.

## Credits

- Model: [gauravvv7/Oralcancer](https://huggingface.co/gauravvv7/Oralcancer)
- Built with: Streamlit, TensorFlow, LangGraph, Cohere
