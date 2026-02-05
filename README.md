# Oral Cancer Detection AI

An AI-powered oral cancer screening and risk assessment application built with Streamlit, LangGraph, and TensorFlow.

## 🔬 Features

- **Image-based Classification**: Upload oral cavity images for AI-powered lesion classification
- **Risk Assessment**: Multi-factor risk analysis considering patient demographics and history
- **LLM-powered Analysis**: Detailed explanations using Cohere Command R+ via LangGraph workflow
- **Actionable Recommendations**: Prioritized next steps based on analysis results
- **Modern UI**: Clean, responsive interface with real-time feedback

## 📁 Project Structure

```
oral_cancer_ai/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # Streamlit main app
│   ├── components/
│   │   ├── __init__.py
│   │   ├── sidebar.py          # Settings & patient info
│   │   ├── image_upload.py     # Image upload component
│   │   └── results_display.py  # Results visualization
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py            # LangGraph workflow
│   │   ├── nodes.py            # Graph nodes
│   │   └── state.py            # State management
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── predictor.py        # ML model wrapper
│   │   └── model_utils.py      # Model utilities
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_processing.py # Image preprocessing
│       └── prompts.py          # LLM prompts
│
├── models/
│   └── oral_cancer_model.h5    # Your trained model (add your own)
│
├── config/
│   └── config.yaml             # Configuration file
│
├── requirements.txt
├── .env                        # Environment variables
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Cohere API key (for LLM features) - Get one at https://dashboard.cohere.com/
- Trained oral cancer classification model (`.h5` format)

### Installation

1. **Clone the repository**
   ```bash
   cd oral_cancer_ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Edit .env file with your Cohere API key
   COHERE_API_KEY=your_cohere_api_key_here
   ```

5. **Model (Auto-Download)**
   - The model will be **automatically downloaded** from Hugging Face on first run
   - Source: `gauravvv7/Oralcancer` (270 MB)
   - Or manually download: `huggingface-cli download gauravvv7/Oralcancer oral-cancer-model.h5 --local-dir models/`

### Running the Application

```bash
cd app
streamlit run main.py
```

The app will be available at `http://localhost:8501`

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model Configuration
model:
  path: "models/oral_cancer_model.h5"
  input_size: [224, 224]
  classes:
    - "Normal"
    - "Oral Squamous Cell Carcinoma"
    - "Oral Lichen Planus"
    - "Leukoplakia"
  confidence_threshold: 0.7

# LLM Configuration
llm:
  provider: "cohere"
  model: "command-r-plus"  # or command-r, command-light
  temperature: 0.3
```

## 🧠 Model Training

The application expects a Keras/TensorFlow model with:
- Input shape: `(224, 224, 3)`
- Output: Softmax probabilities for each class

If you don't have a trained model, the app will use mock predictions for demonstration purposes.

### Sample Training Code

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Example model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train and save
model.fit(train_data, epochs=50)
model.save('models/oral_cancer_model.h5')
```

## 📊 LangGraph Workflow

The analysis pipeline uses LangGraph with the following nodes:

1. **Analyze Prediction** - LLM interprets ML model results
2. **Assess Risk** - Calculates risk level based on prediction + patient factors
3. **Generate Recommendations** - Creates prioritized action items
4. **Finalize** - Completes the workflow

```
[Start] → [Analyze Prediction] → [Assess Risk] → [Generate Recommendations] → [Finalize] → [End]
```

## ⚠️ Medical Disclaimer

This tool is designed to assist healthcare professionals and researchers. It should **NOT** be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare providers for proper diagnosis and treatment.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📧 Contact

For questions or support, please open an issue on the repository.
