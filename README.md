# Chickpea-Quality-Assessment-System

A comprehensive AI-powered system for automated chickpea quality assessment and counting using computer vision and machine learning.

##  Features

- **Quality Classification**: 3-class classification (healthy, broken, discolored)
- **Chickpea Counting**: Advanced contour detection and ML-based counting
- **Image Preprocessing**: Canny edge detection for enhanced feature extraction
- **Interactive UI**: Flask-based web application
- **Comprehensive Analysis**: Per-seed feature extraction and quality scoring
- **Multiple Models**: CNN, Random Forest ensemble, and ViT-based counting

##  Project Structure

```
chickpea-quality-assessment/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app.py                      # Main Flask application (enhanced)
├── chickpea_app.py             # Alternative Flask app
├── 
├── # Core Processing
├── image_processing.py         # Image preprocessing and counting
├── model_training.py           # Model training pipeline
├── evaluation.py               # Model evaluation metrics
├── dataset_preparation.py      # Dataset organization
├── train_chickpea_model.py     # Training script
├── 
├── # Trained Models
├── trained_model/              # Main quality classification model
│   ├── model.safetensors       # Model weights
│   ├── config.json             # Model configuration
│   ├── class_names.json        # Class labels
│   └── inference_example.py    # Usage example
├── 
├── chickpea_model/             # Alternative model version
│   ├── model.safetensors
│   ├── config.json
│   └── evaluation/             # Model evaluation results
├── 
├── count_model/                # Counting model
│   ├── model.safetensors
│   └── preprocessor_config.json
├── 
├── ensemble_model/             # Random Forest ensemble
│   ├── rf_classifier.pkl
│   └── feature_scaler.pkl
├── 
└── dataset/                    # Training and test data
    ├── train/
    │   ├── healthy/            # 202 images
    │   ├── broken/             # 24 images
    │   └── discolored/         # 51 images
    └── test/
        ├── healthy/            # 24 images
        ├── broken/             # 5 images
        └── discolored/         # 14 images
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd chickpea-quality-assessment

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Run the enhanced Flask app
python app.py

### 3. Upload and Analyze

1. Upload a chickpea image (JPG, PNG)
2. Adjust preprocessing parameters if needed
3. View quality classification and counting results
4. Analyze detailed per-seed features

## 🔧 Technical Details

### Models Used

1. **Quality Classification**: Vision Transformer (ViT) fine-tuned on chickpea images
2. **Counting Model**: ViT-based regression for accurate counting
3. **Ensemble Model**: Random Forest combining CNN and traditional features

### Preprocessing Pipeline

1. **Canny Edge Detection**: Highlights boundaries for better feature extraction
2. **Gaussian Blur**: Noise reduction
3. **Contour Detection**: Identifies individual chickpeas
4. **Feature Extraction**: Shape, color, texture, and defect analysis

### Quality Assessment Criteria

- **Healthy**: Intact, good color, proper size
- **Broken**: Cracked, split, or damaged
- **Discolored**: Off-color, stained, or blemished

##  Model Performance

### Quality Classification Model
- **Accuracy**: 56.8%
- **Precision**: 44.1%
- **Recall**: 56.8%
- **F1-Score**: 43.0%

### Counting Model
- **Method**: ViT-based regression
- **Features**: Multi-scale edge detection
- **Confidence**: Estimated per prediction

##  Development

### Training New Models

```bash
# Prepare dataset
python dataset_preparation.py

# Train quality classification model
python train_chickpea_model.py

# Train counting model
python count_model_training.py
```

### Evaluation

```bash
# Evaluate model performance
python evaluation.py
```

## 📈 Features

### Enhanced App (app.py)
- Advanced preprocessing with multiple methods
- Ensemble learning approach
- Comprehensive per-seed analysis
- Interactive visualizations
- Export capabilities

### Standard App (app.py)
- Basic quality classification
- Contour-based counting
- Simple UI for quick analysis

## 🎯 Use Cases

- **Agricultural Quality Control**: Automated chickpea quality assessment
- **Research**: Computer vision research in agriculture
- **Education**: Learning about image classification and counting
- **Production**: Integration into larger agricultural systems

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- Flask 2.3+
- Transformers 4.35+

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

##  License

This project is for educational and research purposes. Please ensure you have appropriate permissions for any commercial use.

##  Acknowledgments

- Hugging Face for the transformer models
- OpenCV for computer vision capabilities
- Flask for the web interface
- The agricultural research community for inspiration

---

**Note**: This system is optimized for chickpea quality assessment but can be adapted for other seed types with appropriate training data.

# project
