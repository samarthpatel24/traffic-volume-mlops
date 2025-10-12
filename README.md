# 🚗 Traffic Volume Prediction MLOps Project

A complete MLOps pipeline for predicting highway traffic volume based on weather conditions, time, and other factors. This project demonstrates best practices in machine learning operations using DVC, Python, and Streamlit.

## 📊 Project Overview

This project predicts interstate traffic volume using the Metro Interstate Traffic Volume dataset. It implements a complete MLOps pipeline with:

- **Data versioning** with DVC
- **Automated pipelines** for preprocessing, training, and evaluation
- **Multiple ML models** (Random Forest, XGBoost, LightGBM, Ensemble)
- **Interactive web application** with Streamlit
- **Comprehensive evaluation** and monitoring

## 🏗️ Project Structure

```
traffic-volume-predictor/
├── data/
│   ├── raw/                          # Raw dataset
│   └── processed/                    # Processed data
├── src/
│   ├── preprocess.py                 # Data preprocessing pipeline
│   ├── train.py                      # Model training pipeline
│   └── evaluate.py                   # Model evaluation pipeline
├── models/                           # Trained models and encoders
├── metrics/                          # Evaluation metrics and reports
├── plots/                           # Generated visualizations
├── notebooks/                       # Jupyter notebooks for analysis
├── config/
│   └── config.yaml                  # Configuration file
├── app.py                           # Streamlit web application
├── dvc.yaml                         # DVC pipeline definition
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## 🔧 Features

### Data Processing
- **Feature Engineering**: Extract time-based features (hour, day_of_week, rush_hour, etc.)
- **Weather Processing**: Convert temperature units, create weather severity scores
- **Categorical Encoding**: Label encoding for weather conditions
- **Feature Scaling**: StandardScaler for numerical features

### Machine Learning Models
- **Random Forest Regressor**: Robust ensemble method
- **XGBoost**: Gradient boosting for high performance
- **LightGBM**: Fast gradient boosting framework
- **Ensemble Model**: Voting regressor combining all models

### MLOps Pipeline
- **DVC Pipeline**: Automated workflow management
- **Version Control**: Data and model versioning
- **Reproducibility**: Consistent results across runs
- **Monitoring**: Performance tracking and evaluation

### Web Application
- **Interactive Interface**: User-friendly Streamlit app
- **Real-time Predictions**: Instant traffic volume forecasts
- **Visualizations**: Traffic patterns and model insights
- **Input Validation**: Error handling and user guidance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd traffic-volume-predictor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Initialize DVC (if not already done):**
```bash
dvc init
```

4. **Run the complete pipeline:**
```bash
dvc repro
```

This will:
- Process the raw data
- Train the machine learning models
- Evaluate model performance
- Generate visualizations and reports

### Running the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📋 Usage

### Training New Models

To train models with different configurations:

1. **Modify configuration:**
Edit `config/config.yaml` to change model parameters

2. **Run specific stages:**
```bash
# Data preprocessing only
dvc repro preprocess

# Training only
dvc repro train

# Evaluation only
dvc repro evaluate
```

3. **Run complete pipeline:**
```bash
dvc repro
```

### Using the Web Application

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Input parameters:**
   - Weather conditions (temperature, rain, snow, clouds)
   - Time information (date, hour)
   - Special conditions (holidays)

3. **Get predictions:**
   - View predicted traffic volume
   - See traffic level classification
   - Analyze input parameters

### Model Evaluation

View model performance:
- Check `metrics/` folder for detailed metrics
- Open `plots/` folder for visualizations
- Read evaluation reports in markdown format

## 📊 Model Performance

The ensemble model typically achieves:
- **R² Score**: ~0.85-0.90
- **RMSE**: ~800-1000 vehicles
- **MAPE**: ~15-20%

Individual model performance varies based on data and hyperparameters.

## 🔍 Key Features Explained

### Feature Engineering
- **Time Features**: Hour, day of week, month, weekend indicator
- **Rush Hour Detection**: Peak traffic periods (7-9 AM, 5-7 PM)
- **Weather Severity**: Numeric scale for weather impact
- **Precipitation Total**: Combined rain and snow

### Model Selection
The project supports multiple models:
- **Random Forest**: Good baseline, interpretable
- **XGBoost**: High performance, handles missing values
- **LightGBM**: Fast training, memory efficient
- **Ensemble**: Combines all models for best results

### Evaluation Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 🛠️ Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  type: 'ensemble'  # 'random_forest', 'xgboost', 'lightgbm', 'ensemble'
  
training:
  test_size: 0.2
  cv_folds: 5

# Model-specific parameters available
```

## 📈 Monitoring and Evaluation

### Automated Evaluation
- Performance metrics automatically generated
- Feature importance analysis
- Residual analysis and visualization
- Time series prediction analysis

### Manual Evaluation
```bash
python src/evaluate.py
```

### Visualizations
- Actual vs Predicted scatter plots
- Residual plots
- Feature importance charts
- Time series analysis

## 🔄 Pipeline Stages

1. **Preprocessing** (`src/preprocess.py`):
   - Data cleaning and validation
   - Feature engineering
   - Train/test splitting
   - Feature scaling

2. **Training** (`src/train.py`):
   - Model training with cross-validation
   - Hyperparameter tuning
   - Model serialization
   - Performance logging

3. **Evaluation** (`src/evaluate.py`):
   - Model testing
   - Metrics calculation
   - Visualization generation
   - Report creation

## 🐛 Troubleshooting

### Common Issues

1. **DVC Pipeline Fails**:
   - Check file paths in `dvc.yaml`
   - Ensure all dependencies are installed
   - Verify data files exist

2. **Model Loading Errors**:
   - Run the complete pipeline first: `dvc repro`
   - Check model files in `models/` directory

3. **Streamlit App Issues**:
   - Ensure models are trained
   - Check encoder files exist
   - Verify all dependencies installed

### Getting Help

- Check the logs in `training.log`
- Review error messages in terminal
- Ensure all files are present in expected locations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Metro Interstate Traffic Volume Dataset
- DVC team for the MLOps framework
- Streamlit for the web application framework
- scikit-learn, XGBoost, and LightGBM communities

---

**Built with ❤️ for MLOps learning and demonstration**