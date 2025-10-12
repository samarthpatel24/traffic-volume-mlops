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

## 🤖 GitHub Actions CI/CD Pipeline

This project includes comprehensive GitHub Actions workflows for MLOps automation:

### 🔄 Automated Workflows

#### 1. **Model Training Pipeline** (`model-training.yml`)
**Triggers:**
- Push to main/master branch (when data, code, or config changes)
- Pull requests
- Manual dispatch with configurable parameters
- Scheduled runs (every Sunday at 2 AM UTC)

**Features:**
- Automatic model retraining on data changes
- Configurable model types (ensemble, random_forest, xgboost, lightgbm)
- Data validation and quality checks
- Model performance validation
- Automatic artifact storage and versioning
- Model releases on successful training

#### 2. **CI/CD Pipeline** (`ci-cd.yml`)
**Triggers:**
- Push to any branch
- Pull requests

**Features:**
- Code linting with flake8
- Code formatting checks with black
- Import and configuration validation
- Security scanning with safety
- Dependency auditing

#### 3. **Streamlit App Deployment** (`deploy-streamlit.yml`)
**Triggers:**
- Push to main/master (when app files change)
- Manual dispatch with environment selection

**Features:**
- App testing and validation
- Docker containerization
- Multi-platform deployment packages
- Deployment documentation generation

#### 4. **Data Drift Monitoring** (`data-drift-monitoring.yml`)
**Triggers:**
- Daily scheduled runs (3 AM UTC)
- Manual dispatch with configurable thresholds

**Features:**
- Statistical drift detection (KS test, Chi-square test)
- Automated drift visualization
- GitHub issue creation on drift detection
- Automatic model retraining trigger on significant drift
- Drift reports and monitoring artifacts

### 🚀 Using GitHub Actions

#### Manual Model Training
```bash
# Go to Actions tab in GitHub
# Select "Model Training Pipeline"
# Click "Run workflow"
# Choose model type and provide reason
```

#### Manual Deployment
```bash
# Go to Actions tab in GitHub
# Select "Deploy Streamlit App"
# Click "Run workflow"
# Choose deployment environment
```

#### Monitoring Data Drift
```bash
# Go to Actions tab in GitHub
# Select "Data Drift Monitoring"
# Click "Run workflow"
# Set custom threshold if needed
```

### 📊 Automated MLOps Features

- **🔄 Continuous Training**: Models retrain automatically when new data is pushed
- **📈 Performance Monitoring**: Automatic validation against performance thresholds
- **🚨 Drift Detection**: Daily monitoring with automatic alerts and retraining
- **📦 Artifact Management**: Automatic storage and versioning of models and metrics
- **🏷️ Model Releases**: Tagged releases with performance metrics
- **📋 Issue Tracking**: Automatic issue creation for drift alerts and failures
- **🐳 Containerization**: Docker support for consistent deployments

### 🔧 Configuration

#### Performance Thresholds
Edit in `model-training.yml`:
```yaml
min_r2: 0.8        # Minimum R² score
max_rmse: 1000     # Maximum RMSE
```

#### Drift Detection Sensitivity
Edit in `data-drift-monitoring.yml`:
```yaml
threshold: 0.05    # Statistical significance level
```

#### Deployment Settings
Edit in `deploy-streamlit.yml`:
```yaml
environments: [staging, production]
```

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

4. **GitHub Actions Failures**:
   - Check workflow logs in Actions tab
   - Verify repository secrets are set
   - Ensure required files are committed
   - Check Python version compatibility

### GitHub Actions Troubleshooting

#### Model Training Workflow Issues
- **Data validation fails**: Check data format and required columns
- **Performance thresholds not met**: Review model configuration and data quality
- **DVC errors**: Ensure proper DVC setup and file tracking

#### Deployment Workflow Issues
- **App import errors**: Check dependencies and Python imports
- **Docker build fails**: Review Dockerfile and requirements
- **Missing model files**: Ensure model training completed successfully

#### Drift Detection Issues
- **Statistical tests fail**: Check data formats and column types
- **Visualization errors**: Ensure matplotlib and seaborn are available
- **Issue creation fails**: Check repository permissions and GitHub token

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