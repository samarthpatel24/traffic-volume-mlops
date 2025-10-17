# 🚗 Traffic Volume Prediction MLOps Project

A complete MLOps pipeline for predicting highway traffic volume using weather and time data with DVC, Docker, CI/CD, and AWS deployment.

## 📊 Project Overview

This project implements an end-to-end MLOps solution for predicting interstate traffic volume using machine learning. It features automated pipelines, multiple models (Random Forest, XGBoost, LightGBM, Ensemble), containerization with Docker, and cloud deployment on AWS.

## 🚀 Key Features

- **Complete MLOps Pipeline**: DVC for data versioning and pipeline orchestration
- **Multiple ML Models**: XGBoost, Random Forest, LightGBM, and Ensemble methods
- **Interactive Web App**: Streamlit-based user interface for predictions
- **Containerization**: Docker and Docker Compose for easy deployment
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Cloud Deployment**: AWS EC2 and S3 integration with automated deployment
- **Monitoring**: Health checks, logging, and performance metrics

## 🏗️ Project Structure

```
traffic-volume-predictor/
├── app.py                          # Streamlit web application
├── dvc.yaml                        # DVC pipeline configuration  
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Local development setup
├── requirements.txt                # Python dependencies
├── DEPLOYMENT_GUIDE.md             # Complete deployment guide
├── data/                           # Raw and processed data
│   ├── raw/                        # Original dataset
│   └── processed/                  # Preprocessed data
├── src/                            # ML pipeline source code
│   ├── preprocess.py               # Data preprocessing
│   ├── train.py                    # Model training
│   └── evaluate.py                 # Model evaluation
├── models/                         # Trained model artifacts
├── metrics/                        # Performance metrics and reports
├── plots/                          # Generated visualizations
├── config/                         # Configuration files
│   └── config.yaml                 # Model and pipeline config
├── tests/                          # Unit tests
├── .github/workflows/              # CI/CD pipeline configuration
├── aws/                           # AWS deployment templates
├── scripts/                       # Deployment automation scripts
└── notebooks/                     # Jupyter notebooks for exploration
```

## � Quick Start

1. **Install and setup:**
```bash
git clone <repo-url>
cd traffic-volume-predictor
pip install -r requirements.txt
```

2. **Run pipeline:**
```bash
dvc repro
```

3. **Launch web app:**
```bash
streamlit run app.py
```

## 🤖 GitHub Actions Automation

### Automated Workflows

- **Model Training**: Auto-retraining on data/code changes, scheduled runs
- **CI/CD**: Code linting, testing, security scanning  
- **Deployment**: Streamlit app deployment with Docker
- **Drift Monitoring**: Daily data drift detection with alerts

### Manual Triggers
Use GitHub Actions tab to manually run:
- Model training with custom parameters
- App deployment to different environments
- Data drift monitoring with custom thresholds

## � Usage

### Training Models
```bash
# Complete pipeline
dvc repro

# Individual stages
dvc repro preprocess  # Data preprocessing
dvc repro train       # Model training
dvc repro evaluate    # Model evaluation
```

### Web Application
Input weather conditions, time, and get traffic volume predictions with visualizations.

## 📊 Model Performance

Ensemble model achieves:
- **R² Score**: ~0.85-0.90
- **RMSE**: ~800-1000 vehicles
- **MAPE**: ~15-20%

## � Configuration

Edit `config/config.yaml`:
```yaml
model:
  type: 'ensemble'  # 'random_forest', 'xgboost', 'lightgbm', 'ensemble'
training:
  test_size: 0.2
  cv_folds: 5
```

##  Pipeline Stages

1. **Preprocessing**: Data cleaning, feature engineering, scaling
2. **Training**: Model training with cross-validation
3. **Evaluation**: Testing, metrics, visualizations

## 🐛 Troubleshooting

**Common Issues:**
- **Pipeline fails**: Check `dvc.yaml` paths and dependencies
- **App errors**: Ensure models trained with `dvc repro`
- **GitHub Actions**: Check logs, verify secrets, ensure files committed

## 🤝 Contributing

1. Fork repository
2. Create feature branch  
3. Test changes
4. Submit pull request

## 📄 License

MIT License

---

**Built with ❤️ for MLOps demonstration**