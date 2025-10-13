# 🚗 Traffic Volume Prediction MLOps Project

A complete MLOps pipeline for predicting highway traffic volume using weather and time data with DVC, Python, and Streamlit.

## 📊 Project Overview

Predicts interstate traffic volume using machine learning with automated pipelines, multiple models (Random Forest, XGBoost, LightGBM), and a Streamlit web interface.

## 🏗️ Project Structure

```
traffic-volume-predictor/
├── data/                            # Raw and processed data
├── src/                             # ML pipeline scripts
├── models/                          # Trained models
├── metrics/                         # Evaluation results
├── config/config.yaml               # Configuration
├── app.py                          # Streamlit web app
├── dvc.yaml                        # DVC pipeline
└── requirements.txt                # Dependencies
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