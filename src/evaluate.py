import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import yaml
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_config(config_path='config/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model_and_data(model_path):
    """Load trained model and test data"""
    model = joblib.load(model_path)
    
    # Load test data
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df.drop('traffic_volume', axis=1)
    y_test = test_df['traffic_volume']
    
    return model, X_test, y_test

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics

def create_prediction_plots(y_test, y_pred, model_name):
    """Create prediction visualization plots"""
    os.makedirs('plots', exist_ok=True)
    
    # 1. Actual vs Predicted scatter plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Actual vs Predicted', 'Residuals Plot', 
                       'Prediction Error Distribution', 'Feature Importance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(x=y_test, y=y_pred, mode='markers', 
                  name='Predictions', opacity=0.6),
        row=1, col=1
    )
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name='Perfect Prediction', 
                  line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Residuals plot
    residuals = y_test - y_pred
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', 
                  name='Residuals', opacity=0.6),
        row=1, col=2
    )
    
    # Add horizontal line at y=0
    fig.add_trace(
        go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], 
                  mode='lines', name='Zero Line', 
                  line=dict(color='red', dash='dash')),
        row=1, col=2
    )
    
    # Residuals distribution
    fig.add_trace(
        go.Histogram(x=residuals, name='Residuals Distribution', 
                    nbinsx=30, opacity=0.7),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text=f"Model Evaluation - {model_name}")
    fig.update_xaxes(title_text="Actual Traffic Volume", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Traffic Volume", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Traffic Volume", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)
    fig.update_xaxes(title_text="Residuals", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    # Save plot
    fig.write_html(f'plots/{model_name}_evaluation.html')
    
    # Also save with consistent name for DVC tracking
    fig.write_html('plots/evaluation_plots.html')
    
    return fig

def create_feature_importance_plot(feature_importance_path, model_name):
    """Create feature importance plot"""
    if not os.path.exists(feature_importance_path):
        return None
    
    df_importance = pd.read_csv(feature_importance_path)
    
    # Create bar plot
    fig = px.bar(
        df_importance.head(10), 
        x='importance', 
        y='feature',
        orientation='h',
        title=f'Top 10 Feature Importance - {model_name}',
        labels={'importance': 'Importance Score', 'feature': 'Features'}
    )
    
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    fig.write_html(f'plots/{model_name}_feature_importance.html')
    
    # Also save with consistent name for DVC tracking
    fig.write_html('plots/feature_importance.html')
    
    return fig

def create_time_series_analysis(model_path, model_name):
    """Create time series analysis of predictions"""
    # Load original data with datetime
    df_raw = pd.read_csv('data/raw/Metro_Interstate_Traffic_Volume.csv')
    df_raw['date_time'] = pd.to_datetime(df_raw['date_time'], format='%d-%m-%Y %H:%M')
    
    # Load model and make predictions on a sample
    model = joblib.load(model_path)
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df.drop('traffic_volume', axis=1)
    y_test = test_df['traffic_volume']
    y_pred = model.predict(X_test)
    
    # Create time series plot (sample of data)
    sample_size = min(1000, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=y_test.iloc[indices],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue'),
        marker=dict(size=3)
    ))
    
    fig.add_trace(go.Scatter(
        y=y_pred[indices],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red'),
        marker=dict(size=3)
    ))
    
    fig.update_layout(
        title=f'Traffic Volume Prediction Over Time - {model_name}',
        xaxis_title='Time Points',
        yaxis_title='Traffic Volume',
        height=500
    )
    
    fig.write_html(f'plots/{model_name}_time_series.html')
    
    # Also save with consistent name for DVC tracking
    fig.write_html('plots/time_series_analysis.html')
    
    return fig

def generate_evaluation_report(model_path, model_info_path):
    """Generate comprehensive evaluation report"""
    # Load model info
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_name = model_info['model_name']
    
    # Load model and data
    model, X_test, y_test = load_model_and_data(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Create visualizations
    pred_plot = create_prediction_plots(y_test, y_pred, model_name)
    
    # Feature importance plot
    feature_importance_path = f"metrics/{model_name}_feature_importance.csv"
    importance_plot = create_feature_importance_plot(feature_importance_path, model_name)
    
    # Time series analysis
    time_series_plot = create_time_series_analysis(model_path, model_name)
    
    # Save updated metrics
    metrics_path = f"metrics/{model_name}_evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also save with consistent name for DVC tracking
    with open('metrics/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate summary report
    report = f"""
# Model Evaluation Report: {model_name}

## Model Information
- Model Type: {model_info['model_type']}
- Training Date: {model_info['timestamp']}
- Model Path: {model_path}

## Performance Metrics
- Mean Absolute Error (MAE): {metrics['mae']:.4f}
- Mean Squared Error (MSE): {metrics['mse']:.4f}
- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}
- R² Score: {metrics['r2']:.4f}
- Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%

## Model Interpretation
- R² Score of {metrics['r2']:.4f} indicates the model explains {metrics['r2']*100:.2f}% of the variance in traffic volume.
- RMSE of {metrics['rmse']:.2f} means the average prediction error is approximately {metrics['rmse']:.0f} vehicles.
- MAPE of {metrics['mape']:.2f}% indicates the average percentage error in predictions.

## Visualizations
- Prediction plots: plots/{model_name}_evaluation.html
- Feature importance: plots/{model_name}_feature_importance.html
- Time series analysis: plots/{model_name}_time_series.html

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save report
    report_path = f"metrics/{model_name}_evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Evaluation report generated: {report_path}")
    print(f"Performance Summary:")
    print(f"- R² Score: {metrics['r2']:.4f}")
    print(f"- RMSE: {metrics['rmse']:.2f}")
    print(f"- MAPE: {metrics['mape']:.2f}%")
    
    return metrics, report_path

def evaluate_latest_model():
    """Evaluate the most recently trained model"""
    # Use the latest model files for DVC tracking
    model_path = 'models/latest_model.pkl'
    model_info_path = 'models/latest_model_info.json'
    
    if not os.path.exists(model_path) or not os.path.exists(model_info_path):
        print("Latest model files not found! Please run training first.")
        return
    
    # Load model info
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    print(f"Evaluating model: {model_info['model_name']}")
    
    # Generate evaluation report
    metrics, report_path = generate_evaluation_report(model_path, model_info_path)
    
    return metrics, report_path

if __name__ == "__main__":
    evaluate_latest_model()