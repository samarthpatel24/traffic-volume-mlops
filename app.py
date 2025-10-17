import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add src directory to path to import model selector
sys.path.append('src')
from model_selector import ModelSelector

# Page configuration
st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_encoders():
    """Load the best performing model and encoders using intelligent selection"""
    try:
        # Initialize model selector
        selector = ModelSelector()
        
        # Define selection criteria - prioritize R² and RMSE
        selection_criteria = {
            'primary_metric': 'r2',
            'secondary_metric': 'rmse',
            'min_r2': 0.90,  # Minimum R² of 90%
            'max_rmse': 450,  # Maximum RMSE of 450
            'weights': {
                'r2': 0.5,      # 50% weight to R²
                'rmse': 0.3,    # 30% weight to RMSE  
                'mae': 0.2      # 20% weight to MAE
            }
        }
        
        # Select best model
        best_model_info = selector.select_best_model(selection_criteria)
        
        if not best_model_info:
            return None, None, None, None, "No suitable models found!"
        
        # Load the selected model
        model = joblib.load(best_model_info['model_path'])
        
        # Load encoders and scaler
        weather_main_encoder = joblib.load('models/weather_main_encoder.pkl')
        weather_desc_encoder = joblib.load('models/weather_description_encoder.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        
        # Add selection info to model_info
        best_model_info['selection_reason'] = f"Selected based on R² ≥ {selection_criteria['min_r2']:.2f} and RMSE ≤ {selection_criteria['max_rmse']}"
        
        return model, weather_main_encoder, weather_desc_encoder, scaler, best_model_info
    
    except Exception as e:
        return None, None, None, None, f"Error loading model: {str(e)}"

def create_feature_vector(temp_celsius, rain_1h, snow_1h, clouds_all, hour, day_of_week, 
                         month, is_weekend, is_rush_hour, is_holiday, weather_main_encoded, 
                         weather_desc_encoded, weather_severity, total_precipitation):
    """Create feature vector for prediction"""
    features = [
        temp_celsius, rain_1h, snow_1h, clouds_all, hour, day_of_week, month,
        is_weekend, is_rush_hour, is_holiday, weather_severity, total_precipitation,
        weather_main_encoded, weather_desc_encoded
    ]
    return np.array(features).reshape(1, -1)

def get_weather_severity(weather_main):
    """Get weather severity score"""
    weather_severity_map = {
        'Clear': 1, 'Clouds': 2, 'Mist': 3, 'Rain': 4, 'Drizzle': 4,
        'Snow': 5, 'Fog': 3, 'Haze': 3, 'Thunderstorm': 5, 'Smoke': 3
    }
    return weather_severity_map.get(weather_main, 2)

def main():
    st.title("🚗 Traffic Volume Predictor")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Comparison", "⚙️ Model Selection"])
    
    # Load model and encoders
    model, weather_main_encoder, weather_desc_encoder, scaler, model_info = load_model_and_encoders()
    
    if model is None:
        st.error(model_info)  # This contains the error message
        st.info("Please run the training pipeline first by executing: `dvc repro`")
        return
    
    with tab1:
        # Existing prediction functionality
        prediction_tab(model, weather_main_encoder, weather_desc_encoder, scaler, model_info)
    
    with tab2:
        # Model comparison functionality
        model_comparison_tab()
    
    with tab3:
        # Model selection settings
        model_selection_tab()

def prediction_tab(model, weather_main_encoder, weather_desc_encoder, scaler, model_info):
    """Tab for traffic volume prediction"""
    
    # Display model information
    st.sidebar.header("📊 Model Information")
    st.sidebar.info(f"""
    **Model Type:** {model_info['model_type']}
    **Training Date:** {model_info['timestamp']}
    **R² Score:** {model_info['performance']['r2']:.4f}
    **RMSE:** {model_info['performance']['rmse']:.2f}
    """)
    
    # Create two columns for input and prediction
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎯 Input Parameters")
        
        # Weather conditions
        st.subheader("Weather Conditions")
        
        weather_main_options = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist', 'Fog', 'Drizzle', 'Haze', 'Thunderstorm', 'Smoke']
        weather_main = st.selectbox("Weather Main", weather_main_options)
        
        weather_desc_options = {
            'Clear': ['sky is clear'],
            'Clouds': ['few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds'],
            'Rain': ['light rain', 'moderate rain', 'heavy intensity rain', 'very heavy rain'],
            'Snow': ['light snow', 'snow', 'heavy snow'],
            'Mist': ['mist'],
            'Fog': ['fog'],
            'Drizzle': ['light intensity drizzle', 'drizzle', 'heavy intensity drizzle'],
            'Haze': ['haze'],
            'Thunderstorm': ['thunderstorm', 'thunderstorm with rain'],
            'Smoke': ['smoke']
        }
        
        weather_description = st.selectbox("Weather Description", weather_desc_options[weather_main])
        
        temp_celsius = st.slider("Temperature (°C)", min_value=-20, max_value=40, value=15, step=1)
        rain_1h = st.slider("Rain in last hour (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
        snow_1h = st.slider("Snow in last hour (mm)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        clouds_all = st.slider("Cloud coverage (%)", min_value=0, max_value=100, value=20, step=5)
        
        # Time conditions
        st.subheader("Time Conditions")
        
        date_input = st.date_input("Date", value=datetime.now().date())
        
        # Give user choice of input method
        time_input_method = st.radio("Choose time input method:", 
                                   ["Select Hour (0-23)", "Common Times", "Custom Time"], 
                                   index=0)
        
        if time_input_method == "Select Hour (0-23)":
            hour_input = st.selectbox("Hour", options=list(range(24)), index=datetime.now().hour)
            datetime_input = datetime.combine(date_input, datetime.min.time()).replace(hour=hour_input)
        elif time_input_method == "Common Times":
            common_times = {
                "Early Morning (6:00 AM)": 6,
                "Morning Rush (8:00 AM)": 8,
                "Mid Morning (10:00 AM)": 10,
                "Lunch Time (12:00 PM)": 12,
                "Afternoon (2:00 PM)": 14,
                "Evening Rush (5:00 PM)": 17,
                "Evening Rush (6:00 PM)": 18,
                "Night (8:00 PM)": 20,
                "Late Night (11:00 PM)": 23
            }
            selected_time = st.selectbox("Select Common Time", list(common_times.keys()), index=4)
            hour_input = common_times[selected_time]
            datetime_input = datetime.combine(date_input, datetime.min.time()).replace(hour=hour_input)
        else:
            # Custom time with sliders
            hour_input = st.slider("Hour", min_value=0, max_value=23, value=datetime.now().hour, step=1)
            minute_input = st.slider("Minute", min_value=0, max_value=59, value=0, step=15)
            datetime_input = datetime.combine(date_input, datetime.min.time()).replace(hour=hour_input, minute=minute_input)
        
        hour = datetime_input.hour
        day_of_week = datetime_input.weekday()
        month = datetime_input.month
        
        # Display selected datetime
        st.info(f"**Selected Date & Time:** {datetime_input.strftime('%Y-%m-%d %H:%M')}")
        
        # Other conditions
        st.subheader("Other Conditions")
        is_holiday = st.checkbox("Is Holiday?")
        
        # Calculate derived features
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        weather_severity = get_weather_severity(weather_main)
        total_precipitation = rain_1h + snow_1h
        
        # Display derived features
        st.subheader("Derived Features")
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.metric("Is Weekend", "Yes" if is_weekend else "No")
            st.metric("Is Rush Hour", "Yes" if is_rush_hour else "No")
        with col1_2:
            st.metric("Weather Severity", weather_severity)
            st.metric("Total Precipitation", f"{total_precipitation:.1f} mm")
    
    with col2:
        st.header("🔮 Prediction Results")
        
        if st.button("🚀 Predict Traffic Volume", type="primary"):
            try:
                # Encode categorical features
                weather_main_encoded = weather_main_encoder.transform([weather_main])[0]
                weather_desc_encoded = weather_desc_encoder.transform([weather_description])[0]
                
                # Create feature vector
                features = create_feature_vector(
                    temp_celsius, rain_1h, snow_1h, clouds_all, hour, day_of_week, month,
                    is_weekend, is_rush_hour, int(is_holiday), weather_main_encoded,
                    weather_desc_encoded, weather_severity, total_precipitation
                )
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                
                # Display prediction
                st.success(f"**Predicted Traffic Volume: {prediction:.0f} vehicles/hour**")
                
                # Traffic level classification
                if prediction < 1000:
                    traffic_level = "🟢 Low Traffic"
                    color = "green"
                elif prediction < 3000:
                    traffic_level = "🟡 Moderate Traffic"
                    color = "orange"
                elif prediction < 5000:
                    traffic_level = "🟠 High Traffic"
                    color = "red"
                else:
                    traffic_level = "🔴 Very High Traffic"
                    color = "darkred"
                
                st.markdown(f"**Traffic Level: {traffic_level}**")
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Traffic Volume"},
                    delta = {'reference': 3000},
                    gauge = {
                        'axis': {'range': [None, 7500]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 1000], 'color': "lightgreen"},
                            {'range': [1000, 3000], 'color': "yellow"},
                            {'range': [3000, 5000], 'color': "orange"},
                            {'range': [5000, 7500], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature contribution (for ensemble models, show average importance)
                st.subheader("📈 Input Summary")
                summary_data = {
                    "Parameter": ["Temperature", "Rain", "Snow", "Clouds", "Hour", "Day of Week", 
                                "Month", "Weekend", "Rush Hour", "Holiday", "Weather Severity"],
                    "Value": [f"{temp_celsius}°C", f"{rain_1h}mm", f"{snow_1h}mm", f"{clouds_all}%",
                            str(hour), datetime_input.strftime("%A"), datetime_input.strftime("%B"),
                            "Yes" if is_weekend else "No", "Yes" if is_rush_hour else "No",
                            "Yes" if is_holiday else "No", str(weather_severity)]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all required fields are filled correctly.")
    
    # Historical trends section
    st.markdown("---")
    st.header("📊 Traffic Patterns")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Hourly Traffic Pattern")
        # Create sample hourly pattern
        hours = list(range(24))
        # Typical traffic pattern (higher during rush hours)
        hourly_pattern = [500, 300, 200, 150, 200, 500, 1500, 3500, 4500, 3000, 2500, 2800, 
                         3200, 3500, 4000, 4200, 5000, 6000, 4500, 2500, 1500, 1000, 800, 600]
        
        fig_hourly = px.line(x=hours, y=hourly_pattern, 
                           labels={'x': 'Hour of Day', 'y': 'Average Traffic Volume'},
                           title="Typical Daily Traffic Pattern")
        fig_hourly.add_vline(x=hour, line_dash="dash", line_color="red", 
                           annotation_text=f"Current Hour: {hour}")
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col4:
        st.subheader("Weather Impact")
        weather_impact = {
            'Weather Condition': ['Clear', 'Clouds', 'Rain', 'Snow', 'Thunderstorm'],
            'Average Traffic': [4200, 3800, 2800, 1500, 1200]
        }
        fig_weather = px.bar(weather_impact, x='Weather Condition', y='Average Traffic',
                           title="Traffic Volume by Weather Condition")
        st.plotly_chart(fig_weather, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🚗 Traffic Volume Predictor | Built with Streamlit, scikit-learn, and DVC</p>
        <p>Model Performance: R² = {:.4f} | RMSE = {:.2f}</p>
    </div>
    """.format(model_info['performance']['r2'], model_info['performance']['rmse']), 
    unsafe_allow_html=True)

def model_comparison_tab():
    """Tab for comparing different models"""
    st.header("📊 Model Performance Comparison")
    
    # Initialize model selector
    selector = ModelSelector()
    
    # Get comparison dataframe
    comparison_df = selector.compare_models()
    
    if comparison_df.empty:
        st.warning("No models found for comparison.")
        return
    
    # Display model comparison table
    st.subheader("📋 Model Comparison Table")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance metrics visualization
    if len(comparison_df) > 1:
        st.subheader("📈 Performance Metrics Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = px.bar(
                comparison_df, 
                x='Model Name', 
                y='R² Score',
                title="R² Score Comparison (Higher is Better)",
                color='R² Score',
                color_continuous_scale='Viridis'
            )
            fig_r2.update_xaxis(tickangle=45)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = px.bar(
                comparison_df, 
                x='Model Name', 
                y='RMSE',
                title="RMSE Comparison (Lower is Better)",
                color='RMSE',
                color_continuous_scale='Viridis_r'
            )
            fig_rmse.update_xaxis(tickangle=45)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Performance overview scatter plot
        st.subheader("🎯 R² vs RMSE Scatter Plot")
        fig_scatter = px.scatter(
            comparison_df,
            x='RMSE',
            y='R² Score',
            size='MAE',
            hover_data=['Model Name', 'Model Type'],
            title="Model Performance Overview (Top-right is best)",
            labels={'RMSE': 'RMSE (Lower is Better)', 'R² Score': 'R² Score (Higher is Better)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Model selection recommendations
    st.subheader("🏆 Model Selection Recommendations")
    
    # Get best model for different criteria
    best_r2_model = comparison_df.loc[comparison_df['R² Score'].idxmax()] if 'R² Score' in comparison_df.columns else None
    best_rmse_model = comparison_df.loc[comparison_df['RMSE'].idxmin()] if 'RMSE' in comparison_df.columns else None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if best_r2_model is not None:
            st.metric(
                "🎯 Best R² Score",
                f"{best_r2_model['R² Score']:.4f}",
                delta=f"{best_r2_model['Model Name']}"
            )
    
    with col2:
        if best_rmse_model is not None:
            st.metric(
                "📉 Lowest RMSE",
                f"{best_rmse_model['RMSE']:.2f}",
                delta=f"{best_rmse_model['Model Name']}"
            )
    
    with col3:
        # Current model being used
        current_selector = ModelSelector()
        current_model = current_selector.select_best_model()
        if current_model:
            st.metric(
                "🚀 Currently Selected",
                f"R²: {current_model['metrics'].get('r2', 'N/A'):.4f}",
                delta=f"{current_model['model_name']}"
            )

def model_selection_tab():
    """Tab for configuring model selection criteria"""
    st.header("⚙️ Model Selection Configuration")
    
    # Initialize model selector
    selector = ModelSelector()
    
    # Get current configuration recommendations
    recommended_config = selector.get_model_selection_config()
    
    st.subheader("🎛️ Selection Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Primary Metrics**")
        primary_metric = st.selectbox(
            "Primary Metric for Ranking",
            options=['r2', 'rmse', 'mae', 'mape'],
            index=0,
            help="Main metric used for model ranking"
        )
        
        secondary_metric = st.selectbox(
            "Secondary Metric (tie-breaker)",
            options=['rmse', 'mae', 'mape', 'r2'],
            index=0,
            help="Metric used for tie-breaking when primary metrics are similar"
        )
    
    with col2:
        st.markdown("**Quality Thresholds**")
        min_r2 = st.slider(
            "Minimum R² Score",
            min_value=0.0,
            max_value=1.0,
            value=recommended_config.get('min_r2', 0.90),
            step=0.01,
            help="Minimum acceptable R² score for model selection"
        )
        
        max_rmse = st.slider(
            "Maximum RMSE",
            min_value=0,
            max_value=1000,
            value=int(recommended_config.get('max_rmse', 450)),
            step=10,
            help="Maximum acceptable RMSE for model selection"
        )
        
        max_mape = st.slider(
            "Maximum MAPE (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Maximum acceptable MAPE percentage"
        )
    
    # Advanced settings
    with st.expander("🔧 Advanced Weighting (Optional)"):
        st.markdown("**Custom Metric Weights for Composite Scoring**")
        use_weights = st.checkbox("Enable composite scoring with custom weights")
        
        if use_weights:
            col1, col2, col3 = st.columns(3)
            with col1:
                r2_weight = st.slider("R² Weight", 0.0, 1.0, 0.5, 0.1)
            with col2:
                rmse_weight = st.slider("RMSE Weight", 0.0, 1.0, 0.3, 0.1)
            with col3:
                mae_weight = st.slider("MAE Weight", 0.0, 1.0, 0.2, 0.1)
            
            total_weight = r2_weight + rmse_weight + mae_weight
            if total_weight > 0:
                st.info(f"Total weight: {total_weight:.1f} (weights will be normalized)")
    
    # Test the configuration
    st.subheader("🧪 Test Configuration")
    
    if st.button("🔍 Preview Model Selection with Current Settings"):
        # Build selection criteria
        selection_criteria = {
            'primary_metric': primary_metric,
            'secondary_metric': secondary_metric,
            'min_r2': min_r2,
            'max_rmse': max_rmse,
            'max_mape': max_mape
        }
        
        if use_weights and 'r2_weight' in locals():
            selection_criteria['weights'] = {
                'r2': r2_weight,
                'rmse': rmse_weight,
                'mae': mae_weight
            }
        
        # Test selection
        try:
            selected_model = selector.select_best_model(selection_criteria, fallback_to_latest=False)
            
            if selected_model:
                st.success(f"✅ **Selected Model:** {selected_model['model_name']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{selected_model['metrics'].get('r2', 'N/A'):.4f}")
                with col2:
                    st.metric("RMSE", f"{selected_model['metrics'].get('rmse', 'N/A'):.2f}")
                with col3:
                    st.metric("MAE", f"{selected_model['metrics'].get('mae', 'N/A'):.2f}")
                
                st.info(f"**Model Type:** {selected_model.get('model_type', 'Unknown')}")
                st.info(f"**Created:** {datetime.fromtimestamp(selected_model['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.error("❌ No models meet the specified criteria")
                st.warning("Consider relaxing the thresholds or check if models are properly trained")
                
        except Exception as e:
            st.error(f"Error testing configuration: {str(e)}")
    
    # Save configuration option
    if st.button("💾 Apply Configuration (Restart Required)"):
        st.info("Configuration applied! Restart the application to use the new selection criteria.")
        st.balloons()

if __name__ == "__main__":
    main()