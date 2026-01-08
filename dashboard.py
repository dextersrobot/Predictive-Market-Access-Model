"""
Market Access Predictor Dashboard

Interactive Streamlit dashboard for:
1. Exploring the training data
2. Viewing model performance
3. Making predictions on new drugs
4. Understanding feature importance

Run with: streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Page config
st.set_page_config(
    page_title="Market Access Predictor",
    page_icon="💊",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


@st.cache_data
def load_data():
    """Load processed data."""
    features_path = DATA_DIR / "features" / "engineered_features.csv"
    if features_path.exists():
        return pd.read_csv(features_path)
    return None


@st.cache_data
def load_model_results():
    """Load model training results."""
    results_path = MODELS_DIR / "model_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_model(model_name: str):
    """Load a trained model."""
    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_resource
def load_scaler():
    """Load the feature scaler."""
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None


def main():
    st.title("💊 Market Access Predictor")
    st.markdown("""
    Predict coverage barriers for newly approved drugs based on regulatory, 
    clinical, and economic features.
    """)
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Overview", "Data Explorer", "Model Performance", "Make Predictions", "Feature Importance"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Make Predictions":
        show_predictions()
    elif page == "Feature Importance":
        show_feature_importance()


def show_overview():
    """Show project overview."""
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Sources")
        st.markdown("""
        - **FDA Approvals**: Drug approval data from openFDA
        - **CMS Coverage**: National Coverage Determinations
        - **Orphan Drugs**: FDA orphan drug designations
        - **NADAC Pricing**: National drug acquisition costs
        """)
    
    with col2:
        st.subheader("🎯 Prediction Target")
        st.markdown("""
        The model predicts whether a drug will face **high coverage barriers**
        based on:
        - Regulatory pathway (priority review, orphan status)
        - Clinical characteristics (therapeutic area, dosage form)
        - Economic factors (price tier, market size)
        - Timing (approval year, quarter)
        """)
    
    # Load and display data stats
    df = load_data()
    if df is not None:
        st.subheader("📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Features", len([c for c in df.columns if c not in ['target_high_barrier', 'coverage_difficulty_score']]))
        
        if 'target_high_barrier' in df.columns:
            col3.metric("High Barrier %", f"{df['target_high_barrier'].mean()*100:.1f}%")
        
        if 'approval_year' in df.columns:
            year_min = int(df['approval_year'].min())
            year_max = int(df['approval_year'].max())
            col4.metric("Year Range", f"{year_min}-{year_max}")
    else:
        st.warning("No data loaded. Run the data collection pipeline first.")


def show_data_explorer():
    """Show data exploration interface."""
    st.header("Data Explorer")
    
    df = load_data()
    if df is None:
        st.warning("No data available. Run the pipeline first.")
        return
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'therapeutic_area' in df.columns:
            areas = ['All'] + sorted(df['therapeutic_area'].unique().tolist())
            selected_area = st.selectbox("Therapeutic Area", areas)
            if selected_area != 'All':
                df = df[df['therapeutic_area'] == selected_area]
    
    with col2:
        if 'approval_year' in df.columns:
            years = sorted(df['approval_year'].dropna().unique())
            year_range = st.slider(
                "Approval Year",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(int(min(years)), int(max(years)))
            )
            df = df[(df['approval_year'] >= year_range[0]) & (df['approval_year'] <= year_range[1])]
    
    with col3:
        if 'is_orphan' in df.columns:
            orphan_filter = st.selectbox("Orphan Status", ['All', 'Orphan Only', 'Non-Orphan Only'])
            if orphan_filter == 'Orphan Only':
                df = df[df['is_orphan'] == True]
            elif orphan_filter == 'Non-Orphan Only':
                df = df[df['is_orphan'] == False]
    
    st.write(f"Showing {len(df)} records")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if 'therapeutic_area' in df.columns:
            fig = px.histogram(
                df, x='therapeutic_area',
                title='Distribution by Therapeutic Area',
                color='therapeutic_area'
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'approval_year' in df.columns:
            yearly = df.groupby('approval_year').size().reset_index(name='count')
            fig = px.line(
                yearly, x='approval_year', y='count',
                title='Approvals by Year',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Coverage difficulty distribution
    if 'coverage_difficulty_score' in df.columns:
        fig = px.histogram(
            df, x='coverage_difficulty_score',
            nbins=30,
            title='Coverage Difficulty Score Distribution',
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Data Sample")
    display_cols = [
        'brand_name', 'generic_name', 'therapeutic_area', 'approval_year',
        'is_orphan', 'is_priority_review', 'coverage_difficulty_score'
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols].head(100))


def show_model_performance():
    """Show model performance metrics."""
    st.header("Model Performance")
    
    results = load_model_results()
    if results is None:
        st.warning("No model results available. Train models first.")
        return
    
    # Model comparison
    st.subheader("Model Comparison")
    
    metrics_df = pd.DataFrame([
        {
            'Model': name.replace('_', ' ').title(),
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1 Score': r['f1'],
            'ROC AUC': r['roc_auc']
        }
        for name, r in results.items()
    ])
    
    # Highlight best model
    st.dataframe(
        metrics_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']),
        use_container_width=True
    )
    
    # Bar chart comparison
    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric]
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis_title='Score',
        yaxis_range=[0, 1]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(len(results))
    
    for i, (name, r) in enumerate(results.items()):
        with cols[i]:
            cm = np.array(r['confusion_matrix'])
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Low Barrier', 'High Barrier'],
                y=['Low Barrier', 'High Barrier'],
                title=name.replace('_', ' ').title(),
                color_continuous_scale='Blues',
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)


def show_predictions():
    """Interface for making predictions on new drugs."""
    st.header("Make Predictions")
    
    model = load_model('random_forest')
    scaler = load_scaler()
    
    if model is None or scaler is None:
        st.warning("Models not trained yet. Run the training pipeline first.")
        return
    
    st.markdown("Enter drug characteristics to predict coverage barriers:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regulatory Features")
        is_priority = st.checkbox("Priority Review", value=False)
        is_orphan = st.checkbox("Orphan Designation", value=False)
        is_nme = st.checkbox("New Molecular Entity", value=False)
        is_accelerated = st.checkbox("Accelerated/Breakthrough", value=False)
        
        st.subheader("Clinical Features")
        therapeutic_area = st.selectbox(
            "Therapeutic Area",
            ['oncology', 'immunology', 'neurology', 'cardiology', 
             'infectious', 'rare_disease', 'diabetes', 'respiratory', 'other']
        )
        is_biologic = st.checkbox("Biologic (BLA)", value=False)
        is_complex = st.checkbox("Complex Dosage (Injectable)", value=False)
    
    with col2:
        st.subheader("Economic Features")
        price_tier = st.slider("Price Tier", 0, 5, 2, 
                               help="0=Unknown, 1=Low, 5=Very High")
        market_size = st.slider("Market Size Proxy", 0, 2, 1,
                                help="0=Small, 1=Medium, 2=Large")
        
        st.subheader("Timing")
        approval_year = st.slider("Approval Year", 2020, 2025, 2024)
        approval_quarter = st.selectbox("Approval Quarter", [1, 2, 3, 4])
    
    # Make prediction
    if st.button("Predict Coverage Barriers", type="primary"):
        # Prepare features
        area_encoding = {
            'oncology': 0, 'immunology': 1, 'neurology': 2, 'cardiology': 3,
            'infectious': 4, 'rare_disease': 5, 'diabetes': 6, 'respiratory': 7, 'other': 8
        }
        
        features = {
            'is_priority_review': int(is_priority),
            'is_orphan': int(is_orphan),
            'is_new_molecular_entity': int(is_nme),
            'likely_accelerated': int(is_accelerated),
            'regulatory_advantage_score': sum([is_priority, is_orphan, is_nme]),
            'therapeutic_area_encoded': area_encoding.get(therapeutic_area, 8),
            'is_high_value_area': int(therapeutic_area in ['oncology', 'rare_disease', 'immunology']),
            'is_complex_dosage': int(is_complex),
            'is_oral_simple': int(not is_complex),
            'is_biologic': int(is_biologic),
            'log_launch_price': np.log1p(price_tier * 100),
            'price_tier': price_tier,
            'market_size_proxy': market_size,
            'years_since_2010': approval_year - 2010,
            'month_sin': np.sin(2 * np.pi * (approval_quarter * 3) / 12),
            'month_cos': np.cos(2 * np.pi * (approval_quarter * 3) / 12),
            'is_q4_approval': int(approval_quarter == 4),
            'is_election_year': int(approval_year in [2020, 2024])
        }
        
        # Create dataframe and scale
        feature_df = pd.DataFrame([features])
        
        # Get expected features from model
        try:
            X_scaled = scaler.transform(feature_df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ HIGH COVERAGE BARRIERS PREDICTED")
                else:
                    st.success("✅ LOW COVERAGE BARRIERS PREDICTED")
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{max(probability)*100:.1f}%",
                    delta=None
                )
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={'text': "Probability of High Barriers"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.subheader("Interpretation")
            factors = []
            if is_priority:
                factors.append("✅ Priority review suggests unmet medical need")
            if is_orphan:
                factors.append("✅ Orphan status typically means fewer barriers")
            if price_tier >= 4:
                factors.append("⚠️ High price tier may trigger payer scrutiny")
            if is_biologic:
                factors.append("⚠️ Biologics face more complex coverage decisions")
            if therapeutic_area in ['oncology', 'rare_disease']:
                factors.append("✅ High-value therapeutic area tends to have faster coverage")
            
            for f in factors:
                st.markdown(f)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


def show_feature_importance():
    """Show feature importance analysis."""
    st.header("Feature Importance")
    
    # Load random forest model for feature importance
    model = load_model('random_forest')
    
    if model is None:
        st.warning("Model not available. Train models first.")
        return
    
    # Load feature metadata
    meta_path = DATA_DIR / "features" / "feature_metadata.json"
    if not meta_path.exists():
        st.warning("Feature metadata not found.")
        return
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_columns']
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        # Horizontal bar chart
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (Random Forest)',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature categories
        st.subheader("Feature Categories")
        
        categories = {
            'Regulatory': ['is_priority_review', 'is_orphan', 'is_new_molecular_entity', 
                          'likely_accelerated', 'regulatory_advantage_score'],
            'Clinical': ['therapeutic_area_encoded', 'is_high_value_area', 
                        'is_complex_dosage', 'is_oral_simple', 'is_biologic'],
            'Economic': ['log_launch_price', 'price_tier', 'market_size_proxy'],
            'Temporal': ['years_since_2010', 'month_sin', 'month_cos', 
                        'is_q4_approval', 'is_election_year']
        }
        
        category_importance = {}
        for cat, feats in categories.items():
            cat_feats = [f for f in feats if f in feature_names]
            if cat_feats:
                indices = [feature_names.index(f) for f in cat_feats]
                category_importance[cat] = sum(importance[i] for i in indices)
        
        cat_df = pd.DataFrame([
            {'Category': k, 'Total Importance': v}
            for k, v in category_importance.items()
        ]).sort_values('Total Importance', ascending=False)
        
        fig = px.pie(
            cat_df,
            values='Total Importance',
            names='Category',
            title='Importance by Feature Category'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.subheader("Key Insights")
        
        top_features = importance_df.tail(5)['Feature'].tolist()
        
        st.markdown(f"""
        **Top 5 Most Important Features:**
        1. {top_features[4] if len(top_features) > 4 else 'N/A'}
        2. {top_features[3] if len(top_features) > 3 else 'N/A'}
        3. {top_features[2] if len(top_features) > 2 else 'N/A'}
        4. {top_features[1] if len(top_features) > 1 else 'N/A'}
        5. {top_features[0] if len(top_features) > 0 else 'N/A'}
        
        **Interpretation for Market Access Strategy:**
        - Focus on securing favorable regulatory designations (priority review, orphan status)
        - Price positioning relative to alternatives is critical
        - Therapeutic area context shapes payer expectations
        - Timing of approval can influence coverage negotiations
        """)


if __name__ == "__main__":
    main()
