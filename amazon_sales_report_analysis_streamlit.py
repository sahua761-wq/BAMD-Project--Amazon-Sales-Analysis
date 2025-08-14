# Amazon Sales SKU Classification Streamlit App
import pandas as pd
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(page_title="Amazon SKU Classifier", page_icon="📦", layout="wide")

from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

from model_training import X1, X2, X3, X4, N_df, S_df, E_df, W_df, lr_model, lr_model2, lr_model3, lr_model4, df


all_regional_data = pd.concat([
    N_df.assign(Region='North'),
    S_df.assign(Region='South'),
    E_df.assign(Region='East'),
    W_df.assign(Region='West')
], ignore_index=True)


# Title and description
st.title("📦 Amazon Sales SKU Classifier")
st.subheader("Predicting SKU Performance in Indian Regions")

# Add description
st.markdown("""
This application classifies SKUs based on whether they will sell more than 2 quantities in a specific region.
- **Class 0**: SKU will sell 2 or fewer quantities
- **Class 1**: SKU will sell more than 2 quantities
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Choose Section", ["Home", "SKU Classification", "Data Analysis"])

# Home Section
if nav == 'Home':
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## About This Application

        This application helps predict SKU performance in different regions of India based on various product characteristics.

        **Features:**
        - Region-specific predictions for North, South, East, and West India
        - Classification based on multiple product attributes
        - Interactive prediction interface
        - Data visualization capabilities

        **How to Use:**
        1. Navigate to the "SKU Classification" section
        2. Select your target region
        3. Input product characteristics
        4. Get prediction results instantly
        """)

    with col2:
        # Display some statistics
        st.markdown("### Quick Stats")
        st.metric("Total Regions", "4")
        st.metric("Product Categories", "9")
        st.metric("Classification Classes", "2")

        # Calculate percentages by region
        region_stats = all_regional_data.groupby(['Region', 'Volume_Class']).size().unstack(fill_value=0)
        region_percentages = region_stats.div(region_stats.sum(axis=1), axis=0) * 100

        fig = go.Figure(data=[
            go.Bar(name='Class 0 (≤2 quantities)',
                  x=region_percentages.index,
                  y=region_percentages[0] if 0 in region_percentages.columns else [0]*len(region_percentages)),
            go.Bar(name='Class 1 (>2 quantities)',
                  x=region_percentages.index,
                  y=region_percentages[1] if 1 in region_percentages.columns else [0]*len(region_percentages))
        ])
        fig.update_layout(
            title="Actual Volume Class Distribution by Region (%)",
            barmode='stack',
            height=400,
            yaxis_title="Percentage (%)"
        )
        st.plotly_chart(fig, use_container_width=True)

# SKU Classification Section
elif nav == 'SKU Classification':
    st.header("🎯 SKU Performance Prediction")

    # Region selection
    REGIONS = ["North", "South", "East", "West"]

    st.markdown("### Step 1: Select Target Region")
    selected_region = st.selectbox("Choose the region for prediction:", REGIONS, index=0)

    st.markdown(f"**Selected Region:** {selected_region}")
    st.divider()
    if selected_region == 'North':
        model = lr_model
    elif selected_region == 'South':
        model = lr_model2
    elif selected_region == 'West':
        model = lr_model3
    elif selected_region == 'East':
        model = lr_model4

    # Input form
    st.markdown("### Step 2: Enter Product Details")

    col1, col2 = st.columns(2)

    CATEGORIES = ["Blouse", "Bottom", "Dupatta", "Ethnic Dress", "kurta", "Saree", "Set", "Top", "Western Dress"]
    PROMOTION = [0,1]
    FULFILLMENT = ["Amazon", "Merchant"]
    SALES_CHANNEL = ["Amazon.in", "Non-Amazon"]
    SIZES = ['S','M', 'L', 'XL', 'XXL', '3XL', 'XS', 'Free', '4XL', '5XL', '6XL']
    B2B_OPTIONS = [True, False]

    def encode_region_features(region, category, size, b2b, promotion_ids, fulfillment, sales_channel):
        if region == 'North':
            cols = X1.columns  # Import X1 from model_training.py
        elif region == 'South':
            cols = X2.columns
        elif region == 'East':
            cols = X3.columns
        elif region == 'West':
            cols = X4.columns

        # Create DataFrame with one row
        df_input = pd.DataFrame([{
            'Category': category,
            'Size': size,
            'B2B': b2b,
            'Promotion IDs': promotion_ids,
            'Fulfillment': fulfillment,
            'Sales Channel': sales_channel
        }])

        # One-hot encode
        df_encoded = pd.get_dummies(df_input)

        # Add missing columns (set to 0)
        for col in cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Reorder columns to match training
        df_encoded = df_encoded[cols]

        return df_encoded

    with col1:
        category = st.selectbox("Product Category:", CATEGORIES, index=0)
        size = st.selectbox("Size:", SIZES, index=6)  # Default to 'L'
        b2b = st.selectbox("B2B:", B2B_OPTIONS, index=1)  # Default to 'FALSE'

    with col2:
        promotion_ids = st.selectbox("Promotion IDs:", PROMOTION, index=0)
        fulfillment = st.selectbox("Fulfillment:", FULFILLMENT, index=0)  # Default to 'Amazon'
        sales_channel = st.selectbox("Sales Channel:", SALES_CHANNEL, index=0)  # Default to 'Amazon.in'

    st.divider()

    # Prediction section
    st.markdown("### Step 3: Get Prediction")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Predict SKU Performance", type="primary", use_container_width=True):
            # Encode features
            encoded_features = encode_region_features(selected_region, category, size, b2b, promotion_ids, fulfillment, sales_channel)

            # Make prediction (replace with your actual prediction logic)
            prediction = model.predict(encoded_features)[0]
            prediction_proba = model.predict_proba(encoded_features)[0]

            st.success("Prediction Complete!")

            # Display results
            col_result1, col_result2 = st.columns(2)

            with col_result1:
                if prediction == 1:
                    st.success(f"**Prediction: Class {prediction}**")
                    st.write("✅ This SKU is predicted to sell **MORE than 2 quantities** in the selected region.")
                else:
                    st.warning(f"**Prediction: Class {prediction}**")
                    st.write("⚠️ This SKU is predicted to sell **2 or FEWER quantities** in the selected region.")

            with col_result2:
                st.markdown("**Prediction Confidence:**")
                st.write(f"Class 0 probability: {prediction_proba[0]:.2%}")
                st.write(f"Class 1 probability: {prediction_proba[1]:.2%}")

                # Confidence bar
                confidence = max(prediction_proba)
                st.progress(confidence)
                st.write(f"Confidence Level: {confidence:.2%}")

    # Display input summary
    with st.expander("📋 Input Summary"):
        input_data = {
            "Parameter": ["Region", "Category", "Size", "B2B", "Promotion IDs", "Fulfillment", "Sales Channel"],
            "Value": [selected_region, category, size, b2b, promotion_ids, fulfillment, sales_channel]
        }
        input_df = pd.DataFrame(input_data)
        st.table(input_df)

# Data Analysis Section
elif nav == 'Data Analysis':
    st.header("📊 Data Analysis & Insights")

    # Sample visualizations (replace with actual data analysis)
    col1, col2 = st.columns(2)

    with col1:
        # Category distribution
        category_counts = df['Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        

        fig1 = px.bar(category_counts, x='Category', y='Count',
                     title='SKU Distribution by Category',
                     color='Count', color_continuous_scale='viridis')
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Regional performance
        # Count Volume class = 1 from each regional dataframe
        regional_performance = pd.DataFrame({
            'Region': ['North', 'South', 'East', 'West'],
            'High_Performance_SKUs': [
                (N_df['Volume_Class'] == 1).sum(),
                (S_df['Volume_Class'] == 1).sum(),  # Note: S_df not S-df
                (E_df['Volume_Class'] == 1).sum(),
                (W_df['Volume_Class'] == 1).sum()
            ]
        })

        fig2 = px.pie(regional_performance, values='High_Performance_SKUs', names='Region',
                     title='High Performance SKUs by Region (Volume Class = 1)')
        st.plotly_chart(fig2, use_container_width=True)

    # Size distribution
    size_performance = all_regional_data[all_regional_data['Volume_Class'] == 1]['Size'].value_counts().reset_index()
    size_performance.columns = ['Size', 'Class_1_Count']

    fig3 = px.bar(size_performance, x='Size', y='Class_1_Count',
                 title='Class 1 Count by Size',
                 labels={'Class_1_Count': 'Number of Class 1 SKUs'})
    st.plotly_chart(fig3, use_container_width=True)

    # Feature importance
    st.markdown("### Feature Importance")
    def get_feature_importance_from_models(models, feature_names):
        """Extract feature importance from logistic regression models"""

        # Combine coefficients from all regional models
        all_coefficients = []

        for region, model in models.items():
            # Get absolute values of coefficients
            coeffs = np.abs(model.coef_[0])  # model.coef_[0] for binary classification
            all_coefficients.append(coeffs)

        # Average coefficients across all regions
        avg_coefficients = np.mean(all_coefficients, axis=0)

        # Normalize to get relative importance (optional)
        normalized_importance = avg_coefficients / np.sum(avg_coefficients)

        return normalized_importance

    # Use it in your code
    feature_names = ['Category', 'Size', 'B2B', 'Promotion IDs', 'Fulfillment', 'Sales Channel']

    actual_models = {
        'North': lr_model,   # Your North region model
        'South': lr_model2,
        'East': lr_model4,
        'West': lr_model3
    }

    # Training columns per region (from pd.get_dummies)
    region_columns = {
        "North": X1.columns,
        "South": X2.columns,
        "East": X3.columns,
        "West": X4.columns
    }

    def aggregate_feature_importance(model, columns, original_features):
        coeffs = np.abs(model.coef_[0])
        df = pd.DataFrame({'col': columns, 'coef': coeffs})

        aggregated = {}
        for feat in original_features:
            # sum coefficients of all columns that start with the feature name
            aggregated[feat] = df[df['col'].str.contains(feat)]['coef'].sum()

        return pd.DataFrame({
            'Feature': list(aggregated.keys()),
            'Importance': list(aggregated.values())
        })

    # Loop over regions and show feature importance
    for region, model in actual_models.items():
        cols = region_columns[region]  # training columns for this region
        feature_importance = aggregate_feature_importance(model, cols, feature_names)

        st.subheader(f"Feature Importance - {region}")
        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                    title=f"Feature Importance for {region}")
        st.plotly_chart(fig, use_container_width=True)


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Amazon Sales SKU Classifier | Built with Streamlit</p>
    <p>For questions or support, contact your data science team</p>
</div>
""", unsafe_allow_html=True)

