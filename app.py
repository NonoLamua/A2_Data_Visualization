import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, confusion_matrix, 
    classification_report, roc_curve, 
    auc, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import pickle
import io
import base64
from typing import Tuple, Dict, List, Any, Optional, Union

# Set page configuration
st.set_page_config(
    page_title="ML Model Trainer",
    page_icon="🧠",
    layout="wide",
)

# Initialize session state variables if they don't exist
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'is_classification' not in st.session_state:
    st.session_state.is_classification = False
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# Cache function to load seaborn datasets
@st.cache_data
def load_seaborn_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a dataset from seaborn"""
    if dataset_name == "iris":
        return sns.load_dataset("iris")
    elif dataset_name == "titanic":
        return sns.load_dataset("titanic")
    elif dataset_name == "diamonds":
        return sns.load_dataset("diamonds")
    elif dataset_name == "tips":
        return sns.load_dataset("tips")
    elif dataset_name == "planets":
        return sns.load_dataset("planets")
    else:
        return pd.DataFrame()

def identify_variable_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numerical and categorical columns in a DataFrame"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'category', 'bool']).columns.tolist()
    return numerical_cols, categorical_cols

def preprocess_data(df: pd.DataFrame, features: List[str], target: str, is_classification: bool) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the dataset for machine learning"""
    # Select features and target
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle target encoding for classification
    if is_classification:
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state.label_encoders['target'] = le
    
    # Handle categorical features in X
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
    
    # Apply one-hot encoding to categorical features
    if len(categorical_features) > 0:
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # Handle missing values (add this part)
    # For numerical features, impute with median
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0 and X[numeric_cols].isnull().any().any():
        imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
    # Check if there are any remaining NaN values (e.g., in one-hot encoded columns)
    if X.isnull().any().any():
        # Fill remaining NaNs with 0
        X = X.fillna(0)
    
    return X, y

def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    model_type: str,
    params: Dict[str, Any]
) -> Any:
    """Train a model based on the selected type and parameters"""
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest Regressor":
        model = RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None) if params.get("max_depth", 0) > 0 else None,
            random_state=42
        )
    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 100),
            random_state=42
        )
    elif model_type == "Random Forest Classifier":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None) if params.get("max_depth", 0) > 0 else None,
            random_state=42
        )
    else:
        st.error("Invalid model type")
        return None

    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    is_classification: bool
) -> Dict[str, Any]:
    """Evaluate the trained model and return results"""
    results = {}
    
    if is_classification:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        results["classification_report"] = classification_report(y_test, y_pred)
        
        # For ROC curve
        if len(np.unique(y_test)) == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            results["fpr"] = fpr
            results["tpr"] = tpr
            results["roc_auc"] = auc(fpr, tpr)
        
        if hasattr(model, 'feature_importances_'):
            results["feature_importance"] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results["feature_importance"] = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    else:
        # Regression
        y_pred = model.predict(X_test)
        results["mse"] = mean_squared_error(y_test, y_pred)
        results["r2"] = r2_score(y_test, y_pred)
        results["residuals"] = y_test - y_pred
        
        if hasattr(model, 'feature_importances_'):
            results["feature_importance"] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results["feature_importance"] = model.coef_
    
    return results

def get_model_download_link(model, filename="model.pkl"):
    """Generate a download link for the trained model"""
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download trained model</a>'

def main():
    st.title("🧠 Machine Learning Model Trainer")
    
    # Sidebar for dataset selection and upload
    with st.sidebar:
        st.header("1. Dataset Selection")
        
        dataset_option = st.radio(
            "Choose a dataset source:",
            ["Built-in Datasets", "Upload Your Own CSV"]
        )
        
        if dataset_option == "Built-in Datasets":
            seaborn_datasets = ["iris", "titanic", "diamonds", "tips", "planets"]
            selected_dataset = st.selectbox(
                "Select a Seaborn dataset",
                seaborn_datasets,
                index=0,
                key="seaborn_dataset_select"
            )
            
            if st.button("Load Dataset", key="load_seaborn_btn"):
                st.session_state.dataset = load_seaborn_dataset(selected_dataset)
                st.success(f"Loaded {selected_dataset} dataset!")
                
        else:  # Upload dataset
            uploaded_file = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help="Upload your own dataset as a CSV file"
            )
            
            if uploaded_file is not None:
                try:
                    st.session_state.dataset = pd.read_csv(uploaded_file)
                    st.success("Dataset uploaded successfully!")
                except Exception as e:
                    st.error(f"Error loading the dataset: {e}")
    
    # Main content area
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        # Dataset preview
        st.header("Dataset Preview")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head())
        
        # Data summary
        if st.checkbox("Show Data Summary"):
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Numerical Columns Summary")
                st.write(df.describe())
            with col2:
                st.write("Data Types")
                st.write(df.dtypes)
            
            st.write("Missing Values")
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing Count']
            missing_data['Missing Percentage'] = missing_data['Missing Count'] / len(df) * 100
            st.write(missing_data)
        
        # Feature selection
        st.header("2. Feature Selection")
        numerical_cols, categorical_cols = identify_variable_types(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Target Variable")
            target_options = numerical_cols + categorical_cols
            target = st.selectbox("Select target variable", target_options)
            
            if target in numerical_cols:
                st.session_state.problem_type = st.radio(
                    "Problem Type",
                    ["Regression", "Classification"],
                    index=0
                )
                st.session_state.is_classification = st.session_state.problem_type == "Classification"
            else:
                st.session_state.problem_type = "Classification"
                st.session_state.is_classification = True
            
            st.session_state.target = target
        
        with col2:
            st.subheader("Feature Variables")
            available_features = [col for col in df.columns if col != target]
            
            available_num_features = [col for col in numerical_cols if col != target]
            available_cat_features = [col for col in categorical_cols if col != target]
            
            selected_num_features = st.multiselect(
                "Select numerical features",
                available_num_features,
                default=available_num_features
            )
            
            selected_cat_features = st.multiselect(
                "Select categorical features",
                available_cat_features,
                default=available_cat_features
            )
            
            st.session_state.features = selected_num_features + selected_cat_features
        
        # Model configuration
        st.header("3. Model Configuration")
        
        with st.form(key="model_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Selection")
                
                if st.session_state.is_classification:
                    model_options = ["Logistic Regression", "Random Forest Classifier"]
                else:
                    model_options = ["Linear Regression", "Random Forest Regressor"]
                
                model_type = st.selectbox(
                    "Select model type",
                    model_options
                )
                
                test_size = st.slider("Test size (%)", 10, 50, 20)
                random_state = 42
            
            with col2:
                st.subheader("Model Parameters")
                
                params = {}
                if model_type in ["Random Forest Regressor", "Random Forest Classifier"]:
                    params["n_estimators"] = st.number_input(
                        "Number of trees",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10
                    )
                    params["max_depth"] = st.number_input(
                        "Max depth (0 for unlimited)",
                        min_value=0,
                        max_value=50,
                        value=0,
                        step=1
                    )
                elif model_type == "Logistic Regression":
                    params["C"] = st.number_input(
                        "Regularization parameter (C)",
                        min_value=0.01,
                        max_value=10.0,
                        value=1.0,
                        step=0.1
                    )
                    params["max_iter"] = st.number_input(
                        "Maximum iterations",
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100
                    )
            
            # Submit button
            submit_button = st.form_submit_button(label="Fit Model")
            
            if submit_button:
                if not st.session_state.features:
                    st.error("Please select at least one feature")
                else:
                    with st.spinner("Training model..."):
                        try:
                            # Preprocess data - this is the improved part to fix the error
                            X, y = preprocess_data(
                                df, 
                                st.session_state.features, 
                                st.session_state.target,
                                st.session_state.is_classification
                            )
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size/100, random_state=random_state
                            )
                            
                            # Save test data for later
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            
                            # Train model
                            st.session_state.model_type = model_type
                            st.session_state.trained_model = train_model(
                                X_train, y_train, model_type, params
                            )
                            
                            # Evaluate model
                            st.session_state.model_results = evaluate_model(
                                st.session_state.trained_model,
                                X_test,
                                y_test,
                                st.session_state.is_classification
                            )
                            
                            st.success("Model trained successfully!")
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                            st.error("Please check your data and model configuration.")
        
        # Results section (only show if model has been trained)
        if st.session_state.trained_model is not None:
            st.header("4. Model Results")
            
            results = st.session_state.model_results
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                
                if st.session_state.is_classification:
                    if 'accuracy' in results:
                        st.metric("Accuracy", f"{results['accuracy']:.4f}")
                    else:
                        st.warning("Accuracy metric not available")
                    
                    # Classification report
                    if 'classification_report' in results:
                        st.text("Classification Report")
                        st.text(results['classification_report'])
                    else:
                        st.warning("Classification report not available")
                else:
                    st.metric("Mean Squared Error", f"{results['mse']:.4f}")
                    st.metric("R² Score", f"{results['r2']:.4f}")
            
            with col2:
                st.subheader("Feature Importance")
                
                # Plot feature importance
                if "feature_importance" in results:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    features = X_test.columns
                    importance = results["feature_importance"]
                    
                    # Sort by importance
                    indices = np.argsort(importance)
                    
                    # Display only top 15 features if there are many
                    if len(features) > 15:
                        indices = indices[-15:]
                    
                    plt.barh(range(len(indices)), importance[indices])
                    plt.yticks(range(len(indices)), [features[i] for i in indices])
                    plt.xlabel('Feature Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Feature importance not available for this model.")
            
            # Additional visualizations based on model type
            st.subheader("Model Visualizations")
            
            if st.session_state.is_classification:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confusion Matrix
                    st.write("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        results["confusion_matrix"],
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        ax=ax
                    )
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig)
                
                with col2:
                    # ROC Curve (only for binary classification)
                    if "roc_auc" in results:
                        st.write(f"ROC Curve (AUC = {results['roc_auc']:.4f})")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plt.plot(
                            results["fpr"],
                            results["tpr"],
                            label=f'ROC curve (area = {results["roc_auc"]:.4f})'
                        )
                        plt.plot([0, 1], [0, 1], 'k--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic')
                        plt.legend(loc="lower right")
                        st.pyplot(fig)
                    else:
                        st.info("ROC curve only available for binary classification.")
            else:
                # Residual plot for regression
                st.write("Residual Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Residual vs Predicted plot
                y_pred = st.session_state.trained_model.predict(X_test)
                
                plt.scatter(y_pred, results["residuals"])
                plt.axhline(y=0, color='r', linestyle='-')
                plt.xlabel('Predicted values')
                plt.ylabel('Residuals')
                plt.title('Residuals vs. Predicted Values')
                st.pyplot(fig)
                
                # Residual distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(results["residuals"], kde=True, ax=ax)
                plt.xlabel('Residuals')
                plt.title('Residual Distribution')
                st.pyplot(fig)
            
            # Model download section
            st.header("5. Export Model")
            
            st.markdown(
                get_model_download_link(st.session_state.trained_model, 
                                      f"{st.session_state.model_type.replace(' ', '_').lower()}.pkl"), 
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
