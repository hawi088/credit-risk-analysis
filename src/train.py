"""
Model Training and Tracking for Credit Risk Model
Task 5 - Complete Implementation

Author: Kerod, Mahbubah, Feven
Date: 31 May 2026
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import warnings
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing import process_raw_data

warnings.filterwarnings('ignore')

# Set MLflow tracking URI to local directory
mlflow.set_tracking_uri("file:./mlruns")


def load_and_prepare_data():
    """
    Load the processed dataset with target variable.
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load raw data
    df_raw = pd.read_csv('data/raw/data.csv')
    print(f"Raw data: {df_raw.shape[0]:,} transactions")
    
    # Process features with target
    df = process_raw_data(df_raw)
    
    # Separate features and target
    target_col = 'is_high_risk'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().to_string()}")
    
    return X, y


def prepare_categorical_features(X):
    """
    Convert categorical features to numeric using one-hot encoding.
    """
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"\nEncoding categorical features: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and log results to MLflow.
    """
    results = {}
    
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
        }
    }
    
    for model_name, model_config in models.items():
        print("\n" + "=" * 60)
        print(f"TRAINING: {model_name}")
        print("=" * 60)
        
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            print(f"Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"Best CV Score (ROC-AUC): {best_score:.4f}")
            print(f"Best Parameters: {best_params}")
            
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_roc_auc", best_score)
            
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"\nTest Set Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_metric("test_roc_auc", roc_auc)
            
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
            
            mlflow.sklearn.log_model(best_model, model_name)
            
            results[model_name] = {
                'model': best_model,
                'best_params': best_params,
                'cv_score': best_score,
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'test_roc_auc': roc_auc
            }
    
    return results


def select_best_model(results):
    """
    Select the best model based on ROC-AUC score.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame([
        {
            'Model': name,
            'CV ROC-AUC': metrics['cv_score'],
            'Test Accuracy': metrics['test_accuracy'],
            'Test Precision': metrics['test_precision'],
            'Test Recall': metrics['test_recall'],
            'Test F1': metrics['test_f1'],
            'Test ROC-AUC': metrics['test_roc_auc']
        }
        for name, metrics in results.items()
    ]).sort_values('Test ROC-AUC', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    best_roc_auc = comparison_df.iloc[0]['Test ROC-AUC']
    
    print(f"\n" + "=" * 60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test ROC-AUC: {best_roc_auc:.4f}")
    print("=" * 60)
    
    return best_model_name, best_model, comparison_df


def register_best_model(best_model, best_model_name):
    """
    Register the best model in MLflow Model Registry.
    """
    print("\n" + "=" * 60)
    print("REGISTERING BEST MODEL")
    print("=" * 60)
    
    with mlflow.start_run(run_name=f"REGISTER_{best_model_name}"):
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            registered_model_name="CreditRisk_Best_Model"
        )
        
        print(f"Model registered as: CreditRisk_Best_Model")
        print(f"Model type: {best_model_name}")
    
    return True


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("CREDIT RISK MODEL TRAINING PIPELINE")
    print("Task 5: Model Training and Tracking")
    print("=" * 60)
    
    X, y = load_and_prepare_data()
    
    X = prepare_categorical_features(X)
    
    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train target distribution:\n{y_train.value_counts().to_string()}")
    print(f"Test target distribution:\n{y_test.value_counts().to_string()}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    results = train_models(X_train, X_test, y_train, y_test)
    
    best_model_name, best_model, comparison_df = select_best_model(results)
    
    register_best_model(best_model, best_model_name)
    
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print("\nModel comparison results saved to 'model_comparison_results.csv'")
    
    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    print("Best model saved to 'best_model.pkl'")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    
    return results, best_model, comparison_df


if __name__ == "__main__":
    results, best_model, comparison_df = main()