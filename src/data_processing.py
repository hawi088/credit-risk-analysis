"""
Feature Engineering Pipeline for Credit Risk Model
Task 3 - Complete Implementation
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Try to import xverse for WoE transformation
try:
    from xverse.transformer import WOE
    XVERSE_AVAILABLE = True
except ImportError:
    print("Warning: xverse not installed. Run: pip install xverse")
    XVERSE_AVAILABLE = False


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level data to customer-level features.
    
    Creates:
    - Frequency: Number of transactions per customer
    - Monetary: Total transaction amount per customer
    - AvgAmount: Average transaction amount
    - StdAmount: Standard deviation of transaction amounts
    - MinAmount: Minimum transaction amount
    - MaxAmount: Maximum transaction amount
    - Recency: Days since last transaction
    - CreditRatio: Proportion of positive vs negative transactions
    """
    
    def __init__(self):
        self.snapshot_date = None
        
    def fit(self, X, y=None):
        if 'TransactionStartTime' in X.columns:
            self.snapshot_date = pd.to_datetime(X['TransactionStartTime']).max()
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Customer-level aggregations
        customer_features = df.groupby('CustomerId').agg({
            'TransactionId': 'count',  # Frequency
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],  # Monetary & variability
            'TransactionStartTime': 'max'  # For recency
        }).round(2)
        
        customer_features.columns = [
            'Frequency', 'Monetary', 'AvgAmount', 'StdAmount', 
            'MinAmount', 'MaxAmount', 'LastTransactionDate'
        ]
        customer_features = customer_features.reset_index()
        
        # Calculate Recency (days since last transaction)
        if self.snapshot_date:
            customer_features['Recency'] = (
                self.snapshot_date - customer_features['LastTransactionDate']
            ).dt.days
        
        # Calculate Credit Ratio (proportion of positive transactions)
        credit_ratio = df.groupby('CustomerId').apply(
            lambda x: (x['Amount'] > 0).sum() / len(x) if len(x) > 0 else 0
        ).reset_index(name='CreditRatio')
        customer_features = customer_features.merge(credit_ratio, on='CustomerId', how='left')
        
        # Average transaction value (absolute)
        avg_value = df.groupby('CustomerId')['Value'].mean().reset_index(name='AvgTransactionValue')
        customer_features = customer_features.merge(avg_value, on='CustomerId', how='left')
        
        # Handle missing StdAmount (customers with single transaction)
        customer_features['StdAmount'] = customer_features['StdAmount'].fillna(0)
        customer_features = customer_features.drop(['LastTransactionDate'], axis=1)
        
        return customer_features


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts time-based features from transaction timestamps.
    
    Creates:
    - TransactionHour: Hour of day (0-23)
    - TransactionDay: Day of month (1-31)
    - TransactionMonth: Month (1-12)
    - TransactionYear: Year
    - TransactionDayOfWeek: Day of week (0=Monday, 6=Sunday)
    - IsWeekend: Binary flag (1=weekend, 0=weekday)
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDay'] = df['TransactionStartTime'].dt.day
            df['TransactionMonth'] = df['TransactionStartTime'].dt.month
            df['TransactionYear'] = df['TransactionStartTime'].dt.year
            df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
            df['IsWeekend'] = (df['TransactionDayOfWeek'] >= 5).astype(int)
        
        return df


class TransactionTimeAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level time features to customer level.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        # Aggregate time features per customer
        time_features = df.groupby('CustomerId').agg({
            'TransactionHour': ['mean', 'std'],
            'TransactionDay': ['mean', 'std'],
            'TransactionMonth': ['mean', 'std'],
            'IsWeekend': 'mean',
            'TransactionDayOfWeek': ['mean', 'std']
        }).round(2)
        
        time_features.columns = [
            'AvgTransactionHour', 'StdTransactionHour',
            'AvgTransactionDay', 'StdTransactionDay',
            'AvgTransactionMonth', 'StdTransactionMonth',
            'WeekendTransactionRatio',
            'AvgDayOfWeek', 'StdDayOfWeek'
        ]
        time_features = time_features.reset_index()
        
        # Fill NaN values for customers with single transaction
        for col in ['StdTransactionHour', 'StdTransactionDay', 'StdTransactionMonth', 'StdDayOfWeek']:
            time_features[col] = time_features[col].fillna(0)
        
        return time_features


class CategoryAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates categorical feature distributions per customer.
    
    Creates:
    - TopProductCategory: Most frequent product category
    - TopChannelId: Most frequent channel
    - ProductCategoryDiversity: Number of unique product categories
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        # Most frequent product category per customer
        if 'ProductCategory' in df.columns:
            most_frequent_category = df.groupby('CustomerId')['ProductCategory'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
            ).reset_index(name='TopProductCategory')
        else:
            most_frequent_category = pd.DataFrame({'CustomerId': df['CustomerId'].unique(), 
                                                    'TopProductCategory': 'unknown'})
        
        # Most frequent channel per customer
        if 'ChannelId' in df.columns:
            most_frequent_channel = df.groupby('CustomerId')['ChannelId'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
            ).reset_index(name='TopChannelId')
        else:
            most_frequent_channel = pd.DataFrame({'CustomerId': df['CustomerId'].unique(), 
                                                   'TopChannelId': 'unknown'})
        
        # Category diversity (number of unique categories)
        if 'ProductCategory' in df.columns:
            category_diversity = df.groupby('CustomerId')['ProductCategory'].nunique().reset_index(
                name='ProductCategoryDiversity')
        else:
            category_diversity = pd.DataFrame({'CustomerId': df['CustomerId'].unique(), 
                                                'ProductCategoryDiversity': 0})
        
        result = most_frequent_category.merge(most_frequent_channel, on='CustomerId', how='outer')
        result = result.merge(category_diversity, on='CustomerId', how='outer')
        
        return result


class FraudFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts fraud-related features per customer."""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        if 'FraudResult' in df.columns:
            fraud_features = df.groupby('CustomerId').agg({
                'FraudResult': ['sum', 'mean']
            }).round(4)
            fraud_features.columns = ['FraudCount', 'FraudRate']
            fraud_features = fraud_features.reset_index()
        else:
            fraud_features = pd.DataFrame({'CustomerId': df['CustomerId'].unique(), 
                                           'FraudCount': 0, 'FraudRate': 0})
        
        return fraud_features


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation to skewed numerical features.
    Handles negative values by shifting.
    """
    
    def __init__(self, features=None):
        self.features = features
        self.skewed_features = ['Monetary', 'Frequency', 'AvgAmount', 'MinAmount', 'MaxAmount']
        self.shift_values = {}
        
    def fit(self, X, y=None):
        # Calculate shift values for each feature
        for feature in self.skewed_features:
            if feature in X.columns:
                min_val = X[feature].min()
                self.shift_values[feature] = -min_val + 1 if min_val < 0 else 0
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        features_to_transform = self.features if self.features else self.skewed_features
        
        for feature in features_to_transform:
            if feature in df.columns:
                shift = self.shift_values.get(feature, 0)
                df[f'{feature}_log'] = np.log1p(df[feature] + shift)
                df = df.drop([feature], axis=1)
        
        return df


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Applies Weight of Evidence (WoE) transformation to categorical features.
    Requires binary target variable.
    """
    
    def __init__(self, target_column=None):
        self.target_column = target_column
        self.woe_transformer = None
        self.categorical_features = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        
        # Check if target column exists
        if self.target_column and self.target_column in df.columns and XVERSE_AVAILABLE:
            # Identify categorical columns
            self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            
            if self.categorical_features:
                print(f"Applying WoE transformation to: {self.categorical_features}")
                X_cat = df[self.categorical_features].copy()
                y_target = df[self.target_column].copy()
                
                self.woe_transformer = WOE()
                self.woe_transformer.fit(X_cat, y_target)
                X_woe = self.woe_transformer.transform(X_cat)
                
                # Add WOE columns and drop original categorical columns
                for col in X_woe.columns:
                    df[f'{col}_WOE'] = X_woe[col]
                df = df.drop(self.categorical_features, axis=1)
                
                # Print IV values for feature selection
                iv_df = self.woe_transformer.iv_df
                print("\nInformation Values (IV) for feature selection:")
                print(iv_df.to_string(index=False))
        
        return df


def create_full_pipeline(target_column=None, apply_woe=False):
    """
    Creates the complete feature engineering pipeline.
    
    Args:
        target_column: Name of target column for WoE transformation
        apply_woe: Whether to apply WoE transformation
    
    Returns:
        Pipeline: Complete sklearn pipeline
    """
    # This is a meta-pipeline since our transformers work sequentially
    # We'll implement as a function that applies transformers in order
    
    pipeline_steps = [
        ('aggregator', CustomerAggregator()),
        ('time_extractor', TimeFeatureExtractor()),
        ('time_aggregator', TransactionTimeAggregator()),
        ('category_aggregator', CategoryAggregator()),
        ('fraud_extractor', FraudFeatureExtractor()),
        ('log_transformer', LogTransformer()),
    ]
    
    if apply_woe and target_column:
        pipeline_steps.append(('woe_transformer', WOETransformer(target_column=target_column)))
    
    # Note: Scaler would be applied after train/test split in Task 5
    # because scaling should be fit on training data only
    
    return pipeline_steps


def process_raw_data(df_raw, target_column=None, apply_woe=False):
    """
    Main function to process raw transaction data into model-ready format.
    
    Args:
        df_raw: Raw transaction DataFrame
        target_column: Name of target column (for WoE)
        apply_woe: Whether to apply WoE transformation
    
    Returns:
        DataFrame: Model-ready features
    """
    print("FEATURE ENGINEERING PIPELINE")
    
    # Step 1: Customer aggregator (RFM features)
    print("\n[1/7] Creating customer-level RFM features...")
    aggregator = CustomerAggregator()
    aggregator.fit(df_raw)
    customer_df = aggregator.transform(df_raw)
    print(f"     → Created {customer_df.shape[1]} features for {customer_df.shape[0]} customers")
    
    # Step 2: Time feature extraction
    print("\n[2/7] Extracting time-based features...")
    time_extractor = TimeFeatureExtractor()
    df_with_time = time_extractor.transform(df_raw)
    print(f"     → Added hour, day, month, year, dayofweek, weekend flags")
    
    # Step 3: Time feature aggregation
    print("\n[3/7] Aggregating time features per customer...")
    time_aggregator = TransactionTimeAggregator()
    time_features = time_aggregator.transform(df_with_time)
    print(f"     → Created {time_features.shape[1] - 1} time aggregate features")
    
    # Step 4: Category aggregation
    print("\n[4/7] Aggregating categorical features...")
    category_aggregator = CategoryAggregator()
    category_features = category_aggregator.transform(df_raw)
    print(f"     → Created {category_features.shape[1] - 1} category features")
    
    # Step 5: Fraud feature extraction
    print("\n[5/7] Extracting fraud features...")
    fraud_extractor = FraudFeatureExtractor()
    fraud_features = fraud_extractor.transform(df_raw)
    print(f"     → Created FraudCount and FraudRate features")
    
    # Step 6: Merge all features
    print("\n[6/7] Merging all features...")
    merged_df = customer_df.merge(time_features, on='CustomerId', how='left')
    merged_df = merged_df.merge(category_features, on='CustomerId', how='left')
    merged_df = merged_df.merge(fraud_features, on='CustomerId', how='left')
    print(f"     → Merged shape: {merged_df.shape}")
    
    # Step 7: Log transformation for skewed features
    print("\n[7/7] Applying log transformation to skewed features...")
    log_transformer = LogTransformer()
    log_transformer.fit(merged_df)
    merged_df = log_transformer.transform(merged_df)
    print(f"     → After log transformation: {merged_df.shape}")
    
    # Drop CustomerId (identifier)
    if 'CustomerId' in merged_df.columns:
        merged_df = merged_df.drop(['CustomerId'], axis=1)
    
    # Fill any remaining missing values
    merged_df = merged_df.fillna(0)
    
    # Optional: Apply WoE transformation
    if apply_woe and target_column and target_column in merged_df.columns:
        print("\n[Optional] Applying WoE transformation...")
        woe_transformer = WOETransformer(target_column=target_column)
        merged_df = woe_transformer.transform(merged_df)
    print(f"FINAL FEATURE MATRIX: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")

    print(f"\nFeatures created:\n{merged_df.columns.tolist()}")
    
    return merged_df


if __name__ == "__main__":
    # Test the complete pipeline
    print("Testing Complete Feature Engineering Pipeline...\n")
    
    # Load raw data
    df = pd.read_csv('data/raw/data.csv')
    print(f"Loaded: {df.shape[0]:,} transactions, {df.shape[1]} columns")
    
    # Process features
    features = process_raw_data(df)
    
    print(f"\nSample of final features:")
    print(features.head(10))
    
    print(f"\nFeature statistics:")
    print(features.describe())