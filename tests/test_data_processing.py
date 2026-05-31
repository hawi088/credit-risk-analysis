"""
Unit Tests for Feature Engineering Pipeline
Task 5 - Unit Testing

Author: Kerod, Mahbubah, Feven
Date: 31 May 2026
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    CustomerAggregator, 
    TimeFeatureExtractor, 
    LogTransformer,
    process_raw_data
)


class TestCustomerAggregator:
    """Tests for CustomerAggregator class."""
    
    def test_output_shape(self):
        """Test that CustomerAggregator returns expected number of columns."""
        # Create sample data with 3 customers (minimum for K-Means)
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3'],
            'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
            'Amount': [100, 200, 50, 75, 300, 400],
            'Value': [100, 200, 50, 75, 300, 400],
            'TransactionStartTime': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-03', '2024-01-05', '2024-01-06'],
            'ProductCategory': ['A', 'B', 'A', 'C', 'B', 'C'],
            'ChannelId': ['CH1', 'CH1', 'CH2', 'CH2', 'CH1', 'CH2'],
            'FraudResult': [0, 0, 0, 1, 0, 0]
        })
        
        aggregator = CustomerAggregator()
        aggregator.fit(df)
        result = aggregator.transform(df)
        
        # Check that CustomerId is present
        assert 'CustomerId' in result.columns
        assert result.shape[0] == 3  # 3 unique customers
    
    def test_recency_calculation(self):
        """Test that Recency is calculated correctly."""
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2'],
            'TransactionId': ['T1', 'T2', 'T3', 'T4'],
            'Amount': [100, 200, 50, 75],
            'Value': [100, 200, 50, 75],
            'TransactionStartTime': ['2024-01-01', '2024-01-10', '2024-01-01', '2024-01-05'],
            'ProductCategory': ['A', 'A', 'B', 'B'],
            'ChannelId': ['CH1', 'CH1', 'CH2', 'CH2'],
            'FraudResult': [0, 0, 0, 0]
        })
        
        aggregator = CustomerAggregator()
        aggregator.fit(df)
        result = aggregator.transform(df)
        
        # Recency should be days between last transaction and snapshot date
        assert 'Recency' in result.columns
        assert result['Recency'].iloc[0] >= 0


class TestTimeFeatureExtractor:
    """Tests for TimeFeatureExtractor class."""
    
    def test_time_features_created(self):
        """Test that time features are properly extracted."""
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C1'],
            'TransactionId': ['T1', 'T2'],
            'Amount': [100, 200],
            'Value': [100, 200],
            'TransactionStartTime': ['2024-01-01 10:30:00', '2024-01-02 15:45:00'],
            'ProductCategory': ['A', 'B'],
            'ChannelId': ['CH1', 'CH1'],
            'FraudResult': [0, 0]
        })
        
        extractor = TimeFeatureExtractor()
        result = extractor.transform(df)
        
        # Check that time features were added
        assert 'TransactionHour' in result.columns
        assert 'TransactionDay' in result.columns
        assert 'TransactionMonth' in result.columns
        assert 'TransactionYear' in result.columns
        assert 'TransactionDayOfWeek' in result.columns
        assert 'IsWeekend' in result.columns
    
    def test_hour_extraction(self):
        """Test that hour is correctly extracted."""
        df = pd.DataFrame({
            'CustomerId': ['C1'],
            'TransactionId': ['T1'],
            'Amount': [100],
            'Value': [100],
            'TransactionStartTime': ['2024-01-01 14:30:00'],
            'ProductCategory': ['A'],
            'ChannelId': ['CH1'],
            'FraudResult': [0]
        })
        
        extractor = TimeFeatureExtractor()
        result = extractor.transform(df)
        
        assert result['TransactionHour'].iloc[0] == 14


class TestLogTransformer:
    """Tests for LogTransformer class."""
    
    def test_log_transformation(self):
        """Test that log transformation is applied correctly."""
        df = pd.DataFrame({
            'Monetary': [100, 1000, 10000],
            'Frequency': [1, 10, 100]
        })
        
        transformer = LogTransformer()
        transformer.fit(df)
        result = transformer.transform(df)
        
        # Check that log columns were created
        assert 'Monetary_log' in result.columns
        assert 'Frequency_log' in result.columns
        
        # Check that original columns were dropped
        assert 'Monetary' not in result.columns
        assert 'Frequency' not in result.columns
    
    def test_negative_values_handling(self):
        """Test that negative values are handled properly."""
        df = pd.DataFrame({
            'Monetary': [-100, 0, 1000],
            'Frequency': [1, 5, 10]
        })
        
        transformer = LogTransformer()
        transformer.fit(df)
        result = transformer.transform(df)
        
        # Should not produce NaN values
        assert not result['Monetary_log'].isna().any()
        assert not result['Frequency_log'].isna().any()


def test_process_raw_data_returns_dataframe():
    """Test that main processing function returns a DataFrame."""
    # Create sample data with at least 3 customers (required for K-Means)
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3'],
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        'Amount': [100, 200, 50, 75, 300, 400],
        'Value': [100, 200, 50, 75, 300, 400],
        'TransactionStartTime': ['2024-01-01', '2024-01-10', '2024-01-01', '2024-01-05', '2024-01-07', '2024-01-08'],
        'ProductCategory': ['A', 'B', 'A', 'C', 'B', 'C'],
        'ChannelId': ['CH1', 'CH1', 'CH2', 'CH2', 'CH1', 'CH2'],
        'FraudResult': [0, 0, 0, 1, 0, 0]
    })
    
    result = process_raw_data(df)
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that target column exists
    assert 'is_high_risk' in result.columns


def test_target_column_binary():
    """Test that target column is binary (0 or 1)."""
    # Create sample data with at least 3 customers
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3'],
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        'Amount': [100, 200, 50, 75, 500, 600],
        'Value': [100, 200, 50, 75, 500, 600],
        'TransactionStartTime': ['2024-01-01', '2024-01-10', '2024-01-01', '2024-01-03', '2024-01-05', '2024-01-06'],
        'ProductCategory': ['A', 'A', 'B', 'B', 'C', 'C'],
        'ChannelId': ['CH1', 'CH1', 'CH2', 'CH2', 'CH1', 'CH1'],
        'FraudResult': [0, 0, 0, 0, 0, 0]
    })
    
    result = process_raw_data(df)
    
    # Check target is binary
    assert result['is_high_risk'].isin([0, 1]).all()


def test_no_missing_values():
    """Test that output has no missing values."""
    # Create sample data with at least 3 customers
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3'],
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        'Amount': [100, 200, 50, 75, 150, 250],
        'Value': [100, 200, 50, 75, 150, 250],
        'TransactionStartTime': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-03', '2024-01-04', '2024-01-05'],
        'ProductCategory': ['A', 'B', 'A', 'C', 'B', 'A'],
        'ChannelId': ['CH1', 'CH1', 'CH2', 'CH2', 'CH1', 'CH2'],
        'FraudResult': [0, 0, 0, 1, 0, 0]
    })
    
    result = process_raw_data(df)
    
    # Check no missing values
    assert result.isnull().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])