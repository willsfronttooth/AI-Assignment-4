# src/preprocessing.py
"""
Data loading and preprocessing utilities.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def missing_value_summary(df):
    return df.isnull().sum()

def drop_or_impute(df, drop_thresh=0.5, numeric_strategy='median', categorical_strategy='mode'):
    """
    - Drops rows missing product_id or product_name (can't infer identity).
    - For numeric columns, imputes median by default.
    - For categorical columns, imputes mode by default.
    """
    df = df.copy()
    # Drop rows missing product_id or product_name
    if 'product_id' in df.columns:
        df = df[~df['product_id'].isnull()]
    if 'product_name' in df.columns:
        df = df[~df['product_name'].isnull()]
    # Numeric imputation
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'product_id' in num_cols:
        num_cols.remove('product_id')
    for col in num_cols:
        if df[col].isnull().sum() == 0:
            continue
        if numeric_strategy == 'median':
            val = df[col].median()
        elif numeric_strategy == 'mean':
            val = df[col].mean()
        else:
            val = df[col].median()
        df[col].fillna(val, inplace=True)
    # Categorical imputation
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() == 0:
            continue
        if categorical_strategy == 'mode':
            val = df[col].mode().iloc[0]
        else:
            val = df[col].mode().iloc[0]
        df[col].fillna(val, inplace=True)
    return df

def detect_outliers_iqr(df, cols, k=1.5):
    """
    Returns boolean series indicating rows with outlier in any given column.
    """
    mask = pd.Series(False, index=df.index)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        mask = mask | ((df[col] < lower) | (df[col] > upper))
    return mask

def cap_outliers(df, cols, k=1.5):
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df

def scale_features(df, cols, method='standard'):
    df = df.copy()
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler