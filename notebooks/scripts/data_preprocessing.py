import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

def load_data():
    from ucimlrepo import fetch_ucirepo
    
    # fetch dataset 
    steel_plates_faults = fetch_ucirepo(id=198) 
    
    # features and targets (as pandas dataframes) 
    X = steel_plates_faults.data.features 
    y = steel_plates_faults.data.targets 
    
    return X, y

# Drop features with high correlation
def drop_high_correlated_features(X):
    corr = X.corr()
    high_corr = [(i, j) for i in corr.columns for j in corr.columns if (abs(corr.loc[i, j]) > 0.9) and (i != j)]
    drop_cols = [i for i, j in high_corr]
    X_reduced = X.drop(columns=drop_cols)
    return X_reduced

# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Calculate class weights
def calculate_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    return class_weights


def split_data(X, y, test_size=0.2, seed=5296):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    X_train, X_test = scale_features(X_train, X_test)

    # Convert one-hot encoded targets to string labels, then to numeric labels
    y_train = y_train.idxmax(axis=1)
    y_test = y_test.idxmax(axis=1)
    
    # Convert string labels to numeric labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    class_weights = calculate_class_weights(y_train_encoded)
    return X_train, X_test, y_train_encoded, y_test_encoded, class_weights