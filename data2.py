import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Load the data
data = pd.read_csv('card_transdata.csv')

# Display initial data for verification
print(data.head())
print(data.describe())
print(data.isnull().sum())

# Transformation logarithmique pour réduire l'impact des valeurs aberrantes
data['log_distance_from_home'] = np.log1p(data['distance_from_home'])
data['log_distance_from_last_transaction'] = np.log1p(data['distance_from_last_transaction'])
data['log_ratio_to_median_purchase_price'] = np.log1p(data['ratio_to_median_purchase_price'])

# Define transformers
numerical_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
binary_features = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']

# Transformer for numerical features: impute missing values and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Transformer for binary features: impute missing values (if necessary)
binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Column transformer to apply transformations to appropriate columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('bin', binary_transformer, binary_features)
    ])

# Fit and transform the data
data_preprocessed = preprocessor.fit_transform(data)

# Converting processed data back to a DataFrame
column_names = numerical_features + binary_features
data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=column_names)
data_preprocessed_df['fraud'] = data['fraud']  # Append 'fraud' column for plotting

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5)
X_pca = pca.fit_transform(data_preprocessed_df.drop(columns=['fraud']))

# Convert PCA results to a DataFrame
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)
X_pca_df['fraud'] = data['fraud']

# Print the explained variance ratio for each principal component
print("Explained Variance Ratio by Principal Component:")
print(pca.explained_variance_ratio_)

# Split the PCA-transformed data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_df.drop(columns=['fraud']), X_pca_df['fraud'], test_size=0.3, random_state=42, stratify=X_pca_df['fraud'])
from sklearn.ensemble import GradientBoostingClassifier

# Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=5, min_samples_leaf=2)
gb.fit(X_train_pca, y_train_pca)

# Make predictions
y_pred_gb = gb.predict(X_test_pca)

# Evaluate the model
print("Gradient Boosting with PCA:")
print(classification_report(y_test_pca, y_pred_gb))
print("Confusion Matrix:\n", confusion_matrix(y_test_pca, y_pred_gb))
print("ROC AUC Score:", roc_auc_score(y_test_pca, y_pred_gb))
