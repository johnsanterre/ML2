# Building a Real-World Housing Price Predictor

## Introduction
This chapter walks through the complete process of building a neural network for real estate price prediction using the California Housing Dataset. We'll cover each step from data exploration to model deployment, emphasizing practical implementation considerations.

## 1. Problem Setup & Data Exploration

### 1.1 The California Housing Dataset
The California Housing Dataset represents a real-world regression problem with several key features:
- Median house values for California districts
- Derived from the 1990 U.S. census
- Contains ~20,000 entries with 8 features
- Includes both numerical and categorical data

#### Feature Description
- MedInc: Median income in block group
- HouseAge: Median house age in block group
- AveRooms: Average rooms per household
- AveBedrms: Average bedrooms per household
- Population: Block group population
- AveOccup: Average occupancy
- Latitude: Block group latitude
- Longitude: Block group longitude

### 1.2 Data Analysis
Before building our model, we need to understand our data:

```python
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load data
housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

# Basic statistics
stats = {
    'mean': np.mean(X, axis=0),
    'std': np.std(X, axis=0),
    'min': np.min(X, axis=0),
    'max': np.max(X, axis=0)
}

for i, feature in enumerate(feature_names):
    print(f"{feature}:")
    print(f"  Mean: {stats['mean'][i]:.2f}")
    print(f"  Std:  {stats['std'][i]:.2f}")
    print(f"  Min:  {stats['min'][i]:.2f}")
    print(f"  Max:  {stats['max'][i]:.2f}")

# Check for missing values
missing = np.isnan(X).sum(axis=0)
for i, count in enumerate(missing):
    print(f"{feature_names[i]}: {count} missing values")

# Correlation analysis
def correlation_matrix(X, y):
    # Combine features and target
    data = np.column_stack([X, y])
    # Calculate correlation matrix
    corr = np.corrcoef(data.T)
    return corr

correlations = correlation_matrix(X, y)
```

### 1.3 Data Visualization
Key visualizations help understand relationships:
- Distribution plots for each feature
- Correlation heatmap
- Scatter plots against target variable
- Geographic distribution plots

## 2. Data Preprocessing Pipeline

### 2.1 Feature Scaling
Proper scaling is crucial for neural network performance:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 2.2 Train/Validation/Test Split
We implement a robust splitting strategy:

```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: create validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)
```

## 3. Model Development

### 3.1 PyTorch Implementation
We create a custom dataset class and model architecture:

```python
import torch
import torch.nn as nn

class HousingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HousingModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
```

### 3.2 Training Loop
Implementing an efficient training process:

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch.unsqueeze(1)).item()
                
        print(f'Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, '
              f'Val Loss = {val_loss/len(val_loader):.4f}')
```

## 4. Model Evaluation & Improvement

### 4.1 Performance Metrics
We implement multiple evaluation metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

```python
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predictions.extend(y_pred.numpy())
            actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
```

### 4.2 Hyperparameter Tuning
Key parameters to optimize:
- Learning rate
- Network architecture
- Batch size
- Optimizer choice

### 4.3 Regularization Techniques
Implementing various regularization methods:
- Dropout layers
- L1/L2 regularization
- Early stopping
- Learning rate scheduling

## 5. Model Deployment

### 5.1 Saving the Model
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler,
}, 'housing_model.pth')
```

### 5.2 Inference Pipeline
Creating a robust inference system:
```python
def predict_price(features, model, scaler):
    model.eval()
    features_scaled = scaler.transform(features)
    with torch.no_grad():
        prediction = model(torch.FloatTensor(features_scaled))
    return prediction.numpy()
```

## Summary
Building a real-world housing price predictor involves:
1. Thorough data analysis and preprocessing
2. Careful model architecture design
3. Robust training and evaluation
4. Practical deployment considerations

The skills developed in this implementation provide a foundation for tackling similar regression problems with neural networks. 