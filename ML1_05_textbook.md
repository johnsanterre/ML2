# Linear and Logistic Regression: From Direct Solutions to Iterative Methods

## Introduction
Linear regression and logistic regression form the foundation of many machine learning approaches. While these methods may seem simple, they introduce crucial concepts that extend to more complex models. This chapter progresses from basic linear regression through to multiclass logistic regression, emphasizing both theoretical understanding and practical implementation.

## 1. Linear Regression Solutions

### 1.1 Ordinary Least Squares (OLS)
Ordinary Least Squares provides a direct, analytical solution to the linear regression problem. Given a set of data points, OLS finds the linear relationship that minimizes the sum of squared residuals between the predicted and actual values.

#### Direct Solution Method
The linear regression model can be expressed as:
y = Xβ + ε

where:
- y is the vector of target values
- X is the matrix of features
- β is the vector of parameters to be estimated
- ε represents the error terms

The OLS solution is given by:
β = (X'X)⁻¹X'y

#### Computational Considerations
While OLS provides an exact solution, it has limitations:
- Requires matrix inversion, which is O(n³)
- Memory intensive for large datasets
- Numerically unstable for ill-conditioned matrices
- Not suitable for online learning

### 1.2 Iterative Approach
When direct solutions become impractical, iterative methods offer an alternative approach.

#### Gradient Descent Formulation
The gradient descent algorithm iteratively updates parameters:
β(t+1) = β(t) - α∇L(β)

where:
- α is the learning rate
- L(β) is the loss function
- ∇L(β) is the gradient of the loss function

## 2. Binary Logistic Regression

### 2.1 Problem Formulation
While linear regression predicts continuous values, many real-world problems require binary classification. Logistic regression extends linear regression to classification by applying a sigmoid function to the linear model.

#### From Linear to Logistic
The logistic regression model transforms the linear combination of features into probabilities:
P(y=1|x) = σ(Xβ)

where σ(z) is the sigmoid function:
σ(z) = 1/(1 + e⁻ᶻ)

#### Decision Boundaries
The decision boundary in logistic regression is the surface where P(y=1|x) = 0.5, which occurs when Xβ = 0. This creates a linear boundary in feature space.

### 2.2 Gradient Descent Solution
Unlike linear regression, logistic regression has no closed-form solution and requires iterative optimization.

#### Loss Function
The negative log-likelihood loss function is:
L(β) = -Σ[y_i log(p_i) + (1-y_i)log(1-p_i)]

where:
- y_i is the true label (0 or 1)
- p_i is the predicted probability

## 3. Multiclass Extension

### 3.1 One-vs-All Approach
For problems with K > 2 classes, one approach is to train K separate binary classifiers.

#### Multiple Binary Classifiers
For each class k:
- Treat class k as positive (1)
- All other classes as negative (0)
- Train a binary logistic regression model
- Predict using maximum probability across all classifiers

### 3.2 Softmax Regression
A more elegant solution for multiclass problems is softmax regression, also known as multinomial logistic regression.

#### Multinomial Model
The softmax function generalizes the sigmoid to K classes:
P(y=k|x) = exp(Xβₖ)/Σexp(Xβⱼ)

where:
- βₖ is the parameter vector for class k
- The sum in denominator is over all classes

## 4. Optimization Methods

### 4.1 Gradient Descent Variations
Different approaches to gradient computation offer various trade-offs between computation speed and convergence stability.

#### Batch Gradient Descent
- Uses entire dataset for each update
- Stable but computationally expensive
- Guaranteed to converge to local minimum
- Memory intensive for large datasets

#### Stochastic Gradient Descent
- Updates parameters using single examples
- Faster iteration but noisier updates
- Better for large datasets
- Requires careful learning rate scheduling

#### Mini-batch Approach
- Compromise between batch and stochastic
- Updates using small random batches
- Balances computation and convergence
- Popular in practice

### 4.2 Implementation Details

#### Learning Rate Selection
- Too large: may diverge
- Too small: slow convergence
- Common strategies:
  * Start larger and decay
  * Adaptive methods (e.g., Adam)

#### Convergence Monitoring
- Track loss function
- Monitor parameter changes
- Use validation set performance
- Implement early stopping

## Summary
The progression from linear to logistic regression, and from binary to multiclass classification, introduces fundamental concepts in machine learning:
- Direct vs. iterative solutions
- Linear vs. non-linear transformations
- Binary vs. multiclass classification
- Optimization strategies and trade-offs

These concepts form the foundation for understanding more complex models and algorithms in machine learning. 