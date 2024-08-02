import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def train_ridge(X, y, lambdas):
    # Train Ridge regression models for each value of lambda
    models = {}
    for lambda_ in lambdas:
        model = Ridge(alpha=lambda_)
        model.fit(X, y)
        models[lambda_] = model
    return models

def tune_lambda(models, X_I, y_I):
    # Find the best lambda based on interventional data
    best_lambda = None
    best_score = float('-inf')
    for lambda_, model in models.items():
        score = model.score(X_I, y_I)
        if score > best_score:
            best_score = score
            best_lambda = lambda_
    return best_lambda

def causal_regularization(y_O, X_O, y_I, X_I, lambdas):
    # Step 1: Train models on observational data
    models = train_ridge(X_O, y_O, lambdas)
    
    # Step 2: Use interventional data to find the best lambda
    best_lambda = tune_lambda(models, X_I, y_I)
    
    # Step 3: Use the best model for predictions
    best_model = models[best_lambda]
    return best_model

# Example usage:
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(0)
    X_O = np.random.rand(100, 10)
    y_O = X_O @ np.random.rand(10) + np.random.randn(100)
    
    X_I = np.random.rand(20, 10)
    y_I = X_I @ np.random.rand(10) + np.random.randn(20)
    
    lambdas = np.logspace(-4, 4, 10)
    
    best_model = causal_regularization(y_O, X_O, y_I, X_I, lambdas)
    
    # Predict using the best model
    y_pred = best_model.predict(X_I)
    print("Best Lambda:", best_model.alpha)
    print("Predictions:", y_pred)
