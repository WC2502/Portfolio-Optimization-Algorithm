import numpy as np
from utils.helpers import load_data, calculate_returns
from portfolio.optimizer import markowitz_optimize
from portfolio.allocator import risk_parity
from models.ml_model import MLModels
from risk.risk_control import enforce_risk_controls

# Step 1: Load and process data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
prices = load_data(tickers)
returns = calculate_returns(prices)
cov_matrix = returns.cov()

# Step 2: Train ML model
X = returns.shift(1).dropna()
y = returns.loc[X.index]
model = MLModels()
model.train(X, y.mean(axis=1))

# Step 3: Predict future returns
future_returns = model.predict(X.tail(1))

# Step 4: Optimize portfolio
markowitz_weights = markowitz_optimize(returns, cov_matrix)
risk_parity_weights = risk_parity(markowitz_weights, cov_matrix)

# Step 5: Combine weights with ML predictions
final_weights = 0.5 * markowitz_weights + 0.5 * risk_parity_weights
expected_return = np.dot(final_weights, future_returns)

# Step 6: Risk checks
control = enforce_risk_controls(returns @ final_weights)

# Step 7: Print results
print("Expected Annualized Return:", round(expected_return * 252, 2))
print("Drawdown OK:", control["Risk_OK"])
print("Weights:", dict(zip(tickers, final_weights.round(3))))
