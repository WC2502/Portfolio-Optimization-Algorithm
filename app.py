from flask import Flask, request, render_template, jsonify
from utils.helpers import load_data, calculate_returns
from portfolio.optimizer import markowitz_optimize
from portfolio.allocator import risk_parity
from models.ml_model import MLModels
from risk.risk_control import enforce_risk_controls
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/allocate', methods=['POST'])
def allocate():
    try:
        tickers = request.form['tickers'].split(',')
        prices = load_data(tickers)
        returns = calculate_returns(prices)
        cov_matrix = returns.cov()

        X = returns.shift(1).dropna()
        y = returns.loc[X.index]
        model = MLModels()
        model.train(X, y.mean(axis=1))
        future_returns = model.predict(X.tail(1))

        markowitz_weights = markowitz_optimize(returns, cov_matrix)
        risk_parity_weights = risk_parity(markowitz_weights, cov_matrix)
        final_weights = 0.5 * markowitz_weights + 0.5 * risk_parity_weights
        expected_return = np.dot(final_weights, future_returns)

        control = enforce_risk_controls(returns @ final_weights)

        return render_template('result.html', 
                               expected_return=round(expected_return * 252, 2),
                               drawdown_ok=control['Risk_OK'],
                               weights=dict(zip(tickers, final_weights.round(3))))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
