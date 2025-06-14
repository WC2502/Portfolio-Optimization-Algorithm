import numpy as np
from scipy.optimize import minimize

def markowitz_optimize(returns, cov_matrix, risk_free_rate=0.01):
    n_assets = returns.shape[1]

    def sharpe_ratio(weights):
        port_return = np.sum(weights * returns.mean())
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_free_rate) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = n_assets * [1. / n_assets]

    result = minimize(sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
