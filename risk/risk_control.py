import numpy as np

def calculate_var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_cvar(returns, confidence=0.95):
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()

def max_drawdown(cumulative_returns):
    high_water_mark = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - high_water_mark) / high_water_mark
    return drawdowns.min()

def enforce_risk_controls(returns):
    var = calculate_var(returns)
    cvar = calculate_cvar(returns)
    drawdown = max_drawdown(np.cumsum(returns))
    
    return {
        "VaR": var,
        "CVaR": cvar,
        "Drawdown": drawdown,
        "Risk_OK": drawdown > -0.05
    }
