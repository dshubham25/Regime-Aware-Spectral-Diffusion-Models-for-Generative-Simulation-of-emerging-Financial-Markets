import numpy as np

def compute_volatility(returns, window=21):
    vol = np.zeros_like(returns)
    for i in range(window, len(returns)):
        vol[i] = np.std(returns[i-window:i])
    return vol

def compute_drawdown(prices):
    peak = prices[0]
    drawdowns = []

    for p in prices:
        if p > peak:
            peak = p
        drawdowns.append((p - peak) / peak)

    return np.array(drawdowns)

def assign_regimes(volatility, drawdown):
    regimes = np.zeros_like(volatility)

    for i in range(len(volatility)):
        if drawdown[i] < -0.15:
            regimes[i] = 2  # Crisis
        elif volatility[i] > np.percentile(volatility, 70):
            regimes[i] = 1  # Volatile
        else:
            regimes[i] = 0  # Stable

    return regimes
