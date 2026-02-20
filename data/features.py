import numpy as np

def compute_log_returns(prices):
    prices = np.array(prices)
    returns = np.log(prices[1:] / prices[:-1])
    return returns
