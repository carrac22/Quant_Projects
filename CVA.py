import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random as r

# Market Information
risk_free = 0.1

# Share Specific Information
S_0 = 100
sigma = 0.3

# Call Option Specific Information
strike = 110
T = 1

# Firm Specific Information
V_0 = 200
sigma_firm = 0.25
debt = 180
recovery_rate = 0.2

def terminal_value(S_0, risk_free_rate, sigma, Z, T):
    """
    Generates the terminal share price given some random normal values, Z
    """
    x = S_0*np.exp((risk_free_rate-sigma**2/2)*T+sigma*np.sqrt(T)*Z)
    return x

def call_payoff(S_T, K):
    """
    Function for evaluating the call price in Monte Carlo Estimation
    """
    x = np.maximum(S_T-K, 0)
    return x

#Correlations
np.random.seed(0)
corr_tested = np.linspace(-1,1,21)

#empty vectors
cva_estimates = [None]*len(corr_tested)
cva_std = [None]*len(corr_tested)

#MC simulation
for i in range(len(corr_tested)):
    
    correlation = corr_tested[i]
    if (correlation == 1 or correlation == -1):
        norm_vec_0 = norm.rvs(size = 50000)
        norm_vec_1 = correlation*norm_vec_0
        corr_norm_matrix = np.array([norm_vec_0, norm_vec_1])
        
    else:
        corr_matrix = np.array([[1, correlation], [correlation,1]])
        norm_matrix = norm.rvs(size = np.array([2,50000]))
        corr_norm_matrix = np.matmul(np.linalg.cholesky(corr_matrix), norm_matrix)
     
    # create an array of stock values using the first row of the matrix of
    # correlated standard normals. (creates an array of 50 000 stock values)
    term_stock_val = terminal_value(S_0, risk_free, sigma, corr_norm_matrix[0,], T)
    
    # create an array of call values for the given stock values.
    call_val = call_payoff(term_stock_val, strike)
    
    # create an array of terminal firm values. (creates an array of 50 000 firm values)
    term_firm_val = terminal_value(V_0, risk_free, sigma_firm, corr_norm_matrix[1,], T)
    
    # Using the call values in the formula
    amount_lost = np.exp(-risk_free*T)*(1-recovery_rate)*(term_firm_val < debt)*call_val
    cva_estimates[i] = np.mean(amount_lost)
    cva_std[i] = np.std(amount_lost)/np.sqrt(50000)