import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from api_stock import fetch_financial_data as fd  
from portfolio_utils import top_100

np.random.seed(42)
###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################
def implied_equilibrium_returns(Sigma, w_mkt, delta=2.5):
    """
    Black-Litterman 'implied returns':
       pi = delta * Sigma * w_mkt
    where:
       Sigma: (N x N) covariance matrix (as a NumPy array)
       w_mkt: (N,) market cap weights (NumPy array)
       delta: risk-aversion coefficient
    """
    w_mkt = w_mkt.reshape(-1, 1)  # Ensure column vector
    pi = delta * Sigma @ w_mkt
    return pi.ravel()

def black_litterman_posterior(pi, Sigma, P, Q, view_confidences, tau=0.05):
    """
    Black-Litterman posterior with Idzorek’s method for Omega estimation.
    """
    pi = pi.reshape(-1, 1)
    
    # Mispricing: Difference between views and market equilibrium
    mispricing = Q.reshape(-1, 1) - (P @ pi)
    
    # Compute Omega using Idzorek's confidence-based method
    Omega = np.diag([(1 - tau_v) / tau_v * (P[j:j+1, :] @ Sigma @ P[j:j+1, :].T).item() for j, tau_v in enumerate(view_confidences)])
    
    # Compute the middle term (weights the correction based on confidence)
    middle = np.linalg.inv(P @ (tau * Sigma) @ P.T + Omega)
    
    # Compute the correction term (adjusting market equilibrium based on investor views)
    correction = (tau * Sigma) @ P.T @ middle @ mispricing
    
    # Final posterior expected returns
    mu_bl = pi + correction
    return mu_bl.ravel()


def solve_mean_variance(mu, Sigma, risk_aversion=3.0, no_short=True):
    """
    Solve the mean-variance problem:
       maximize w^T mu - (risk_aversion/2) w^T Sigma w
    subject to sum(w) = 1, w >= 0 (if no_short=True)
    """
    n = len(mu)
    
    # If unconstrained, closed form:
    if not no_short:
        invS = np.linalg.inv(Sigma)
        w_star = (1.0 / risk_aversion) * invS @ mu
        # Normalize
        w_star /= w_star.sum()
        return w_star
    
    # Otherwise, numerical approach using SLSQP
    def objective(weights):
        ret = weights @ mu
        var = weights @ Sigma @ weights
        # negative utility
        return -(ret - (risk_aversion/2)*var)
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0,1)]*n
    
    w0 = np.ones(n)/n
    result = minimize(
        objective, 
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result.x

###############################################################################
# 2) DOWNLOAD AND PREP DATA
###############################################################################
start_date = "2015-09-01"
end_date   = "2023-01-01"

ciks = top_100

possible_share_tags = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding"
] 

share_ammount = fd(cik_dict=ciks, variables=possible_share_tags, start_date=start_date, end_date=end_date, single_variable=True )  
share_ammount = share_ammount.bfill().ffill()   
share_ammount = share_ammount.iloc[1:]

stocks = list(share_ammount.columns) 
# Download monthly data, then resample to Q
data = yf.download(stocks, start=start_date, end=end_date, interval="1mo")["Adj Close"]
data_q = data.resample("Q").last()
returns_q = data_q.pct_change().dropna()  
print(returns_q)
share_ammount.index = returns_q.index 

# Download S&P 500 (market) similarly
market_data = yf.download("^GSPC", start=start_date, end=end_date, interval="1mo")["Adj Close"]
market_data_q = market_data.resample("Q").last()
market_returns_q = market_data_q.pct_change().dropna() 

#risk free rate 
risk_free = yf.download("^IRX", start=start_date, end=end_date, interval="1mo")["Adj Close"]/100 
risk_free = risk_free.resample("Q").last().dropna()  
risk_free=risk_free.iloc[1:] 
risk_free.index = returns_q.index

rolling_window = 4
dates, portfolio_returns , weights_history = [], [], [] 
delta, tau = 3.0, 0.05  

print(len(risk_free), len(returns_q), len(share_ammount))

for i in range(rolling_window, len(returns_q)):
    past_slice = returns_q.iloc[i - rolling_window : i]  
    Sigma = (past_slice.cov()).values
    last_date = past_slice.index[-1] 

    # Extracting data from prices and share amounts 
    period_prices = data_q.loc[last_date, stocks].values  
    period_share_amm = share_ammount.loc[last_date, stocks].values 
    market_cap_inidi = period_prices * period_share_amm
    total_market_cap = market_cap_inidi.sum()
    market_cap_weights = market_cap_inidi / total_market_cap 
    risk_free_period = risk_free.loc[last_date]

    # Implied equilibrium returns 
    pi = implied_equilibrium_returns(Sigma, market_cap_weights, delta) 
    top_n = 50
    top_idx = np.argsort(pi)[::1][:top_n] 
    pi = pi[top_idx] 
    Sigma =Sigma[np.ix_(top_idx, top_idx)]  
    selected_stocks = [stocks[i] for i in top_idx] 


    # S&P500 returns 
    past_market_slice = market_returns_q.iloc[i - rolling_window : i] 
    r_m_annual = past_market_slice.mean()
    market_var = past_market_slice.var()

    # CAPM
    betas_full = np.array([
        past_slice[stock].cov(past_market_slice.squeeze()) / market_var 
        for stock in stocks
    ]) 

    betas = betas_full[top_idx]
    capm_view = risk_free_period + betas * (r_m_annual - risk_free_period)
    capm_view = capm_view.reshape(-1, 1)
    # Set up the view matrix P and Q for the filtered universe
    P = np.eye(top_n)
    Q = capm_view
    # Black-Litterman posterior weights 
    view_confidences = np.full(top_n, 0.5)  # Example: 50% confidence in all views
    view_confidences = np.clip(view_confidences, 1e-6, 1)  

    Omega = np.diag([
        (1 - tau_v) / tau_v * (P[j:j+1, :] @ Sigma @ P[j:j+1, :].T).item()
        for j, tau_v in enumerate(view_confidences)
    ])

    # Black-Litterman posterior returns incorporating view confidence
    mu_bl = black_litterman_posterior(pi, Sigma, P, Q, view_confidences, tau)
    w_star = solve_mean_variance(mu_bl, Sigma, delta, True)
    
    next_return_vector = returns_q.iloc[i][selected_stocks].values 
    port_ret = np.dot(w_star, next_return_vector)  # Returns gained from next period calculated with estimated weights 
    
    # Store results
    dates.append(returns_q.index[i])  
    portfolio_returns.append(port_ret)
    weights_history.append(w_star)  

# Convert portfolio returns to series
portfolio_returns_s = pd.Series(portfolio_returns, index=dates, name="Portfolio_return") 
weights_df = pd.DataFrame(weights_history, index=dates, columns=selected_stocks)  
weights_df['check'] = weights_df.sum(axis=1)

# Compute cumulative returns
cum_ret = (1 + portfolio_returns_s).cumprod() 
aligned_market_returns = market_returns_q.loc[dates] 
market_cum = (1 + aligned_market_returns).cumprod()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(cum_ret, label="BL Portfolio", color="blue", linewidth=2)
plt.plot(market_cum, label="S&P 500", color="orange", linewidth=2, linestyle="--")
plt.title("Rolling Black-Litterman vs. S&P 500 (Cumulative Returns)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=12)
plt.grid(alpha=0.5)
plt.legend(fontsize=10)
plt.show()

# Show quarterly return table
summary_table = pd.DataFrame({
    "Quarter": dates,
    "Portfolio Return (%)": portfolio_returns_s.values * 100,
    "S&P 500 Return (%)": aligned_market_returns.values * 100
})
summary_table.set_index("Quarter", inplace=True)
print("\nQuarterly Returns (%)")
print(summary_table.round(2))

# Show final weights
print("\nFinal Weights on the last rebalancing date:")
print(weights_df.iloc[-1].round(4))

# ----------------------------------------
# Additional Graphics: Highlighting Profit and Loss Periods
# ----------------------------------------

# Prepare the x-axis positions
quarters = portfolio_returns_s.index
ind = np.arange(len(quarters))  # numerical positions for each quarter
width = 0.35  # width of each bar

# Assign different hues based on return sign for each asset
bl_colors = ['seagreen' if ret >= 0 else 'tomato' for ret in portfolio_returns_s]
sp_colors = ['dodgerblue' if ret >= 0 else 'darkorange' for ret in aligned_market_returns]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 7))

# Plot the Black-Litterman portfolio bars (shifted left)
rects1 = ax.bar(ind - width/2, portfolio_returns_s.values, width,
                color=bl_colors, label='Black-Litterman Portfolio')

# Plot the S&P 500 bars (shifted right)
rects2 = ax.bar(ind + width/2, aligned_market_returns.values, width,
                color=sp_colors, label='S&P 500')

# Draw a horizontal line at y=0
ax.axhline(0, color='black', linewidth=0.8)

# Set axis labels and title
ax.set_xlabel('Quarter', fontsize=12)
ax.set_ylabel('Return', fontsize=12)
ax.set_title('Quarterly Returns Comparison: Black-Litterman vs. S&P 500', fontsize=14)

# Set x-ticks to display the quarters (formatted as Year-Month)
ax.set_xticks(ind)
ax.set_xticklabels([d.strftime("%Y-%m") for d in quarters], rotation=45)

# Add the legend
ax.legend(fontsize=12)

# Adjust layout to prevent clipping of tick-labels and show the plot
plt.tight_layout()
plt.show() 

# Get the last period's date
last_period = weights_df.index[-1]

# Get the last period's weights
last_weights = weights_df.iloc[-1].round(4)

# Print the result
print(f"\nOptimal Weights for Period {last_period}:")
for stock, weight in last_weights.items():
    print(f"{stock}: {weight*100:.2f}%") 

