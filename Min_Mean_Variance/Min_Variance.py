import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import Counter
from portfolio_utils import top_100

# -----------------------------
# Optimization Functions
# -----------------------------
def mean_variance_opt(Sigma: np.ndarray, mu: np.ndarray, risk_aversion: float = 3.0):
    """
    Compute the mean-variance optimal portfolio weights under the constraint
    of no short-selling (weights >= 0) and full-investment (sum(weights)=1).
    """
    n = len(mu)
    
    def objective(w):
        # Negative of utility: w^T mu - 0.5 * risk_aversion * w^T Sigma w
        return - (np.dot(w, mu) - 0.5 * risk_aversion * np.dot(w, Sigma @ w))
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success:
        return res.x
    else:
        raise ValueError("Mean-variance optimization did not converge: " + res.message)

def min_variance_opt(Sigma: np.ndarray):
    """
    Compute the minimum-variance portfolio weights under the constraint
    of no short-selling (weights >= 0) and full-investment (sum(weights)=1).
    """
    n = Sigma.shape[0]
    
    def objective(w):
        return np.dot(w, Sigma @ w)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success:
        return res.x
    else:
        raise ValueError("Minimum variance optimization did not converge: " + res.message)

# -----------------------------
# Download Data & Prepare Returns
# -----------------------------
start_date = "2015-09-01"
end_date = "2023-01-01"
stocks = list(top_100.keys())

# Stock prices and returns
prices = yf.download(stocks, start=start_date, end=end_date, interval="1mo")['Adj Close']
prices = prices.resample("Q").last()
returns = prices.pct_change().dropna()
prices = prices.iloc[1:]

# Market (S&P 500) data and returns
market_data = yf.download("^GSPC", start_date, end_date, interval="1mo")['Adj Close']
market_data = market_data.resample("Q").last()
market_returns = market_data.pct_change().dropna()

# Risk-free rate (using IRX as proxy)
risk_free = yf.download("^IRX", start_date, end_date, interval="1mo")['Adj Close']
risk_free = risk_free.resample("Q").last().dropna()

# -----------------------------
# Simulation Functions
# -----------------------------
def simulate_performance(rolling_window, top_n):
    """
    Run the simulation over the entire out-of-sample period for a given
    rolling window and top_n. Returns the final cumulative return for both
    the MVM and MV models.
    """
    mvm_returns = []
    mv_returns = []
    
    for i in range(rolling_window, len(returns)):
        past_slice = returns.iloc[i - rolling_window:i]
        Sigma = past_slice.cov()
        past_market_slice = market_returns.iloc[i - rolling_window:i]
        last_date = past_slice.index[-1]
        risk_free_period = risk_free.loc[last_date]

        # Compute CAPM betas for each stock
        betas = np.array([
            past_slice[stock].cov(past_market_slice.squeeze()) / past_market_slice.var()
            for stock in stocks
        ])
        
        market_returns_mean = past_market_slice.mean()
        capmt_returns = risk_free_period + betas * (market_returns_mean - risk_free_period)
        
        # Select top_n stocks (highest expected returns)
        top_idx = np.argsort(capmt_returns)[::-1][:top_n]
        capmt_returns_selected = capmt_returns[top_idx]
        selected_stocks = [stocks[j] for j in top_idx]
        
        Sigma_selected = Sigma.loc[selected_stocks, selected_stocks]
        try:
            optimal_weights_MVM = mean_variance_opt(Sigma_selected, capmt_returns_selected, risk_aversion=3.0)
            optimal_weights_MV = min_variance_opt(Sigma_selected)
        except Exception as e:
            continue  # Skip period if optimization fails
        
        next_return_vector = returns.iloc[i][selected_stocks].values
        mvm_returns.append(next_return_vector @ optimal_weights_MVM)
        mv_returns.append(next_return_vector @ optimal_weights_MV)
    
    final_cum_mvm = np.prod(1 + np.array(mvm_returns))
    final_cum_mv  = np.prod(1 + np.array(mv_returns))
    
    return final_cum_mvm, final_cum_mv

def simulate_time_series(rolling_window, top_n):
    """
    Run the simulation and return the series (and dates) of portfolio returns
    for both models, as well as the list of selected stocks per period.
    """
    mvm_returns = []
    mv_returns = []
    date_list = []
    selected_stocks_list = []  # Record selected stocks for each period
    
    for i in range(rolling_window, len(returns)):
        past_slice = returns.iloc[i - rolling_window:i]
        Sigma = past_slice.cov()
        past_market_slice = market_returns.iloc[i - rolling_window:i]
        last_date = past_slice.index[-1]
        risk_free_period = risk_free.loc[last_date]

        betas = np.array([
            past_slice[stock].cov(past_market_slice.squeeze()) / past_market_slice.var()
            for stock in stocks
        ])
        
        market_returns_mean = past_market_slice.mean()
        capmt_returns = risk_free_period + betas * (market_returns_mean - risk_free_period)
        
        top_idx = np.argsort(capmt_returns)[::-1][:top_n]
        capmt_returns_selected = capmt_returns[top_idx]
        selected_stocks = [stocks[j] for j in top_idx]
        selected_stocks_list.append(selected_stocks)  # Record for this period
        
        Sigma_selected = Sigma.loc[selected_stocks, selected_stocks]
        try:
            optimal_weights_MVM = mean_variance_opt(Sigma_selected, capmt_returns_selected, risk_aversion=3.0)
            optimal_weights_MV = min_variance_opt(Sigma_selected)
        except Exception as e:
            continue
        
        date_list.append(returns.index[i])
        next_return_vector = returns.iloc[i][selected_stocks].values
        mvm_returns.append(next_return_vector @ optimal_weights_MVM)
        mv_returns.append(next_return_vector @ optimal_weights_MV)
    
    return date_list, np.array(mvm_returns), np.array(mv_returns), selected_stocks_list

# -----------------------------
# Grid Search over Hyper-parameters
# -----------------------------
rolling_window_list = [4, 6, 8, 10, 12]      # (in quarters)
top_n_list = [5, 10, 15, 20, 25]              # Number of stocks to select

results_MVM = np.zeros((len(rolling_window_list), len(top_n_list)))
results_MV = np.zeros((len(rolling_window_list), len(top_n_list)))

print("Starting grid search ...")
for i, rw in enumerate(rolling_window_list):
    for j, tn in enumerate(top_n_list):
        print(f"Simulating for rolling window = {rw} and top_n = {tn} ...")
        perf_mvm, perf_mv = simulate_performance(rw, tn)
        results_MVM[i, j] = perf_mvm
        results_MV[i, j] = perf_mv

df_MVM = pd.DataFrame(results_MVM, index=rolling_window_list, columns=top_n_list)
df_MV = pd.DataFrame(results_MV, index=rolling_window_list, columns=top_n_list) 


# -----------------------------
# Plot Grid Search Results
# -----------------------------
# For the MVM model:
plt.figure(figsize=(10, 6))
for tn in top_n_list:
    plt.plot(rolling_window_list, df_MVM[tn], marker='o', label=f'top_n = {tn}')
plt.xlabel('Rolling Window Size (quarters)')
plt.ylabel('Final Cumulative Return')
plt.title('Grid Search: MVM Performance vs. Rolling Window Size')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for rw in rolling_window_list:
    plt.plot(top_n_list, df_MVM.loc[rw], marker='o', label=f'Window = {rw}')
plt.xlabel('Number of Stocks Selected (top_n)')
plt.ylabel('Final Cumulative Return')
plt.title('Grid Search: MVM Performance vs. Number of Stocks Selected')
plt.legend()
plt.grid(True)
plt.show()

# For the MV model:
plt.figure(figsize=(10, 6))
for tn in top_n_list:
    plt.plot(rolling_window_list, df_MV[tn], marker='o', label=f'top_n = {tn}')
plt.xlabel('Rolling Window Size (quarters)')
plt.ylabel('Final Cumulative Return')
plt.title('Grid Search: MV Performance vs. Rolling Window Size')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for rw in rolling_window_list:
    plt.plot(top_n_list, df_MV.loc[rw], marker='o', label=f'Window = {rw}')
plt.xlabel('Number of Stocks Selected (top_n)')
plt.ylabel('Final Cumulative Return')
plt.title('Grid Search: MV Performance vs. Number of Stocks Selected')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Select Best Hyper-parameters
# -----------------------------
best_idx_MVM = np.unravel_index(np.argmax(results_MVM, axis=None), results_MVM.shape)
best_rw_MVM = rolling_window_list[best_idx_MVM[0]]
best_top_n_MVM = top_n_list[best_idx_MVM[1]]
print(f"Best hyper-parameters for MVM: Rolling Window = {best_rw_MVM}, Top_n = {best_top_n_MVM}")

best_idx_MV = np.unravel_index(np.argmax(results_MV, axis=None), results_MV.shape)
best_rw_MV = rolling_window_list[best_idx_MV[0]]
best_top_n_MV = top_n_list[best_idx_MV[1]]
print(f"Best hyper-parameters for MV: Rolling Window = {best_rw_MV}, Top_n = {best_top_n_MV}")

# -----------------------------
# Re-estimate Models with Best Hyper-parameters and Produce Performance Plots
# -----------------------------
# Run simulation to get time series of portfolio returns and selected stocks using best hyper-parameters.
dates_MVM, mvm_returns_series, _, selected_list_MVM = simulate_time_series(best_rw_MVM, best_top_n_MVM)
dates_MV, _, mv_returns_series, selected_list_MV   = simulate_time_series(best_rw_MV, best_top_n_MV)

# --- Align Time Series to a Common Date Range ---
s_MVM = pd.Series(mvm_returns_series, index=dates_MVM)
s_MV  = pd.Series(mv_returns_series, index=dates_MV)
common_dates = s_MVM.index.intersection(s_MV.index)
s_MVM = s_MVM.loc[common_dates]
s_MV = s_MV.loc[common_dates]

# Align S&P 500 returns to the same common dates.
s_SP = market_returns.loc[common_dates]

# Compute cumulative returns.
cum_ret_MVM = (1 + s_MVM).cumprod()
cum_ret_MV  = (1 + s_MV).cumprod()
cum_ret_SP  = (1 + s_SP).cumprod()

# Plot cumulative returns (line plot) including S&P 500.
plt.figure(figsize=(12, 6))
plt.plot(common_dates, cum_ret_MVM, label="MVM (best params)", color="blue", linewidth=2)
plt.plot(common_dates, cum_ret_MV,  label="MV (best params)",  color="green", linewidth=2)
plt.plot(common_dates, cum_ret_SP,  label="S&P 500",       color="orange", linestyle="--", linewidth=2)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Return", fontsize=12)
plt.title("Cumulative Returns: MVM vs MV vs S&P 500 (Best Hyper-parameters)", fontsize=14)
plt.grid(alpha=0.5)
plt.legend(fontsize=10)
plt.show()

# Create a grouped bar chart for quarterly returns.
df_quarterly = pd.concat([s_MVM, s_MV, s_SP], axis=1).dropna()
df_quarterly.columns = ["MVM", "MV", "SP"]
quarters = df_quarterly.index
ind = np.arange(len(quarters))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(ind - width, df_quarterly["MVM"], width, color='seagreen', label='MVM (best)')
rects2 = ax.bar(ind,         df_quarterly["MV"],  width, color='dodgerblue', label='MV (best)')
rects3 = ax.bar(ind + width,   df_quarterly["SP"],  width, color='orange', label='S&P 500')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Quarter', fontsize=12)
ax.set_ylabel('Return', fontsize=12)
ax.set_title('Quarterly Returns Comparison (Best Hyper-parameters)', fontsize=14)
ax.set_xticks(ind)
ax.set_xticklabels([d.strftime("%Y-%m") for d in quarters], rotation=45)
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()

# -----------------------------
# Compute and Print Final Recommended Portfolio Weights
# -----------------------------
# For demonstration, we re-run the optimization using the last available data from the best rolling window.
# (This is done for both models separately.)
last_period_slice = returns.iloc[-best_rw_MVM:]
Sigma_last = last_period_slice.cov()
last_date = last_period_slice.index[-1]
risk_free_last = risk_free.loc[last_date]

# Compute CAPM betas for the last period.
betas_last = np.array([
    last_period_slice[stock].cov(market_returns.loc[last_period_slice.index].squeeze()) /
    market_returns.loc[last_period_slice.index].var() for stock in stocks
])
market_returns_last_mean = market_returns.loc[last_period_slice.index].mean()
capmt_returns_last = risk_free_last + betas_last * (market_returns_last_mean - risk_free_last)

# Select the top stocks using best_top_n_MVM.
top_idx_last = np.argsort(capmt_returns_last)[::-1][:best_top_n_MVM]
capmt_returns_last_selected = capmt_returns_last[top_idx_last]
selected_stocks_last = [stocks[j] for j in top_idx_last]

Sigma_last_selected = Sigma_last.loc[selected_stocks_last, selected_stocks_last]

# Compute the optimal weights using MVM and MV.
weights_MVM_last = mean_variance_opt(Sigma_last_selected, capmt_returns_last_selected, risk_aversion=3.0)
weights_MV_last = min_variance_opt(Sigma_last_selected)

print("\nLast recommended portfolio weights (MVM):")
print(pd.Series(weights_MVM_last, index=selected_stocks_last).round(4))

print("\nLast recommended portfolio weights (MV):")
print(pd.Series(weights_MV_last, index=selected_stocks_last).round(4))

# -----------------------------
# Compute Frequency of Selected Stocks & Plot Pie Charts
# -----------------------------
# Flatten the lists of selected stocks for each simulation period.
# -----------------------------
# Compute Frequency of Selected Stocks & Plot Pie Charts (Top 10)
# -----------------------------
# Flatten the lists of selected stocks for each simulation period.
flat_MVM = [stock for period in selected_list_MVM for stock in period]
flat_MV  = [stock for period in selected_list_MV for stock in period]

from collections import Counter
counter_MVM = Counter(flat_MVM)
counter_MV  = Counter(flat_MV)

# Get top 10 most common stocks (if available)
top10_MVM = dict(counter_MVM.most_common(10))
top10_MV  = dict(counter_MV.most_common(10))

# Plot pie chart for MVM selected stocks (Top 10)
plt.figure(figsize=(8,8))
plt.pie(top10_MVM.values(), labels=top10_MVM.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Top 10 Most Frequently Selected Stocks (MVM)')
plt.show()

# Plot pie chart for MV selected stocks (Top 10)
plt.figure(figsize=(8,8))
plt.pie(top10_MV.values(), labels=top10_MV.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Top 10 Most Frequently Selected Stocks (MV)')
plt.show() 

df_returns = pd.DataFrame({
    'MVM Period Return (%)': s_MVM * 100,
    'MV Period Return (%)': s_MV * 100,
    'SP Period Return (%)': s_SP * 100,
}, index=common_dates)

# Compute cumulative returns and convert them to percentage gains (starting at 0%)
df_returns['MVM Cumulative Return (%)'] = (1 + s_MVM).cumprod() * 100 - 100
df_returns['MV Cumulative Return (%)']  = (1 + s_MV).cumprod()  * 100 - 100
df_returns['SP Cumulative Return (%)']  = (1 + s_SP).cumprod()  * 100 - 100

# Print the table rounded to 2 decimal places
print("Period Returns and Cumulative Returns Comparison:")
print(df_returns[['MVM Period Return (%)','MV Period Return (%)','SP Period Return (%)']].round(2)) 
print(df_returns[['MVM Cumulative Return (%)','MV Cumulative Return (%)','SP Cumulative Return (%)']].round(2)) 
