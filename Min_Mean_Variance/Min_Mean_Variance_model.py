import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)  # Manage folder access to utilities

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import Counter
from Utilities.portfolio_utils import top_100

# -------------------------------
# Optimization Functions
# -------------------------------
def mean_variance_opt(Sigma: np.ndarray, mu: np.ndarray, risk_aversion: float = 3.0):
    """
    Compute the mean-variance optimal portfolio weights subject to the constraints of no short-selling (weights >= 0)
    and full investment (sum(weights) = 1).
    """
    n = len(mu)
    
    def objective(w):
        return - (np.dot(w, mu) - 0.5 * risk_aversion * np.dot(w, Sigma @ w))
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

def min_variance_opt(Sigma: np.ndarray):
    """
    Compute the minimum-variance portfolio weights subject to the constraints of no short-selling (weights >= 0)
    and full investment (sum(weights) = 1).
    """
    n = Sigma.shape[0]
    
    def objective(w):
        return np.dot(w, Sigma @ w)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

# -------------------------------
# Download Data & Prepare Returns
# -------------------------------
start_date = "2015-09-01"
end_date = "2024-09-01"
stocks = list(top_100.keys())  # Retrieve stocks from the utility file

# Download stock prices and compute quarterly returns
prices = yf.download(stocks, start=start_date, end=end_date, interval="1mo")['Adj Close']
prices = prices.resample("Q").last()
returns = prices.pct_change().dropna()
prices = prices.iloc[1:]  # Adjust for the missing observation from the percentage change calculation
print("Number of columns", len(prices.columns))

# Download market (S&P 500) data and compute returns
market_data = yf.download("^GSPC", start_date, end_date, interval="1mo")['Adj Close']
market_data = market_data.resample("Q").last()
market_returns = market_data.pct_change().dropna()

# Download risk-free rate data (using IRX as a proxy)
risk_free = yf.download("^IRX", start_date, end_date, interval="1mo")['Adj Close']
risk_free = risk_free.resample("Q").last().dropna()

# -------------------------------
# Simulation Functions
# -------------------------------
def simulate_time_series(rolling_window, top_n):
    """
    Run the simulation and return the time series (with dates) of portfolio returns
    for both models, as well as:
      - the list of selected stocks for each period, and
      - the history of portfolio weights for each period (as dictionaries mapping stock -> weight).
    """
    mvm_returns = []
    mv_returns = []
    date_list = []
    selected_stocks_list = []  # List of lists of stocks chosen at each period
    weights_history_MVM = []   # List of dictionaries for MVM model weights (one dict per period)
    weights_history_MV = []    # List of dictionaries for MV model weights (one dict per period)
    
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
        
        # Select top_n stocks (with the highest expected returns)
        top_idx = np.argsort(capmt_returns)[::-1][:top_n]
        capmt_returns_selected = capmt_returns[top_idx]
        selected_stocks = [stocks[j] for j in top_idx]
        selected_stocks_list.append(selected_stocks)
        
        Sigma_selected = Sigma.loc[selected_stocks, selected_stocks]
        try:
            optimal_weights_MVM = mean_variance_opt(Sigma_selected, capmt_returns_selected, risk_aversion=3.0)
            optimal_weights_MV = min_variance_opt(Sigma_selected)
        except Exception as e:
            continue
        
        # Save the portfolio weights as dictionaries (stock: weight)
        weight_dict_MVM = dict(zip(selected_stocks, optimal_weights_MVM))
        weight_dict_MV = dict(zip(selected_stocks, optimal_weights_MV))
        weights_history_MVM.append(weight_dict_MVM)
        weights_history_MV.append(weight_dict_MV)
        
        date_list.append(returns.index[i])
        next_return_vector = returns.iloc[i][selected_stocks].values
        mvm_returns.append(next_return_vector @ optimal_weights_MVM)
        mv_returns.append(next_return_vector @ optimal_weights_MV)
    
    return date_list, np.array(mvm_returns), np.array(mv_returns), selected_stocks_list, weights_history_MVM, weights_history_MV

# -------------------------------
# Grid Search over Hyper-parameters
# -------------------------------
rolling_window_list = [4, 6, 8, 10, 12]      # in quarters
top_n_list = [5, 10, 15, 20, 25]              # Number of stocks to select

results_MVM = np.zeros((len(rolling_window_list), len(top_n_list)))
results_MV = np.zeros((len(rolling_window_list), len(top_n_list)))

print("Starting grid search ...")
for i, rw in enumerate(rolling_window_list):
    for j, tn in enumerate(top_n_list):
        print(f"Simulating for rolling window = {rw} and top_n = {tn} ...")
        # We use the performance simulation here (only cumulative returns are needed)
        # (You could also modify simulate_time_series if needed.)
        # For brevity, here we use simulate_time_series and ignore weights
        _, mvm_ret, mv_ret, _, _, _ = simulate_time_series(rw, tn)
        results_MVM[i, j] = np.prod(1 + mvm_ret)
        results_MV[i, j] = np.prod(1 + mv_ret)

df_MVM = pd.DataFrame(results_MVM, index=rolling_window_list, columns=top_n_list)
df_MV = pd.DataFrame(results_MV, index=rolling_window_list, columns=top_n_list) 

# -------------------------------
# Plot Grid Search Results
# -------------------------------
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
for tn in top_n_list:
    plt.plot(rolling_window_list, df_MV[tn], marker='o', label=f'top_n = {tn}')
plt.xlabel('Rolling Window Size (quarters)')
plt.ylabel('Final Cumulative Return')
plt.title('Grid Search: MV Performance vs. Rolling Window Size')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Select Best Hyper-parameters
# -------------------------------
best_idx_MVM = np.unravel_index(np.argmax(results_MVM, axis=None), results_MVM.shape)
best_rw_MVM = rolling_window_list[best_idx_MVM[0]]
best_top_n_MVM = top_n_list[best_idx_MVM[1]]
print(f"Best hyper-parameters for MVM: Rolling Window = {best_rw_MVM}, Top_n = {best_top_n_MVM}")

best_idx_MV = np.unravel_index(np.argmax(results_MV, axis=None), results_MV.shape)
best_rw_MV = rolling_window_list[best_idx_MV[0]]
best_top_n_MV = top_n_list[best_idx_MV[1]]
print(f"Best hyper-parameters for MV: Rolling Window = {best_rw_MV}, Top_n = {best_top_n_MV}")

# -------------------------------
# Re-estimate Models with Best Hyper-parameters and Produce Performance Plots
# -------------------------------
# For the best MVM model
dates_MVM, mvm_returns_series, _, selected_list_MVM, weights_history_MVM, _ = simulate_time_series(best_rw_MVM, best_top_n_MVM)
# For the best MV model
dates_MV, _, mv_returns_series, selected_list_MV, _, weights_history_MV = simulate_time_series(best_rw_MV, best_top_n_MV)

# Align Time Series to a Common Date Range 
s_MVM = pd.Series(mvm_returns_series, index=dates_MVM)
s_MV  = pd.Series(mv_returns_series, index=dates_MV)
common_dates = s_MVM.index.intersection(s_MV.index)
s_MVM = s_MVM.loc[common_dates]
s_MV = s_MV.loc[common_dates]
s_SP = market_returns.loc[common_dates]

# Compute cumulative returns (in percentage)
cum_ret_MVM_pct = (1 + s_MVM).cumprod() * 100 - 100
cum_ret_MV_pct  = (1 + s_MV).cumprod()  * 100 - 100
cum_ret_SP_pct  = (1 + s_SP).cumprod()  * 100 - 100

plt.figure(figsize=(12, 6))
plt.plot(common_dates, cum_ret_MVM_pct, label="MVM (best params)", color="blue", linewidth=2)
plt.plot(common_dates, cum_ret_MV_pct,  label="MV (best params)",  color="green", linewidth=2)
plt.plot(common_dates, cum_ret_SP_pct,  label="S&P 500",       color="orange", linestyle="--", linewidth=2)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Return (%)", fontsize=12)
plt.title("Cumulative Returns: MVM vs MV vs S&P 500 (Best Hyper-parameters)", fontsize=14)
plt.grid(alpha=0.5)
plt.legend(fontsize=10)
plt.show()

# -------------------------------
# Create Grouped Bar Chart for Quarterly Returns
# -------------------------------
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

# -------------------------------
# Compute Frequency of Selected Stocks & Plot Pie Charts
# -------------------------------
flat_MVM = [stock for period in selected_list_MVM for stock in period]
flat_MV  = [stock for period in selected_list_MV for stock in period]

counter_MVM = Counter(flat_MVM)
counter_MV  = Counter(flat_MV)

top10_MVM = dict(counter_MVM.most_common(10))
top10_MV  = dict(counter_MV.most_common(10))

quant_colors = list(plt.get_cmap('tab10').colors)

fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
for ax in axs:
    ax.set_facecolor('white')
explode = [0.05] * 10

patches, texts, autotexts = axs[0].pie(
    list(top10_MVM.values()),
    labels=list(top10_MVM.keys()),
    autopct='%1.1f%%',
    startangle=140,
    colors=quant_colors,
    shadow=True,
    explode=explode,
    textprops={'fontsize': 12, 'color': 'black'}
)
axs[0].set_title('Top 10 Most Frequently Selected Stocks (MVM)', fontsize=16, color='black')

patches, texts, autotexts = axs[1].pie(
    list(top10_MV.values()),
    labels=list(top10_MV.keys()),
    autopct='%1.1f%%',
    startangle=140,
    colors=quant_colors,
    shadow=True,
    explode=explode,
    textprops={'fontsize': 12, 'color': 'black'}
)
axs[1].set_title('Top 10 Most Frequently Selected Stocks (MV)', fontsize=16, color='black')
plt.tight_layout()
plt.show()

df_returns = pd.DataFrame({
    'MVM Period Return (%)': s_MVM * 100,
    'MV Period Return (%)': s_MV * 100,
    'SP Period Return (%)': s_SP * 100,
}, index=common_dates)

df_returns['MVM Cumulative Return (%)'] = (1 + s_MVM).cumprod() * 100 - 100
df_returns['MV Cumulative Return (%)']  = (1 + s_MV).cumprod()  * 100 - 100
df_returns['SP Cumulative Return (%)']  = (1 + s_SP).cumprod()  * 100 - 100

print("Period Returns and Cumulative Returns Comparison:")
print(df_returns[['MVM Period Return (%)', 'MV Period Return (%)', 'SP Period Return (%)']].round(2))
print(df_returns[['MVM Cumulative Return (%)', 'MV Cumulative Return (%)', 'SP Cumulative Return (%)']].round(2))

# -------------------------------
# Save the Entire History of Weights for Best Models
# -------------------------------
def save_weights_history(dates, weights_history, filename):
    """
    Convert a list of weight dictionaries (one per date) into a DataFrame and save it as CSV.
    Missing stocks in a period are filled with 0.
    """
    all_stocks = set()
    for weight_dict in weights_history:
        all_stocks.update(weight_dict.keys())
    all_stocks = sorted(all_stocks)
    
    # Create a DataFrame with index = dates and columns = all_stocks
    weights_df = pd.DataFrame(index=dates, columns=all_stocks)
    for date, weight_dict in zip(dates, weights_history):
        for stock in all_stocks:
            weights_df.loc[date, stock] = weight_dict.get(stock, 0)
    weights_df = weights_df.fillna(0)
    weights_df.to_csv(filename)
    print(f"Saved weights history to '{filename}'.")

