import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

# Import our backtest modules (assumed to be in the same directory)
from covariance_forecaster import CovarianceForecaster
from model import MarkRebalancing
from nn import train_model, predict_parameters

def train_garch_nn_for_asset(asset_returns, stock_symbol, window_size=30, target_alpha1=0.15):
    """
    Train a neural network for a single asset using rolling windows from the six-month training period.
    
    Parameters:
        asset_returns (np.ndarray): 1D array of asset returns over six months.
        stock_symbol (str): The ticker symbol to create a unique filename.
        window_size (int): Number of observations per training sample (e.g., 30 days).
        target_alpha1 (float): Placeholder target for alpha1 (replace with MLE if available).
        
    Returns:
        tuple: (alpha0, alpha1, beta1) predicted for the asset using the NN.
    """
    n_obs = len(asset_returns)
    features = []
    targets = []
    
    # Create training samples using non-overlapping windows
    for start in range(0, n_obs - window_size + 1, window_size):
        window_data = asset_returns[start:start+window_size]
        sigma2_emp = np.var(window_data)
        E_r2 = np.mean(window_data**2)
        E_r4 = np.mean(window_data**4)
        gamma4_emp = E_r4 / (E_r2**2) if E_r2 != 0 else 3.0
        features.append([sigma2_emp, gamma4_emp])
        targets.append(target_alpha1)  # Replace with a proper estimator if available
    
    features = np.array(features)
    targets = np.array(targets)
    
    # Build a unique save path for this stock's model
    save_filename = f"garch_nn_{stock_symbol}.pth"
    
    # Train the network using the training samples for this asset and save the model
    model = train_model(features, targets, input_dim=2, num_epochs=500, learning_rate=0.01, 
                        patience=50, save_path=save_filename)
    
    # Use the most recent window (last window_size observations) to compute the input features
    recent_data = asset_returns[-window_size:]
    latest_sigma2 = np.var(recent_data)
    E_r2_recent = np.mean(recent_data**2)
    E_r4_recent = np.mean(recent_data**4)
    latest_gamma4 = E_r4_recent / (E_r2_recent**2) if E_r2_recent != 0 else 3.0
    input_features = np.array([[latest_sigma2, latest_gamma4]])
    
    # Predict the GARCH parameters using the trained model
    alpha0_pred, alpha1_pred, beta_pred = predict_parameters(model, input_features, latest_gamma4, latest_sigma2)
    return float(alpha0_pred[0]), float(alpha1_pred[0]), float(beta_pred[0])

# ---------------------------
# SETTINGS & SIMULATED DATA
# ---------------------------
# Fixed list of stocks
stocks = ["AAPL", "MSFT", "GOOGL"]
num_assets = len(stocks)

# For demonstration, simulate daily returns for 9 months (from 2021-01-01 to 2021-09-30)
start_date = pd.to_datetime("2021-01-01")
end_date = pd.to_datetime("2021-09-30")
dates = pd.date_range(start_date, end_date, freq="B")  # business days
T = len(dates)

# Simulate returns: each asset has daily returns ~ N(0, 1%) (adjust as desired)
np.random.seed(42)
returns_array = np.random.randn(T, num_assets) * 0.01  
returns_data = pd.DataFrame(returns_array, index=dates, columns=stocks)

# ---------------------------
# BACKTEST PARAMETERS
# ---------------------------
# Backtest: training window = 6 months, testing period = 3 months
backtest_start = pd.to_datetime("2021-07-01")
backtest_end = pd.to_datetime("2021-09-30")
training_window_days = 126  # approx six months (approx 21 trading days per month)
rebalance_frequency = 5     # rebalance every 5 days
initial_portfolio_value = 100.0

# ---------------------------
# BACKTEST LOOP
# ---------------------------
portfolio_value = initial_portfolio_value
portfolio_history = []  # list of tuples: (date, portfolio value)
rebalance_dates = []
weights_history = []    # store optimal weights at each rebalancing

# Create a MarkRebalancing instance (it will update its weights every rebalance)
mark_model = MarkRebalancing(n_assets=num_assets)

# Set current_date to start the backtest
current_date = backtest_start

while current_date <= backtest_end:
    # Define training window: last training_window_days before current_date
    training_start_date = current_date - pd.Timedelta(days=training_window_days)
    training_data = returns_data.loc[training_start_date:current_date]
    
    # Skip if insufficient training data
    if len(training_data) < 30:
        current_date += pd.Timedelta(days=rebalance_frequency)
        continue
    
    training_returns = training_data.values  # shape (T_train, num_assets)
    
    # Set forecast horizon to the rebalancing frequency (h days)
    forecaster = CovarianceForecaster(training_returns, forecast_horizon=rebalance_frequency)
    
    # Train a NN for each asset (Option A) with stock-specific model saving.
    garch_params = {}
    for i in range(num_assets):
        asset_returns = training_returns[:, i]  # 1D array of returns for asset i
        stock_symbol = stocks[i]
        params = train_garch_nn_for_asset(asset_returns, stock_symbol, window_size=30, target_alpha1=0.15)
        garch_params[i] = params

    # Assign these parameters to the forecaster instance
    forecaster.garch_params = garch_params
    
    # Forecast the covariance matrix over the next h days
    cov_forecast = forecaster.forecast_covariance(window=100)
    
    # Compute expected returns from the training window (historical mean)
    expected_returns = np.mean(training_returns, axis=0)
    
    # Rebalance the portfolio using the Markowitz optimizer (note: no date parameter here)
    new_weights = mark_model.rebalance(expected_returns, cov_forecast)
    weights_history.append(new_weights)
    rebalance_dates.append(current_date)
    
    # Determine the simulation period: next rebalance_frequency days
    simulation_end_date = current_date + pd.Timedelta(days=rebalance_frequency)
    period_data = returns_data.loc[current_date:simulation_end_date]
    if period_data.empty:
        break
    
    # Compute daily portfolio returns and compound them over the period
    daily_portfolio_returns = period_data.dot(new_weights)
    period_return = np.prod(1 + daily_portfolio_returns) - 1
    portfolio_value = portfolio_value * (1 + period_return)
    portfolio_history.append((simulation_end_date, portfolio_value))
    
    # Move forward to the next rebalancing date
    current_date = simulation_end_date

# Convert portfolio history to DataFrame for plotting
portfolio_df = pd.DataFrame(portfolio_history, columns=["Date", "Portfolio Value"]).set_index("Date")

# For the prediction visualization, show the latest optimal weights.
latest_weights = pd.Series(new_weights, index=stocks)

# ---------------------------
# STREAMLIT VISUALIZATION
# ---------------------------
st.title("Backtest Performance & Prediction Visualizations")

st.subheader("Portfolio Value Over Time")
st.line_chart(portfolio_df)

st.subheader("Latest Optimal Weights")
st.bar_chart(latest_weights)

# Optional: Display the latest forecast covariance matrix as a heatmap.
st.subheader("Latest Forecast Covariance Matrix")
fig, ax = plt.subplots()
cax = ax.matshow(cov_forecast, cmap="viridis")
fig.colorbar(cax)
ax.set_xticks(range(num_assets))
ax.set_xticklabels(stocks)
ax.set_yticks(range(num_assets))
ax.set_yticklabels(stocks)
st.pyplot(fig)
