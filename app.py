import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

from covariance_forecaster import CovarianceForecaster
from model import MarkRebalancing
from nn import train_model, predict_parameters
from data_handling import PolygonDataHandler

stocks = ["AAPL", "MSFT", "GOOGL", "QQQ", "AMD"]
num_assets = len(stocks)

POLYGON_API_KEY = ""

backtest_start = pd.to_datetime("2023-07-01")
backtest_end = pd.to_datetime("2023-09-30")
start_date_hist = pd.to_datetime("2021-01-01")
end_date_hist = backtest_end 

training_window_days = 126 
rebalance_frequency = 5
initial_portfolio_value = 10000.0

polygon_handler = PolygonDataHandler(POLYGON_API_KEY)

@st.cache_data
def fetch_historical_data(tickers, start_date, end_date):
    all_stock_data = {}
    for ticker in tickers:
        try:
            df = polygon_handler.fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if df is None or df.empty:
                st.error(f"No data fetched for {ticker} from {start_date} to {end_date}. Please check ticker and date range.")
                return None
            all_stock_data[ticker] = df
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None
    return all_stock_data

historical_data = fetch_historical_data(stocks, start_date_hist, end_date_hist)

if historical_data is None:
    st.stop()

returns_data = pd.DataFrame(index=None)
for stock in stocks:
    if historical_data and stock in historical_data:
        returns = polygon_handler.compute_returns(historical_data[stock])
        if returns is not None and len(returns) > 0:
            returns_data[stock] = pd.Series(returns, index=historical_data[stock].index[1:])
        else:
            st.warning(f"Could not compute returns for {stock}. Check data.")
    else:
        st.warning(f"No historical data available for {stock} to compute returns.")

if returns_data.empty or returns_data.isnull().all().all():
    st.error("Could not compute returns for any of the stocks. Backtest cannot proceed.")
    st.stop()

if not isinstance(returns_data.index, pd.DatetimeIndex):
    try:
        returns_data.index = pd.to_datetime(returns_data.index)
    except Exception as e:
        st.error(f"Error converting returns_data index to DateTimeIndex: {e}")
        st.stop()

def train_garch_nn_for_asset(asset_returns, stock_symbol, window_size=30, target_alpha1=0.15):
    n_obs = len(asset_returns)
    features = []
    targets = []
    for start in range(0, n_obs - window_size + 1, window_size):
        window_data = asset_returns[start:start+window_size]
        sigma2_emp = np.var(window_data)
        E_r2 = np.mean(window_data**2)
        E_r4 = np.mean(window_data**4)
        gamma4_emp = E_r4 / (E_r2**2) if E_r2 != 0 else 3.0
        features.append([sigma2_emp, gamma4_emp])
        targets.append(target_alpha1)
    features = np.array(features)
    targets = np.array(targets)
    save_filename = f"garch_nn_{stock_symbol}.pth"
    model = train_model(features, targets, input_dim=2, num_epochs=500, learning_rate=0.01, patience=50, save_path=save_filename)
    recent_data = asset_returns[-window_size:]
    latest_sigma2 = np.var(recent_data)
    E_r2_recent = np.mean(recent_data**2)
    E_r4_recent = np.mean(recent_data**4)
    latest_gamma4 = E_r4_recent / (E_r2_recent**2) if E_r2_recent != 0 else 3.0
    input_features = np.array([[latest_sigma2, latest_gamma4]])
    alpha0_pred, alpha1_pred, beta_pred = predict_parameters(model, input_features, latest_gamma4, latest_sigma2)
    return float(alpha0_pred[0]), float(alpha1_pred[0]), float(beta_pred[0])

portfolio_value = initial_portfolio_value
portfolio_history = []
rebalance_dates = []
weights_history = []
mark_model = MarkRebalancing(n_assets=num_assets)
current_date = backtest_start

while current_date <= backtest_end:
    training_start_date = current_date - pd.Timedelta(days=training_window_days)
    training_data_slice = returns_data.loc[training_start_date:current_date]
    if len(training_data_slice) < 30:
        current_date += pd.Timedelta(days=rebalance_frequency)
        continue
    training_returns = training_data_slice.values
    forecaster = CovarianceForecaster(training_returns, forecast_horizon=rebalance_frequency)
    garch_params = {}
    for i in range(num_assets):
        asset_returns = training_returns[:, i]
        stock_symbol = stocks[i]
        params = train_garch_nn_for_asset(asset_returns, stock_symbol, window_size=30, target_alpha1=0.15)
        garch_params[i] = params
    forecaster.garch_params = garch_params
    cov_forecast = forecaster.forecast_covariance(window=100)
    expected_returns = np.mean(training_returns, axis=0)
    new_weights = mark_model.rebalance(expected_returns, cov_forecast)
    weights_history.append(new_weights)
    rebalance_dates.append(current_date)
    simulation_end_date = current_date + pd.Timedelta(days=rebalance_frequency)
    period_data = returns_data.loc[current_date:simulation_end_date]
    if period_data.empty:
        break
    daily_portfolio_returns = period_data.dot(new_weights)
    period_return = np.prod(1 + daily_portfolio_returns) - 1
    portfolio_value = portfolio_value * (1 + period_return)
    portfolio_history.append((simulation_end_date, portfolio_value))
    current_date = simulation_end_date

portfolio_df = pd.DataFrame(portfolio_history, columns=["Date", "Portfolio Value"]).set_index("Date")
weights_df = pd.DataFrame(weights_history, columns=stocks, index=rebalance_dates)
latest_weights = pd.Series(new_weights, index=stocks)

st.title("Backtest Performance & Prediction Visualizations")
st.subheader("Portfolio Value Over Time")
st.line_chart(portfolio_df)
st.subheader("Latest Optimal Weights")
st.bar_chart(latest_weights)
st.subheader("Historical Portfolio Weights")
st.line_chart(weights_df)
st.subheader("Latest Forecast Covariance Matrix")
fig, ax = plt.subplots()
cax = ax.matshow(cov_forecast, cmap="viridis")
fig.colorbar(cax)
ax.set_xticks(range(num_assets))
ax.set_xticklabels(stocks)
ax.set_yticks(range(num_assets))
ax.set_yticklabels(stocks)
st.pyplot(fig)
