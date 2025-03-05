# Portfolio Rebalancing with Markowitz and GARCH(1,1) Neural Network

## Overview
This project implements a portfolio rebalancing strategy that uses Markowitz optimization along with a forecasted covariance matrix. The covariance forecast is obtained by modeling each asset’s variance with a GARCH(1,1) model whose parameters are fine-tuned via a neural network. For every portfolio construction, the NN is applied on the 6 months (approximately 126 trading days) of historical data preceding the start date.

## Project Structure
The project is organized into six scripts:
- **tools.py**  
  Contains utility functions (e.g., setting random seeds, generating random portfolio weights, and simulating portfolios).

- **covariance_forecaster.py**  
  Implements a system to forecast the covariance matrix by:
  - Fitting a univariate GARCH(1,1) model to each asset’s return series.
  - Using a neural network (from `nn.py`) to predict the GARCH parameters based on empirical moments.
  - Combining the forecasted variances with a rolling (or constant) correlation matrix.

- **nn.py**  
  Defines the neural network architecture for predicting the key GARCH parameter (α₁) and includes functions to analytically compute β₁ and α₀. It also provides a training routine that uses simulated data; this network is intended to be fine-tuned on each stock using data from the preceding 6 months.

- **model.py**  
  Implements a Markowitz-based portfolio rebalancing strategy. It uses quadratic programming (via cvxopt) to compute the optimal asset weights given forecasted covariances and expected returns.

- **data_handling.py**  
  Handles data retrieval from polygon.io, including functions to:
  - Fetch aggregated price data.
  - Compute returns and higher moments (variance and fourth standardized moment) needed for the NN.

- **app.py**  
  Provides a Streamlit-based front-end for:
  - Running backtests.
  - Rebalancing the portfolio at fixed intervals.
  - Visualizing the portfolio performance and the latest optimal weights.
  
  In the backtest loop, for every rebalancing the training data is sliced from the last 6 months, ensuring that the NN is effectively “trained” (or at least applied) for each asset using the most recent 6 months of data.

## How It Works
1. **Training Window:**  
   For every portfolio rebalancing, the NN uses the 6 months of historical data (approximately 126 trading days) preceding the start date. It computes empirical moments (variance and fourth standardized moment) from this window and then predicts the GARCH(1,1) parameters for each asset.

2. **GARCH Forecasting:**  
   The `CovarianceForecaster` class in *covariance_forecaster.py* applies the NN (loaded from `garch_nn_model.pth`) to fine-tune the GARCH parameters. These parameters are then used to forecast next-step (or multi-step) variances for each asset.

3. **Portfolio Optimization:**  
   With the forecasted variances and a rolling correlation matrix, the forecasted covariance matrix is built. The Markowitz optimizer in *model.py* then uses this matrix along with historical mean returns to compute optimal portfolio weights via quadratic programming.

4. **Backtesting & Visualization:**  
   The *app.py* script runs a backtest over a specified period, rebalancing the portfolio every few days, and visualizes both the portfolio value over time and the current asset allocations using Streamlit.

## Setup and Installation

1. **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install Dependencies:**
    Install the required packages with:
    ```bash
    pip install -r requirements.txt
    ```

3. **Obtain API Key:**
    For data retrieval, register at [polygon.io](https://polygon.io) to obtain an API key. Then update the API key in `data_handling.py`.


4. **Run the Application:**
    Start the Streamlit application to simulate the backtest and view visualizations:
    ```bash
    streamlit run app.py
    ```

## Notes
- **NN Training on a 6-Month Window:**  
  Each time the portfolio is constructed, the model uses the most recent 6 months of data to compute empirical moments and predict GARCH parameters for every asset.
- **Backtest Parameters:**  
  Adjust the `training_window_days` and `rebalance_frequency` in *app.py* to better fit your needs.
- **Data Source:**  
  The project uses polygon.io for price data. Ensure that your API key is correctly set.

