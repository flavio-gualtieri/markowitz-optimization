import numpy as np
from typing import Optional, Dict, Tuple
import torch
# If nn.py is in the same directory, you can import the necessary functions and classes:
from nn import GARCHParameterNN, predict_parameters

class CovarianceForecaster:
    """
    Forecast a multi-asset covariance matrix by:
      1) Fitting a univariate GARCH(1,1) to each asset's return series
      2) Estimating correlations via rolling or constant correlation
      3) Combining forecasted variances with the correlation matrix 
         for the next-step covariance forecast.

    Attributes
    ----------
    returns : np.ndarray
        Array of shape (T, N), where T is number of time steps, N is number of assets.
    n_assets : int
        Number of assets.
    garch_params : dict
        Dictionary storing fitted GARCH(1,1) parameters for each asset, 
        keyed by asset index, e.g., garch_params[i] = (omega, alpha, beta).
    """

    def __init__(self, returns: np.ndarray, forecast_horizon: int = 1):
        """
        Constructor.

        Parameters
        ----------
        returns : np.ndarray
            Shape (T, N). Each column is an asset's return series.
        forecast_horizon : int
            Number of days in the future to forecast.
        """
        if returns.ndim != 2:
            raise ValueError("returns must be 2D (T, N).")
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.garch_params: Dict[int, Tuple[float, float, float]] = {}
        self.forecast_horizon = forecast_horizon


    def fit_all_univariate_garch_nn(self, nn_model_path="garch_nn_model.pth"):
        """
        Fit each asset’s GARCH(1,1) parameters using the trained ANN.
        For each asset, we compute:
          - σ²_emp: the in-sample variance of the asset returns,
          - Γ₄,emp: the empirical fourth standardized moment, computed as E[r⁴] / (E[r²])².
        
        Then, we use these as inputs to the NN to predict α₁ and subsequently compute β₁ and α₀.
        The fitted parameters are stored in self.garch_params.
        """
        # Assume the NN was trained on two input features: [σ²_emp, Γ₄,emp]
        input_dim = 2
        model = GARCHParameterNN(input_dim)
        model.load_state_dict(torch.load(nn_model_path, map_location=torch.device('cpu')))
        model.eval()

        for i in range(self.n_assets):
            r = self.returns[:, i]
            sigma2_emp = np.var(r)
            E_r2 = np.mean(r**2)
            E_r4 = np.mean(r**4)
            gamma4_emp = E_r4 / (E_r2**2) if E_r2 != 0 else 3.0  # default to 3 if variance is zero

            # Prepare input features as a 2D array (one sample)
            input_features = np.array([[sigma2_emp, gamma4_emp]])
            alpha0, alpha1, beta1 = predict_parameters(model, input_features, gamma4_emp, sigma2_emp)
            
            # Store parameters as a tuple (α₀, α₁, β₁) for asset i
            # (You might wish to extract scalars from the returned arrays if needed.)
            self.garch_params[i] = (float(alpha0[0]), float(alpha1[0]), float(beta1[0]))


    def forecast_variances(self) -> np.ndarray:
        """
        Produces a next-step (or multi-step) variance forecast for each asset 
        based on fitted GARCH(1,1) parameters.

        Parameters
        ----------
        horizon : int
            If 1, do a 1-step-ahead forecast; if >1, produce a horizon-step forecast
            with the usual GARCH iterative formula.

        Returns
        -------
        var_forecasts : np.ndarray
            Shape (N,). Forecasted variances for each asset.
        """
        horizon = self.forecast_horizon

        T = self.returns.shape[0]
        var_forecasts = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            omega, alpha, beta = self.garch_params[i]
            # Last known sigma^2:
            # We'll estimate by the last in-sample GARCH(1,1) recursion
            # or a short rolling variance
            series = self.returns[:, i]
            
            # Re-run forward pass to get final sigma^2 in-sample
            sigma2 = self._garch_in_sample(series, (omega, alpha, beta))

            last_sigma2 = sigma2[-1]
            # Forecast horizon steps:
            # GARCH(1,1) h-step ahead forecast formula for variance
            # sigma_{t+h}^2 = omega + (alpha+beta)*sigma_{t+h-1}^2 for each step
            next_var = last_sigma2
            for _ in range(horizon):
                next_var = omega + (alpha + beta)*next_var

            var_forecasts[i] = next_var

        return var_forecasts

    def _garch_in_sample(self, series: np.ndarray,
                         params: Tuple[float,float,float]) -> np.ndarray:
        """
        Re-run a GARCH(1,1) recursion on the entire in-sample returns, 
        given parameters. Return the sigma^2 path.
        """
        omega, alpha, beta = params
        T = len(series)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(series) if T>1 else omega/(1 - alpha - beta)

        for t in range(1, T):
            sigma2[t] = omega + alpha*series[t-1]**2 + beta*sigma2[t-1]
        return sigma2

    def estimate_correlation_matrix(self, window: int = 100) -> np.ndarray:
        """
        Estimates a rolling correlation matrix using the most recent 'window' returns 
        (or you can choose to do a constant correlation using all historical data).

        Parameters
        ----------
        window : int
            Size of the look-back window for correlation.

        Returns
        -------
        corr_mat : np.ndarray
            Shape (N, N). The correlation matrix among the assets 
            based on the last 'window' returns.
        """
        T = self.returns.shape[0]
        if window >= T:
            window = T  # use all data if the window is too large

        recent_data = self.returns[-window:, :]
        corr_mat = np.corrcoef(recent_data, rowvar=False)  # shape (N, N)
        return corr_mat

    def forecast_covariance(self,
                            window: int = 100,
                            correlation_override: Optional[np.ndarray] = None
                           ) -> np.ndarray:
        """
        Builds the forecasted covariance matrix for the next step (or horizon steps)
        by combining:
          1) Univariate GARCH variance forecasts
          2) Rolling (or constant) correlation matrix

        Parameters
        ----------
        window : int
            Rolling window size for correlation estimation.
        horizon : int
            Forecast horizon steps for GARCH.
        correlation_override : np.ndarray or None
            If provided, uses this matrix instead of the rolling correlation.
            Must be shape (N, N).

        Returns
        -------
        cov_mat_forecast : np.ndarray
            Shape (N, N). Forecasted covariance matrix.
        """
        # 1) Forecast each asset's variance
        var_forecasts = self.forecast_variances()  # shape (N,)
        std_forecasts = np.sqrt(var_forecasts)  # shape (N,)

        # 2) Correlation
        if correlation_override is None:
            corr_mat = self.estimate_correlation_matrix(window=window)
        else:
            corr_mat = correlation_override
        
        # 3) Build covariance = D * R * D
        D = np.diag(std_forecasts)
        cov_mat_forecast = D @ corr_mat @ D
        return cov_mat_forecast


if __name__ == "__main__":
    # EXAMPLE USAGE:

    # 1) Generate synthetic data: T=500 days, N=3 assets
    np.random.seed(42)
    T, N = 500, 3
    # Some correlated random returns:
    base_returns = np.random.randn(T, N)
    # Inject correlation by adding a shared factor:
    factor = np.random.randn(T,1)
    correlated_part = factor @ np.array([[0.5, 0.3, 0.1]])
    returns = base_returns * 0.01 + correlated_part*0.01

    # 2) Build forecaster and fit univariate GARCH
    forecaster = CovarianceForecaster(returns)
    forecaster.fit_all_univariate_garch()  # naive GARCH fit
    
    # 3) Forecast next-step covariance
    cov_forecast = forecaster.forecast_covariance(window=100)
    
    print("Forecasted Covariance Matrix (1-step ahead):")
    print(cov_forecast)
