import numpy as np
from typing import Optional, Dict, Tuple

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


    def fit_all_univariate_garch(self) -> None:
        """
        Fits a naive GARCH(1,1) model to each asset's return series independently.
        Stores parameters (omega, alpha, beta) in self.garch_params[i].
        
        This uses a minimal sum-of-squared-errors approach for demonstration.
        For real production, consider MLE or libraries like 'arch'.
        """
        for i in range(self.n_assets):
            r = self.returns[:, i]
            # Fit naive GARCH(1,1) and store results
            omega, alpha, beta = self._fit_garch_naive(r)
            self.garch_params[i] = (omega, alpha, beta)

    def _fit_garch_naive(self, series: np.ndarray) -> Tuple[float, float, float]:
        """
        Naive parameter estimation for GARCH(1,1) by a simple grid search or 
        sum-of-squares method (for illustration only).

        GARCH(1,1): sigma_t^2 = omega + alpha*r_{t-1}^2 + beta*sigma_{t-1}^2

        Returns
        -------
        (omega, alpha, beta) : tuple
            Fitted parameters for this asset.
        """
        # Hyper-parameters for the naive grid search
        # (In reality you might do a more robust optimization.)
        grid_omega = np.linspace(1e-6, 1e-3, 5)
        grid_alpha = np.linspace(0.01, 0.3, 5)
        grid_beta = np.linspace(0.3, 0.98, 5)

        best_loss = np.inf
        best_params = (0.0, 0.0, 0.0)

        # We'll simulate sigma_t^2 forward for each (omega, alpha, beta)
        # and measure sum of squared difference between sigma_t^2 and r_t^2.
        # This is extremely naive—just to illustrate the concept.
        for w in grid_omega:
            for a in grid_alpha:
                for b in grid_beta:
                    if a + b < 0.999:  # stationarity check
                        # Initialize
                        T = len(series)
                        sigma2 = np.zeros(T)
                        sigma2[0] = np.var(series)  # start guess
                        
                        # Forward pass
                        for t in range(1, T):
                            sigma2[t] = w + a*series[t-1]**2 + b*sigma2[t-1]

                        # Simple SSE loss
                        loss = np.mean((series**2 - sigma2)**2)
                        if loss < best_loss:
                            best_loss = loss
                            best_params = (w, a, b)

        return best_params

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
