import numpy as np
from typing import Optional, Dict, Tuple
import torch
from nn import GARCHParameterNN, predict_parameters

class CovarianceForecaster:

    def __init__(self, returns: np.ndarray, forecast_horizon: int = 1):
        if returns.ndim != 2:
            raise ValueError("returns must be 2D (T, N).")
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.garch_params: Dict[int, Tuple[float, float, float]] = {}
        self.forecast_horizon = forecast_horizon


    def fit_all_univariate_garch_nn(self, nn_model_path="garch_nn_model.pth"):
        input_dim = 2
        model = GARCHParameterNN(input_dim)
        model.load_state_dict(torch.load(nn_model_path, map_location=torch.device('cpu')))
        model.eval()

        for i in range(self.n_assets):
            r = self.returns[:, i]
            sigma2_emp = np.var(r)
            E_r2 = np.mean(r**2)
            E_r4 = np.mean(r**4)
            gamma4_emp = E_r4 / (E_r2**2) if E_r2 != 0 else 3.0
            input_features = np.array([[sigma2_emp, gamma4_emp]])
            alpha0, alpha1, beta1 = predict_parameters(model, input_features, gamma4_emp, sigma2_emp)
            self.garch_params[i] = (float(alpha0[0]), float(alpha1[0]), float(beta1[0]))


    def forecast_variances(self) -> np.ndarray:
        horizon = self.forecast_horizon
        T = self.returns.shape[0]
        var_forecasts = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            omega, alpha, beta = self.garch_params[i]
            print(f"DEBUG: Asset {i} GARCH parameters: omega={omega}, alpha={alpha}, beta={beta}")
            series = self.returns[:, i]
            sigma2 = self._garch_in_sample(series, (omega, alpha, beta))
            last_sigma2 = sigma2[-1]
            print(f"DEBUG: Asset {i} Last sigma2 (in-sample): {last_sigma2}")
            next_var = last_sigma2
            for _ in range(horizon):
                next_var = omega + (alpha + beta) * next_var
                print(f"DEBUG: Asset {i} Iterative variance forecast: {next_var}, omega={omega}, alpha={alpha}, beta={beta}, next_var_prev={next_var - (alpha+beta)*next_var if horizon > 1 else last_sigma2}")
                for _ in range(horizon):
                    next_var = omega + (alpha + beta) * next_var
                    if next_var < 0:
                        next_var = 0
                var_forecasts[i] = next_var
            var_forecasts[i] = next_var

        return var_forecasts


    def _garch_in_sample(self, series: np.ndarray, params: Tuple[float, float, float]) -> np.ndarray:
        omega, alpha, beta = params
        T = len(series)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(series) if T > 1 else omega / (1 - alpha - beta)
        for t in range(1, T):
            sigma2[t] = omega + alpha * series[t - 1]**2 + beta * sigma2[t - 1]
        return sigma2


    def estimate_correlation_matrix(self, window: int = 100) -> np.ndarray:
        T = self.returns.shape[0]
        if window >= T:
            window = T
        recent_data = self.returns[-window:, :]
        corr_mat = np.corrcoef(recent_data, rowvar=False)
        return corr_mat


    def forecast_covariance(self, window: int = 100, correlation_override: Optional[np.ndarray] = None) -> np.ndarray:
        var_forecasts = self.forecast_variances()
        std_forecasts = np.sqrt(var_forecasts)
        print("DEBUG: Variance Forecasts:")
        print(var_forecasts)
        print("DEBUG: Standard Deviation Forecasts:")
        print(std_forecasts)
        if correlation_override is None:
            corr_mat = self.estimate_correlation_matrix(window=window)
        else:
            corr_mat = correlation_override
        print("DEBUG: Correlation Matrix:")
        print(corr_mat)
        D = np.diag(std_forecasts)
        cov_mat_forecast = D @ corr_mat @ D
        return cov_mat_forecast