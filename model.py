import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
from tools import Tools  # assuming tools.py is in the same directory

class MarkRebalancing:
    """
    A class that implements Markowitz-based rebalancing.
    
    It uses forecast covariance matrices (from covariance_forecaster.py)
    and an expected returns vector to generate optimal portfolio weights
    via quadratic programming (cvxopt). The class maintains internal state
    for current weights and a history of weights for analysis.
    """
    
    def __init__(self, n_assets: int, gamma: float = 1.0, initial_weights: np.ndarray = None):
        """
        Initialize the MarkRebalancing model.
        
        Parameters
        ----------
        n_assets : int
            Number of assets in the portfolio.
        gamma : float, optional
            Risk-aversion parameter (not directly used in the QP below,
            but available for future extensions), by default 1.0.
        initial_weights : np.ndarray, optional
            Initial portfolio weights. If None, equal weights are used.
        """
        self.n_assets = n_assets
        self.gamma = gamma
        if initial_weights is not None:
            if len(initial_weights) != n_assets:
                raise ValueError("Length of initial_weights must equal n_assets.")
            self.current_weights = np.array(initial_weights)
        else:
            # Option 1: use equal weights...
            # self.current_weights = np.ones(n_assets) / n_assets
            # Option 2: or use Tools.rand_weights for a random initialization
            self.current_weights = Tools.rand_weights(n_assets)
            
        self.weights_history = [self.current_weights.copy()]
        
        # Turn off cvxopt solver progress output for clarity.
        solvers.options['show_progress'] = False

    def compute_optimal_weights(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> tuple[np.ndarray, list, list]:
        """
        Compute the optimal portfolio weights given expected returns and forecast covariance.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            1D array of expected returns for each asset (shape: (n_assets,)).
        forecast_cov : np.ndarray
            2D forecast covariance matrix (shape: (n_assets, n_assets)).
            
        Returns
        -------
        w_opt : np.ndarray
            Optimal portfolio weights (1D array of shape (n_assets,)).
        frontier_returns : list
            Portfolio returns along the efficient frontier.
        frontier_risks : list
            Portfolio risks (standard deviations) along the efficient frontier.
        """
        n = self.n_assets
        # Convert the forecast covariance and expected returns to cvxopt matrices.
        S = opt.matrix(forecast_cov)
        pbar = opt.matrix(expected_returns)
        
        # Define constraints: no shorting (w_i >= 0) and full investment (sum(w_i) = 1).
        G = -opt.matrix(np.eye(n))
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
        
        # Build an efficient frontier by varying a risk-aversion parameter mu.
        N = 100
        mus = [10**(5.0 * t / N - 1.0) for t in range(N)]
        portfolios = []
        for mu in mus:
            sol = solvers.qp(mu * S, -pbar, G, h, A, b)
            portfolios.append(sol['x'])
            
        # Calculate frontier returns and risks.
        frontier_returns = [blas.dot(pbar, x) for x in portfolios]
        frontier_risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        
        # Fit a quadratic (second-degree polynomial) to the frontier points.
        m1 = np.polyfit(frontier_returns, frontier_risks, 2)
        # Compute a candidate risk-aversion parameter.
        x1 = np.sqrt(m1[2] / m1[0])
        sol_min_var = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)
        w_opt = np.asarray(sol_min_var['x']).flatten()
        
        return w_opt, frontier_returns, frontier_risks
        
    def rebalance(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> np.ndarray:
        """
        Rebalance the portfolio based on the provided expected returns and forecast covariance.
        The method updates the internal current weights and appends them to a history.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            1D array of expected returns (shape: (n_assets,)).
        forecast_cov : np.ndarray
            2D forecast covariance matrix (shape: (n_assets, n_assets)).
            
        Returns
        -------
        new_weights : np.ndarray
            The newly computed optimal portfolio weights.
        """
        new_weights, _, _ = self.compute_optimal_weights(expected_returns, forecast_cov)
        self.current_weights = new_weights
        self.weights_history.append(new_weights.copy())
        return new_weights
        
    def get_current_weights(self) -> np.ndarray:
        """
        Returns the current portfolio weights.
        """
        return self.current_weights
        
    def get_weights_history(self) -> list:
        """
        Returns the history of portfolio weights.
        """
        return self.weights_history
