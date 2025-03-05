import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
from tools import Tools

class MarkRebalancing:
    def __init__(self, n_assets: int, gamma: float = 1.0, initial_weights: np.ndarray = None):
        self.n_assets = n_assets
        self.gamma = gamma
        if initial_weights is not None:
            if len(initial_weights) != n_assets:
                raise ValueError("Length of initial_weights must equal n_assets.")
            self.current_weights = np.array(initial_weights)
        else:
            self.current_weights = Tools.rand_weights(n_assets)
        self.weights_history = [self.current_weights.copy()]
        solvers.options['show_progress'] = False

    def compute_optimal_weights(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> tuple[np.ndarray, list, list]:
        n = self.n_assets
        S = opt.matrix(forecast_cov)
        pbar = opt.matrix(expected_returns)
        print("DEBUG: Expected Returns (pbar):")
        print(pbar)
        print("DEBUG: Covariance Matrix (S):")
        print(S)
        G = -opt.matrix(np.eye(n))
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
        N = 100
        mus = [10**(5.0 * t / N - 1.0) for t in range(N)]
        portfolios = []
        for mu in mus:
            sol = solvers.qp(mu * S, -pbar, G, h, A, b)
            portfolios.append(sol['x'])
        frontier_returns = [blas.dot(pbar, x) for x in portfolios]
        frontier_risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        m1 = np.polyfit(frontier_returns, frontier_risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        sol_min_var = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)
        w_opt = np.asarray(sol_min_var['x']).flatten()
        return w_opt, frontier_returns, frontier_risks

    def rebalance(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> np.ndarray:
        new_weights, _, _ = self.compute_optimal_weights(expected_returns, forecast_cov)
        self.current_weights = new_weights
        self.weights_history.append(new_weights.copy())
        return new_weights

    def get_current_weights(self) -> np.ndarray:
        return self.current_weights

    def get_weights_history(self) -> list:
        return self.weights_history
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
from tools import Tools

class MarkRebalancing:
    def __init__(self, n_assets: int, gamma: float = 1.0, initial_weights: np.ndarray = None):
        self.n_assets = n_assets
        self.gamma = gamma
        if initial_weights is not None:
            if len(initial_weights) != n_assets:
                raise ValueError("Length of initial_weights must equal n_assets.")
            self.current_weights = np.array(initial_weights)
        else:
            self.current_weights = Tools.rand_weights(n_assets)
        self.weights_history = [self.current_weights.copy()]
        solvers.options['show_progress'] = False

    def compute_optimal_weights(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> tuple[np.ndarray, list, list]:
        n = self.n_assets
        S = opt.matrix(forecast_cov)
        pbar = opt.matrix(expected_returns)
        print("DEBUG: Expected Returns (pbar):")
        print(pbar)
        print("DEBUG: Covariance Matrix (S):")
        print(S)
        G = -opt.matrix(np.eye(n))
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
        N = 100
        mus = [10**(5.0 * t / N - 1.0) for t in range(N)]
        portfolios = []
        for mu in mus:
            sol = solvers.qp(mu * S, -pbar, G, h, A, b)
            portfolios.append(sol['x'])
        frontier_returns = [blas.dot(pbar, x) for x in portfolios]
        frontier_risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        m1 = np.polyfit(frontier_returns, frontier_risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        sol_min_var = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)
        w_opt = np.asarray(sol_min_var['x']).flatten()
        return w_opt, frontier_returns, frontier_risks

    def rebalance(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> np.ndarray:
        new_weights, _, _ = self.compute_optimal_weights(expected_returns, forecast_cov)
        self.current_weights = new_weights
        self.weights_history.append(new_weights.copy())
        return new_weights

    def get_current_weights(self) -> np.ndarray:
        return self.current_weights

    def get_weights_history(self) -> list:
        return self.weights_history
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
from tools import Tools

class MarkRebalancing:
    def __init__(self, n_assets: int, gamma: float = 1.0, initial_weights: np.ndarray = None):
        self.n_assets = n_assets
        self.gamma = gamma
        if initial_weights is not None:
            if len(initial_weights) != n_assets:
                raise ValueError("Length of initial_weights must equal n_assets.")
            self.current_weights = np.array(initial_weights)
        else:
            self.current_weights = Tools.rand_weights(n_assets)
        self.weights_history = [self.current_weights.copy()]
        solvers.options['show_progress'] = False

    def compute_optimal_weights(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> tuple[np.ndarray, list, list]:
        n = self.n_assets
        S = opt.matrix(forecast_cov)
        pbar = opt.matrix(expected_returns)
        print("DEBUG: Expected Returns (pbar):")
        print(pbar)
        print("DEBUG: Covariance Matrix (S):")
        print(S)
        G = -opt.matrix(np.eye(n))
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
        N = 100
        mus = [10**(5.0 * t / N - 1.0) for t in range(N)]
        portfolios = []
        for mu in mus:
            sol = solvers.qp(mu * S, -pbar, G, h, A, b)
            portfolios.append(sol['x'])
        frontier_returns = [blas.dot(pbar, x) for x in portfolios]
        frontier_risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        m1 = np.polyfit(frontier_returns, frontier_risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        sol_min_var = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)
        w_opt = np.asarray(sol_min_var['x']).flatten()
        return w_opt, frontier_returns, frontier_risks

    def rebalance(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> np.ndarray:
        new_weights, _, _ = self.compute_optimal_weights(expected_returns, forecast_cov)
        self.current_weights = new_weights
        self.weights_history.append(new_weights.copy())
        return new_weights

    def get_current_weights(self) -> np.ndarray:
        return self.current_weights

    def get_weights_history(self) -> list:
        return self.weights_history
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
from tools import Tools

class MarkRebalancing:
    
    def __init__(self, n_assets: int, gamma: float = 1.0, initial_weights: np.ndarray = None):
        self.n_assets = n_assets
        self.gamma = gamma
        if initial_weights is not None:
            if len(initial_weights) != n_assets:
                raise ValueError("Length of initial_weights must equal n_assets.")
            self.current_weights = np.array(initial_weights)
        else:
            self.current_weights = Tools.rand_weights(n_assets)
        self.weights_history = [self.current_weights.copy()]
        solvers.options['show_progress'] = False


    def compute_optimal_weights(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> tuple[np.ndarray, list, list]:
        n = self.n_assets
        S = opt.matrix(forecast_cov)
        pbar = opt.matrix(expected_returns)
        print("DEBUG: Expected Returns (pbar):")
        print(pbar)
        print("DEBUG: Covariance Matrix (S):")
        print(S)
        G = -opt.matrix(np.eye(n))
        h = opt.matrix(0.0, (n, 1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
        N = 100
        mus = [10**(5.0 * t / N - 1.0) for t in range(N)]
        portfolios = []
        for mu in mus:
            sol = solvers.qp(mu * S, -pbar, G, h, A, b)
            portfolios.append(sol['x'])
        frontier_returns = [blas.dot(pbar, x) for x in portfolios]
        frontier_risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
        m1 = np.polyfit(frontier_returns, frontier_risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        sol_min_var = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)
        w_opt = np.asarray(sol_min_var['x']).flatten()
        return w_opt, frontier_returns, frontier_risks


    def rebalance(self, expected_returns: np.ndarray, forecast_cov: np.ndarray) -> np.ndarray:
        new_weights, _, _ = self.compute_optimal_weights(expected_returns, forecast_cov)
        self.current_weights = new_weights
        self.weights_history.append(new_weights.copy())
        return new_weights


    def get_current_weights(self) -> np.ndarray:
        return self.current_weights


    def get_weights_history(self) -> list:
        return self.weights_history