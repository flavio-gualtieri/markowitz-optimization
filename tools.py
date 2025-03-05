import numpy as np

class Tools:
    
    @staticmethod
    def set_seed(seed_value: int) -> None:
        np.random.seed(seed_value)

    
    @staticmethod
    def rand_weights(n: int) -> np.ndarray:
        k = np.random.rand(n)
        return k / np.sum(k)
    

    @staticmethod
    def random_portfolio(returns: np.ndarray,
                         max_outlier_stdev: float = 2.0) -> tuple[float, float]:
        # Mean returns of each asset
        p = np.asmatrix(np.mean(returns, axis=1))
        # Random weights
        w = np.asmatrix(Tools.rand_weights(returns.shape[0]))
        # Covariance matrix of returns
        C = np.asmatrix(np.cov(returns))
        
        mu = float(w * p.T)
        sigma = float(np.sqrt(w * C * w.T))
    
        # This recursion reduces outliers to keep plots (and portfolios) pretty
        if sigma > max_outlier_stdev:
            return Tools.random_portfolio(returns, max_outlier_stdev)
        return mu, sigma
    

    @staticmethod
    def compute_portfolios(returns: np.ndarray,
                           n_portfolios: int = 500) -> tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        for _ in range(n_portfolios):
            mu, sigma = Tools.random_portfolio(returns)
            means.append(mu)
            stds.append(sigma)
    
        return np.array(means), np.array(stds)