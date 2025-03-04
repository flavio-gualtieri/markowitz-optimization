import numpy as np

class Tools:
    
    @staticmethod
    def set_seed(seed_value: int) -> None:
        """
        Set the random seed for reproducibility across numpy (and optionally Python's random module).
        """
        np.random.seed(seed_value)
        # You can also add:
        # import random
        # random.seed(seed_value)
    
    @staticmethod
    def rand_weights(n: int) -> np.ndarray:
        """
        Generates n random weights that sum to 1.

        Parameters
        ----------
        n : int
            Number of assets (dimensions).

        Returns
        -------
        np.ndarray
            Array of shape (n,) with random weights that sum to 1.
        """
        k = np.random.rand(n)
        return k / np.sum(k)
    
    @staticmethod
    def random_portfolio(returns: np.ndarray,
                         max_outlier_stdev: float = 2.0) -> tuple[float, float]:
        """
        Returns the mean and standard deviation of returns for a single random portfolio.

        Parameters
        ----------
        returns : np.ndarray
            2D array of shape (n_assets, n_obs). Each row corresponds to an asset,
            and each column is an observation in time.
        max_outlier_stdev : float
            A recursion cutoff to prevent extremely large standard deviations
            (keeps the random portfolio in a reasonable range).

        Returns
        -------
        mu : float
            Mean return of the random portfolio.
        sigma : float
            Standard deviation of the random portfolio.
        """
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
        """
        Generate many random portfolios and compute their mean & std.

        Parameters
        ----------
        returns : np.ndarray
            2D array of shape (n_assets, n_obs).
        n_portfolios : int
            Number of random portfolios to generate.

        Returns
        -------
        means : np.ndarray
            1D array of shape (n_portfolios,) with the mean returns of each portfolio.
        stds : np.ndarray
            1D array of shape (n_portfolios,) with the standard deviations of each portfolio.
        """
        means = []
        stds = []
        for _ in range(n_portfolios):
            mu, sigma = Tools.random_portfolio(returns)
            means.append(mu)
            stds.append(sigma)
    
        return np.array(means), np.array(stds)