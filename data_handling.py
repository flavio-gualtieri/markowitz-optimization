# data_handling.py

import requests
import pandas as pd
import numpy as np

class PolygonDataHandler:
    def __init__(self, api_key):
        """
        Initialize the data handler with your polygon.io API key.
        """
        self.api_key = api_key

    # In data_handling.py, add a helper function:
    def slice_data_by_date(df, start_date, end_date):
        """
        Returns a slice of the DataFrame between start_date and end_date.
        """
        return df.loc[start_date:end_date]


    def fetch_data(self, ticker, start_date, end_date, multiplier=1, timespan='day'):
        """
        Retrieve aggregated price data from polygon.io for a given ticker and date range.

        Parameters:
            ticker (str): The ticker symbol (e.g., "AAPL").
            start_date (str): Start date in "YYYY-MM-DD" format.
            end_date (str): End date in "YYYY-MM-DD" format.
            multiplier (int): Aggregation multiplier (default 1).
            timespan (str): Aggregation timespan (e.g., "day", "minute").

        Returns:
            pd.DataFrame: A DataFrame with the aggregated data. The DataFrame index is set to the timestamp.
        """
        base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000,
            "apiKey": self.api_key
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if "results" in data:
            df = pd.DataFrame(data["results"])
            # Convert Unix timestamp (milliseconds) to datetime
            df["t"] = pd.to_datetime(df["t"], unit='ms')
            df.set_index("t", inplace=True)
            return df
        else:
            raise Exception("Error fetching data from polygon.io: " + str(data))
    
    def compute_returns(self, df, price_column="c"):
        """
        Compute simple returns from a DataFrame containing price data.

        Parameters:
            df (pd.DataFrame): DataFrame with price data.
            price_column (str): Column name for the closing price (default is "c").

        Returns:
            np.ndarray: Array of returns.
        """
        prices = df[price_column]
        returns = prices.pct_change().dropna()
        return returns.values

    def compute_variance(self, returns):
        """
        Compute the in-sample variance (empirical variance) from returns.

        Parameters:
            returns (np.ndarray): Array of returns.

        Returns:
            float: The variance of the returns.
        """
        return np.var(returns)

    def compute_moments(self, returns):
        """
        Compute the second and fourth moments of the returns.

        Second moment: E[r^2] (which, if the mean is zero, is the variance).
        Fourth moment: E[r^4].

        Parameters:
            returns (np.ndarray): Array of returns.

        Returns:
            tuple: (E[r^2], E[r^4])
        """
        E_r2 = np.mean(returns**2)
        E_r4 = np.mean(returns**4)
        return E_r2, E_r4

    def compute_gamma4(self, returns):
        """
        Compute the fourth standardized moment (gamma4), defined as:
            gamma4 = E[r^4] / (E[r^2])^2

        Parameters:
            returns (np.ndarray): Array of returns.

        Returns:
            float: The fourth standardized moment.
        """
        E_r2, E_r4 = self.compute_moments(returns)
        if E_r2 == 0:
            return 3.0  # Default value for a normal distribution
        return E_r4 / (E_r2 ** 2)

# Example usage:
if __name__ == "__main__":
    # Replace with your actual polygon.io API key
    api_key = "YOUR_POLYGON_API_KEY"
    handler = PolygonDataHandler(api_key)
    
    # Fetch data for a given ticker and date range
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    df = handler.fetch_data(ticker, start_date, end_date)
    
    # Compute returns and moments
    returns = handler.compute_returns(df)
    variance = handler.compute_variance(returns)
    gamma4 = handler.compute_gamma4(returns)
    
    print("Variance:", variance)
    print("Fourth Standardized Moment (Gamma4):", gamma4)
