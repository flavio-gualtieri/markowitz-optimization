import requests
import pandas as pd
import numpy as np

class PolygonDataHandler:

    def __init__(self, api_key):
        self.api_key = api_key


    def slice_data_by_date(df, start_date, end_date):
        return df.loc[start_date:end_date]


    def fetch_data(self, ticker, start_date, end_date, multiplier=1, timespan='day'):
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
            df["t"] = pd.to_datetime(df["t"], unit='ms')
            df.set_index("t", inplace=True)
            return df
        else:
            raise Exception("Error fetching data from polygon.io: " + str(data))
    

    def compute_returns(self, df, price_column="c"):
        prices = df[price_column]
        returns = prices.pct_change().dropna()
        return returns.values


    def compute_variance(self, returns):
        return np.var(returns)


    def compute_moments(self, returns):
        E_r2 = np.mean(returns**2)
        E_r4 = np.mean(returns**4)
        return E_r2, E_r4


    def compute_gamma4(self, returns):
        E_r2, E_r4 = self.compute_moments(returns)
        if E_r2 == 0:
            return 3.0
        return E_r4 / (E_r2 ** 2)
