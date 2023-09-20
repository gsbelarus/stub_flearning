import pandas as pd


class Metric:
    def __init__(self, name):
        self.name = name
        self.values = []

    def log_value(self, value):
        self.values.append(value)

    def get_dataframe(self):
        return pd.DataFrame(data={'value': self.values})
