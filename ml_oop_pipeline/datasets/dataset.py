import pandas as pd

class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = None

    def load(self):
        self.data = pd.read_csv(self.path)
        return self.data
