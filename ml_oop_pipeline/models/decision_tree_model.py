from sklearn.tree import DecisionTreeRegressor
from ml_oop_pipeline.models.base_model import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        self.model = DecisionTreeRegressor()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
