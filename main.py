from ml_oop_pipeline.datasets.dataset import Dataset
from ml_oop_pipeline.models.linear_regression_model import LinearRegressionModel
from ml_oop_pipeline.models.decision_tree_model import DecisionTreeModel
from ml_oop_pipeline.utils.metrics import mse

def run_pipeline(model):
    dataset = Dataset("ml_oop_pipeline/data/sample_data.csv")
    data = dataset.load()

    X = data[['feature']]
    y = data['target']

    model.train(X, y)
    predictions = model.predict(X)

    print("MSE:", mse(y, predictions))

if __name__ == "__main__":
    print("Running Linear Regression")
    run_pipeline(LinearRegressionModel())

    print("\nRunning Decision Tree")
    run_pipeline(DecisionTreeModel())
