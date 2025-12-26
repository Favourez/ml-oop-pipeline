from ml_oop_pipeline.datasets.dataset import Dataset
from ml_oop_pipeline.models.linear_regression_model import LinearRegressionModel
from ml_oop_pipeline.models.decision_tree_model import DecisionTreeModel
from ml_oop_pipeline.preprocessors.standard_scaler import StandardScaler
from ml_oop_pipeline.preprocessors.normalizer import Normalizer
from ml_oop_pipeline.utils.metrics import mse


def run_pipeline(model, preprocessor=None):
    """
    Run the ML pipeline with optional preprocessing.

    Args:
        model: The model to train and evaluate
        preprocessor: Optional preprocessor to apply to the data
    """
    dataset = Dataset("ml_oop_pipeline/data/sample_data.csv")
    data = dataset.load()

    X = data[['feature']]
    y = data['target']

    # Apply preprocessing if provided
    if preprocessor:
        X = preprocessor.fit_transform(X)

    model.train(X, y)
    predictions = model.predict(X)

    print("MSE:", mse(y, predictions))


if __name__ == "__main__":
    print("=" * 60)
    print("Running ML Pipeline Without Preprocessing")
    print("=" * 60)

    print("\n1. Linear Regression (No Preprocessing)")
    run_pipeline(LinearRegressionModel())

    print("\n2. Decision Tree (No Preprocessing)")
    run_pipeline(DecisionTreeModel())

    print("\n" + "=" * 60)
    print("Running ML Pipeline With StandardScaler Preprocessing")
    print("=" * 60)

    print("\n3. Linear Regression (With StandardScaler)")
    run_pipeline(LinearRegressionModel(), StandardScaler())

    print("\n4. Decision Tree (With StandardScaler)")
    run_pipeline(DecisionTreeModel(), StandardScaler())

    print("\n" + "=" * 60)
    print("Running ML Pipeline With Normalizer Preprocessing")
    print("=" * 60)

    print("\n5. Linear Regression (With Normalizer)")
    run_pipeline(LinearRegressionModel(), Normalizer())

    print("\n6. Decision Tree (With Normalizer)")
    run_pipeline(DecisionTreeModel(), Normalizer())
