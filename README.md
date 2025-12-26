# ML OOP Pipeline

A clean, object-oriented machine learning pipeline built with Python for training and evaluating regression models.

## 📋 Description

This project demonstrates a well-structured, OOP-based approach to building machine learning pipelines. It provides a modular framework for loading datasets, training multiple regression models, and evaluating their performance using standardized metrics.

## 🎯 What It Does & Why

**What it does:**
- Loads and processes CSV datasets
- Applies data preprocessing (StandardScaler, Normalizer)
- Trains multiple regression models (Linear Regression, Decision Tree)
- Evaluates model performance using Mean Squared Error (MSE)
- Provides a clean, extensible architecture for adding new models and preprocessors

**Why it matters:**
- **Modularity**: Easy to add new models by extending the `BaseModel` class
- **Reusability**: Dataset and model classes can be reused across different projects
- **Best Practices**: Follows OOP principles and clean code standards
- **Scalability**: Simple to extend with new features like preprocessing, cross-validation, or additional metrics

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Favourez/ml-oop-pipeline.git
   cd ml-oop-pipeline
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run

Run the main pipeline script:

```bash
python main.py
```

This will:
1. Load the sample dataset from `ml_oop_pipeline/data/sample_data.csv`
2. Train models without preprocessing
3. Train models with StandardScaler preprocessing
4. Train models with Normalizer preprocessing
5. Display MSE for each configuration

## 📊 Example Usage

### Basic Usage

```python
from ml_oop_pipeline.datasets.dataset import Dataset
from ml_oop_pipeline.models.linear_regression_model import LinearRegressionModel
from ml_oop_pipeline.utils.metrics import mse

# Load dataset
dataset = Dataset("ml_oop_pipeline/data/sample_data.csv")
data = dataset.load()

# Prepare features and target
X = data[['feature']]
y = data['target']

# Train model
model = LinearRegressionModel()
model.train(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate
print("MSE:", mse(y, predictions))
```

### Using Preprocessing

Apply preprocessing to your data before training:

```python
from ml_oop_pipeline.datasets.dataset import Dataset
from ml_oop_pipeline.models.linear_regression_model import LinearRegressionModel
from ml_oop_pipeline.preprocessors.standard_scaler import StandardScaler
from ml_oop_pipeline.utils.metrics import mse

# Load dataset
dataset = Dataset("ml_oop_pipeline/data/sample_data.csv")
data = dataset.load()

X = data[['feature']]
y = data['target']

# Apply preprocessing
preprocessor = StandardScaler()
X_scaled = preprocessor.fit_transform(X)

# Train model
model = LinearRegressionModel()
model.train(X_scaled, y)

# Make predictions
predictions = model.predict(X_scaled)
print("MSE:", mse(y, predictions))
```

### Adding a New Model

Create a new model class that inherits from `BaseModel`:

```python
from ml_oop_pipeline.models.base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
```

### Adding a New Preprocessor

Create a new preprocessor class that inherits from `Preprocessor`:

```python
from ml_oop_pipeline.preprocessors.preprocessor import Preprocessor
from sklearn.preprocessing import RobustScaler

class RobustPreprocessor(Preprocessor):
    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def preprocess(self, X):
        return self.scaler.transform(X)
```

## 📦 Dependencies

- **pandas** (>=2.0.0) - Data manipulation and CSV handling
- **scikit-learn** (>=1.3.0) - Machine learning models and metrics
- **numpy** (>=1.24.0) - Numerical computations

See `requirements.txt` for the complete list.

## 📁 Project Structure

```
ml-oop-pipeline/
│
├── main.py                          # Main entry point
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
│
└── ml_oop_pipeline/                 # Main package
    ├── __init__.py
    │
    ├── data/                        # Data directory
    │   └── sample_data.csv          # Sample dataset
    │
    ├── datasets/                    # Dataset handling
    │   ├── __init__.py
    │   └── dataset.py               # Dataset class
    │
    ├── models/                      # Model implementations
    │   ├── __init__.py
    │   ├── base_model.py            # Abstract base model
    │   ├── linear_regression_model.py
    │   └── decision_tree_model.py
    │
    ├── preprocessors/               # Data preprocessing
    │   ├── __init__.py
    │   ├── preprocessor.py          # Abstract base preprocessor
    │   ├── standard_scaler.py       # StandardScaler implementation
    │   └── normalizer.py            # Normalizer implementation
    │
    └── utils/                       # Utility functions
        ├── __init__.py
        └── metrics.py               # Evaluation metrics
```

## 🔧 Features

- ✅ Object-oriented design with abstract base classes
- ✅ Easy model extensibility
- ✅ Data preprocessing with StandardScaler and Normalizer
- ✅ Clean separation of concerns
- ✅ Standardized model and preprocessor interfaces
- ✅ Reusable dataset loader
- ✅ Modular metrics system
- ✅ Polymorphism and inheritance throughout

## 🛣️ Roadmap

Future enhancements planned:
- [x] Add data preprocessing capabilities (StandardScaler, Normalizer)
- [ ] Add more preprocessors (RobustScaler, PCA)
- [ ] Implement cross-validation
- [ ] Add more evaluation metrics (R², MAE, RMSE)
- [ ] Support for classification models
- [ ] Model persistence (save/load trained models)
- [ ] Hyperparameter tuning utilities
- [ ] Visualization tools for model performance

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Favourez**
- GitHub: [@Favourez](https://github.com/Favourez)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

⭐ If you found this project helpful, please give it a star!
