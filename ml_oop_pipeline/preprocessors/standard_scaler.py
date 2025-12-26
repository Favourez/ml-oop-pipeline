from sklearn.preprocessing import StandardScaler as SKStandardScaler
from ml_oop_pipeline.preprocessors.preprocessor import Preprocessor


class StandardScaler(Preprocessor):
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    
    def __init__(self):
        self.scaler = SKStandardScaler()
    
    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            X: Training data
            
        Returns:
            self
        """
        self.scaler.fit(X)
        return self
    
    def preprocess(self, X):
        """
        Perform standardization by centering and scaling.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        return self.scaler.transform(X)

