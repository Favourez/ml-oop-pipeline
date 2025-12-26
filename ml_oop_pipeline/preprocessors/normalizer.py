from sklearn.preprocessing import MinMaxScaler
from ml_oop_pipeline.preprocessors.preprocessor import Preprocessor


class Normalizer(Preprocessor):
    """
    Transform features by scaling each feature to a given range (default 0-1).
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_range = feature_range
    
    def fit(self, X):
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Args:
            X: Training data
            
        Returns:
            self
        """
        self.scaler.fit(X)
        return self
    
    def preprocess(self, X):
        """
        Scale features to the specified range.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        return self.scaler.transform(X)

