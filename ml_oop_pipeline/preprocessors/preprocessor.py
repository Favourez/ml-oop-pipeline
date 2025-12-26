class Preprocessor:
    """Base class for data preprocessing."""
    
    def preprocess(self, X):
        """
        Preprocess the input data.
        
        Args:
            X: Input data to preprocess
            
        Returns:
            Preprocessed data
        """
        raise NotImplementedError("Preprocess method must be implemented")
    
    def fit(self, X):
        """
        Fit the preprocessor to the data.
        
        Args:
            X: Input data to fit
            
        Returns:
            self
        """
        raise NotImplementedError("Fit method must be implemented")
    
    def fit_transform(self, X):
        """
        Fit and transform the data.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        self.fit(X)
        return self.preprocess(X)

