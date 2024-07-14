from abc import abstractmethod
from sklearn.base import BaseEstimator

class BaseModel:
    """
    Base model.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Model training method.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict the probabilities.
        """
        pass