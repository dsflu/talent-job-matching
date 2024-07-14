from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model.
    """

    def __init__(self, **kwargs):
        model = LogisticRegression(**kwargs)
        super().__init__(model)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)